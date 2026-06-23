"""HTTP transport for MCP server."""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from aiohttp import web
from homeassistant.components.http import HomeAssistantView
from homeassistant.core import HomeAssistant
from mcp.server import Server
from multidict import CIMultiDictProxy

from .completions import complete
from .const import DOMAIN
from .prompts import get_prompt, get_prompts
from .resources import get_resources, read_resource
from .tools import call_tool, get_tool_schemas

_LOGGER = logging.getLogger(__name__)

_SSE_KEEPALIVE_SECONDS = 30

# Session idle TTL and notification-buffer retention (one knob): a session is reclaimed
# this many seconds after the last *client* activity (POST or GET keepalive — not server
# notifications), and its coalesced pending notifications live exactly that long, so a
# brief GET reconnect within the window resumes without losing updates.
# ponytail: module constant for now; promote to a config-flow option later.
_SESSION_RETENTION_SECONDS = 300

# Readable annotation for aiohttp request headers (case-insensitive multidict).
Headers = CIMultiDictProxy[str]


def _integration_loaded(hass: HomeAssistant) -> bool:
    """Return True when the config entry is active.

    HA's HTTP stack has no public way to unregister a view, so registered
    views survive `async_unload_entry`. async_unload_entry clears
    `hass.data[DOMAIN]`, so we gate requests on it being populated — when
    the user disables the integration, requests return 503 immediately
    instead of continuing to succeed until the next HA restart (#37).
    """
    return bool(hass.data.get(DOMAIN))


def _service_unavailable() -> web.Response:
    """Build a 503 response for requests made while the integration is disabled."""
    return web.json_response(
        {
            "error": "service_unavailable",
            "error_description": "MCP Server integration is disabled",
        },
        status=503,
    )


def _sessions(hass: HomeAssistant) -> dict[str, dict[str, Any]]:
    """Return the per-client MCP session registry, creating it on first use."""
    return hass.data[DOMAIN].setdefault("mcp_sessions", {})


def notify_resource_updated(hass: HomeAssistant, uri: str) -> None:
    """Mark uri dirty for every subscribed session and wake any live stream.

    Coalesced by uri: repeated updates to the same uri collapse to one pending
    marker (the client re-reads current state on delivery), bounding the buffer
    by the number of distinct subscribed uris.
    """
    for session in _sessions(hass).values():
        if uri in session["uris"]:
            session["pending"][uri] = None
            session["waiter"].set()


def _new_session() -> dict[str, Any]:
    """Create an empty session: subscriptions, coalesced pending buffer, liveness."""
    return {
        "pending": {},
        "uris": set(),
        "last_seen": time.monotonic(),
        "stream_attached": False,
        "waiter": asyncio.Event(),
        "closing": False,
    }


def _close_session(session: dict[str, Any]) -> None:
    """Signal an attached SSE stream (if any) to end and stop further delivery."""
    session["closing"] = True
    session["waiter"].set()


def _invalid_params(msg_id: Any) -> dict[str, Any]:
    """Build a JSON-RPC -32602 error for a missing or non-string uri."""
    return {
        "jsonrpc": "2.0",
        "error": {"code": -32602, "message": "Invalid params: uri must be a non-empty string"},
        "id": msg_id,
    }


def _get_issuer(request: web.Request) -> str | None:
    """Get the OIDC issuer URL from the request, or None if unavailable."""
    try:
        from custom_components.oidc_provider.token_validator import (
            get_issuer_from_request,
        )

        return get_issuer_from_request(request)
    except ImportError:
        return None


def _get_protected_resource_metadata(base_url: str) -> dict[str, Any]:
    """Generate OAuth 2.0 Protected Resource Metadata (RFC 9728)."""
    return {
        "resource": f"{base_url}/api/mcp",
        "authorization_servers": [f"{base_url}/oidc"],
        "bearer_methods_supported": ["header"],
        "resource_signing_alg_values_supported": ["RS256"],
        "resource_documentation": f"{base_url}/api/mcp",
    }


@dataclass(frozen=True, slots=True)
class MCPResult:
    """A JSON-RPC result body plus HTTP response headers for post() to apply."""

    body: dict[str, Any]
    headers: dict[str, str]


class MCPProtectedResourceMetadataView(HomeAssistantView):
    """OAuth 2.0 Protected Resource Metadata endpoint (RFC 9728) at root."""

    url = "/.well-known/oauth-protected-resource"
    name = "api:mcp:metadata:root"
    requires_auth = False

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the metadata view."""
        self.hass = hass

    async def get(self, request: web.Request) -> web.Response:
        """Return protected resource metadata."""
        if not _integration_loaded(self.hass):
            return _service_unavailable()
        base_url = _get_issuer(request)
        if base_url is None:
            return web.json_response({"error": "OIDC provider not available"}, status=404)
        metadata = _get_protected_resource_metadata(base_url)
        return web.json_response(metadata)


class MCPSubpathProtectedResourceMetadataView(HomeAssistantView):
    """OAuth 2.0 Protected Resource Metadata endpoint (RFC 9728) with /mcp suffix."""

    url = "/.well-known/oauth-protected-resource/api/mcp"
    name = "api:mcp:metadata:mcp"
    requires_auth = False

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the metadata view."""
        self.hass = hass

    async def get(self, request: web.Request) -> web.Response:
        """Return protected resource metadata with /mcp suffix."""
        if not _integration_loaded(self.hass):
            return _service_unavailable()
        base_url = _get_issuer(request)
        if base_url is None:
            return web.json_response({"error": "OIDC provider not available"}, status=404)
        metadata = _get_protected_resource_metadata(base_url)
        return web.json_response(metadata)


class MCPEndpointView(HomeAssistantView):
    """MCP HTTP endpoint view."""

    url = "/api/mcp"
    name = "api:mcp"
    requires_auth = False

    def __init__(
        self, hass: HomeAssistant, server: Server, native_auth_enabled: bool = False
    ) -> None:
        """Initialize the MCP endpoint."""
        self.hass = hass
        self.server = server
        self.native_auth_enabled = native_auth_enabled

    async def _validate_token(self, request: web.Request) -> dict[str, Any] | None:
        """Validate the bearer token via OIDC (if available) then native HA auth."""
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return None

        token = auth_header[7:]  # Remove "Bearer " prefix

        # 1. Try OIDC first
        try:
            from custom_components.oidc_provider.token_validator import (
                get_issuer_from_request,
                validate_access_token,
            )

            expected_issuer = get_issuer_from_request(request)
            # This MCP server is the protected resource (RFC 8707); its canonical
            # URI is the resource a compliant client (e.g. Claude) binds the token
            # to. Require the token's aud to match it.
            expected_audience = f"{expected_issuer}/api/mcp"
            try:
                result = validate_access_token(
                    self.hass, token, expected_issuer, expected_audience=expected_audience
                )
            except TypeError:
                # OIDC provider predates resource-aware validation; fall back to
                # the legacy signature so an un-upgraded provider still works.
                result = validate_access_token(self.hass, token, expected_issuer)
            if result is not None:
                return result
        except ImportError as e:
            _LOGGER.debug("OIDC provider not available: %s", e)

        # 2. Fall back to native HA auth (Long-Lived Access Tokens)
        if self.native_auth_enabled:
            refresh_token = self.hass.auth.async_validate_access_token(token)
            if refresh_token is not None:
                return {"sub": refresh_token.user.id}

        return None

    def _unauthorized(self, request: web.Request) -> web.Response:
        """Build a 401 response with an RFC 9728 WWW-Authenticate challenge."""
        base_url = _get_issuer(request)
        if base_url is not None:
            resource_metadata_url = f"{base_url}/.well-known/oauth-protected-resource/api/mcp"
            www_authenticate = (
                f'Bearer realm="MCP Server", resource_metadata="{resource_metadata_url}"'
            )
        else:
            www_authenticate = 'Bearer realm="Home Assistant MCP Server"'
        return web.json_response(
            {
                "error": "invalid_token",
                "error_description": "Invalid or missing token",
            },
            status=401,
            headers={"WWW-Authenticate": www_authenticate},
        )

    async def get(self, request: web.Request) -> web.StreamResponse:
        """Long-lived SSE stream draining a session's coalesced notification buffer."""
        if not _integration_loaded(self.hass):
            return _service_unavailable()
        if not await self._validate_token(request):
            return self._unauthorized(request)
        sid = request.headers.get("Mcp-Session-Id")
        session = _sessions(self.hass).get(sid)
        if session is None:
            return web.Response(status=404)
        if session["stream_attached"]:
            # One SSE stream per session (matches the reference SDK).
            return web.Response(status=409)
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)
        session["stream_attached"] = True
        session["last_seen"] = time.monotonic()
        waiter = session["waiter"]
        try:
            while True:
                waiter.clear()
                if session["closing"]:
                    break
                pending = session["pending"]
                if pending:
                    uris = list(pending)
                    pending.clear()
                    for uri in uris:
                        frame = {
                            "jsonrpc": "2.0",
                            "method": "notifications/resources/updated",
                            "params": {"uri": uri},
                        }
                        await response.write(f"data: {json.dumps(frame)}\n\n".encode())
                    continue
                try:
                    await asyncio.wait_for(waiter.wait(), timeout=_SSE_KEEPALIVE_SECONDS)
                except TimeoutError:
                    # ponytail: fixed keepalive; raise _SSE_KEEPALIVE_SECONDS if a proxy times out.
                    await response.write(b": keepalive\n\n")
                    session["last_seen"] = time.monotonic()
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            # Keep the session alive across reconnects; only detach the stream.
            session["stream_attached"] = False
            session["last_seen"] = time.monotonic()
        return response

    async def delete(self, request: web.Request) -> web.Response:
        """Terminate a session (spec DELETE): drop it and close any live stream."""
        if not _integration_loaded(self.hass):
            return _service_unavailable()
        if not await self._validate_token(request):
            return self._unauthorized(request)
        session = _sessions(self.hass).pop(request.headers.get("Mcp-Session-Id"), None)
        if session is None:
            return web.Response(status=404)
        _close_session(session)
        return web.Response(status=200)

    async def post(self, request: web.Request) -> web.Response:
        """Handle POST requests for MCP messages."""
        if not _integration_loaded(self.hass):
            return _service_unavailable()

        # Validate token
        token_payload = await self._validate_token(request)
        if not token_payload:
            return self._unauthorized(request)

        body = None
        try:
            # Parse JSON-RPC message
            body = await request.json()
            _LOGGER.debug("Received MCP request: %s", body)

            method = body.get("method") if isinstance(body, dict) else None
            if method != "initialize":
                sid = request.headers.get("Mcp-Session-Id")
                if sid is None:
                    # Spec: a required session header that is absent is a bad request.
                    return web.Response(status=400)
                session = _sessions(self.hass).get(sid)
                if session is None:
                    # Spec: a session the server no longer has is 404; client re-inits.
                    return web.Response(status=404)
                session["last_seen"] = time.monotonic()

            response_data = await self._handle_message(body, request.headers)

            if response_data is None:
                # Notification - return 202 Accepted
                return web.Response(status=202)
            if isinstance(response_data, MCPResult):
                return web.json_response(response_data.body, headers=response_data.headers)
            return web.json_response(response_data)

        except Exception as e:
            _LOGGER.error("Error handling MCP request: %s", e, exc_info=True)
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}",
                    },
                    "id": body.get("id") if isinstance(body, dict) else None,
                },
                status=500,
            )

    async def _handle_message(
        self, message: dict[str, Any], headers: Headers
    ) -> dict[str, Any] | MCPResult | None:
        """Handle a JSON-RPC message."""
        method = message.get("method")
        params = message.get("params", {})
        msg_id = message.get("id")

        # Handle initialization
        if method == "initialize":
            sessions = _sessions(self.hass)
            now = time.monotonic()
            for stale in [
                s for s, v in sessions.items() if now - v["last_seen"] > _SESSION_RETENTION_SECONDS
            ]:
                _close_session(sessions.pop(stale))
            session_id = uuid.uuid4().hex
            sessions[session_id] = _new_session()
            return MCPResult(
                body={
                    "jsonrpc": "2.0",
                    "result": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {
                            "tools": {},
                            "resources": {"subscribe": True},
                            "prompts": {},
                        },
                        "serverInfo": {
                            "name": "home-assistant-mcp-server",
                            "version": "0.1.0",
                        },
                    },
                    "id": msg_id,
                },
                headers={"Mcp-Session-Id": session_id},
            )

        # Handle tools/list
        if method == "tools/list":
            tools = await self._get_tools()
            return {
                "jsonrpc": "2.0",
                "result": {"tools": tools},
                "id": msg_id,
            }

        # Handle tools/call
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {})

            result = await self._call_tool(name, arguments)
            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": msg_id,
            }

        # Handle resources/list
        if method == "resources/list":
            result = get_resources()
            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": msg_id,
            }

        # Handle resources/read
        if method == "resources/read":
            uri = params.get("uri", "")
            contents = await read_resource(self.hass, uri)
            return {
                "jsonrpc": "2.0",
                "result": {"contents": contents},
                "id": msg_id,
            }

        # Handle resources/subscribe
        if method == "resources/subscribe":
            uri = params.get("uri")
            if not isinstance(uri, str) or not uri:
                return _invalid_params(msg_id)
            _sessions(self.hass)[headers["Mcp-Session-Id"]]["uris"].add(uri)
            return {"jsonrpc": "2.0", "result": {}, "id": msg_id}

        # Handle resources/unsubscribe
        if method == "resources/unsubscribe":
            uri = params.get("uri")
            if not isinstance(uri, str) or not uri:
                return _invalid_params(msg_id)
            session = _sessions(self.hass)[headers["Mcp-Session-Id"]]
            session["uris"].discard(uri)
            session["pending"].pop(uri, None)
            return {"jsonrpc": "2.0", "result": {}, "id": msg_id}

        # Handle prompts/list
        if method == "prompts/list":
            prompts = get_prompts()
            return {
                "jsonrpc": "2.0",
                "result": {"prompts": prompts},
                "id": msg_id,
            }

        # Handle prompts/get
        if method == "prompts/get":
            name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = await get_prompt(self.hass, name, arguments)
            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": msg_id,
            }

        # Handle completion/complete
        if method == "completion/complete":
            ref = params.get("ref", {})
            argument = params.get("argument", {})
            result = await complete(self.hass, ref, argument)
            return {
                "jsonrpc": "2.0",
                "result": {"completion": result},
                "id": msg_id,
            }

        # Unknown method
        if msg_id is not None:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
                "id": msg_id,
            }

        return None

    async def _get_tools(self) -> list[dict[str, Any]]:
        """Get available tools."""
        return get_tool_schemas()

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool by name."""
        return await call_tool(self.hass, name, arguments)
