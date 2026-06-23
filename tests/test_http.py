"""Tests for HTTP transport, auth, and JSON-RPC routing."""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aiohttp import web as aiohttp_web
from aiohttp.test_utils import TestClient, TestServer
from custom_components.oidc_provider.token_validator import get_issuer_from_request

from custom_components.mcp_server_http_transport import async_unload_entry
from custom_components.mcp_server_http_transport.const import DOMAIN
from custom_components.mcp_server_http_transport.http import (
    _SESSION_RETENTION_SECONDS,
    MCPEndpointView,
    MCPProtectedResourceMetadataView,
    MCPSubpathProtectedResourceMetadataView,
    _close_session,
    _get_issuer,
    _get_protected_resource_metadata,
    _new_session,
    notify_resource_updated,
)

_TEST_SID = "test-session-id"


def test_get_base_url_with_forwarded_headers():
    """Test get_issuer_from_request with X-Forwarded headers (proxy setup)."""
    request = Mock()
    request.headers = {
        "X-Forwarded-Proto": "https",
        "X-Forwarded-Host": "example.com",
    }
    request.url.origin.return_value = "http://localhost:8123"

    result = get_issuer_from_request(request)

    assert result == "https://example.com"
    request.url.origin.assert_not_called()


def test_get_base_url_without_forwarded_headers():
    """Test get_issuer_from_request without X-Forwarded headers (direct connection)."""
    request = Mock()
    request.headers = {}
    request.url.origin.return_value = "http://192.168.1.100:8123"

    result = get_issuer_from_request(request)

    assert result == "http://192.168.1.100:8123"
    request.url.origin.assert_called_once()


def test_get_base_url_with_partial_forwarded_headers():
    """Test get_issuer_from_request with only one X-Forwarded header (should use fallback)."""
    request = Mock()
    request.headers = {
        "X-Forwarded-Proto": "https",
    }
    request.url.origin.return_value = "http://localhost:8123"

    result = get_issuer_from_request(request)

    assert result == "http://localhost:8123"
    request.url.origin.assert_called_once()


def test_get_issuer_returns_none_when_oidc_unavailable():
    """Test _get_issuer returns None when oidc_provider import fails."""
    import sys

    request = Mock()
    # Temporarily remove the mocked oidc module so the import raises ImportError
    saved = sys.modules.pop("custom_components.oidc_provider.token_validator", None)
    saved_parent = sys.modules.pop("custom_components.oidc_provider", None)
    try:
        result = _get_issuer(request)
        assert result is None
    finally:
        if saved is not None:
            sys.modules["custom_components.oidc_provider.token_validator"] = saved
        if saved_parent is not None:
            sys.modules["custom_components.oidc_provider"] = saved_parent


def test_get_protected_resource_metadata():
    """Test _get_protected_resource_metadata returns correct structure."""
    base_url = "https://homeassistant.local"

    metadata = _get_protected_resource_metadata(base_url)

    assert metadata["resource"] == f"{base_url}/api/mcp"
    assert metadata["authorization_servers"] == [f"{base_url}/oidc"]
    assert metadata["bearer_methods_supported"] == ["header"]
    assert metadata["resource_signing_alg_values_supported"] == ["RS256"]
    assert metadata["resource_documentation"] == f"{base_url}/api/mcp"


class TestMCPProtectedResourceMetadataView:
    """Test the MCP protected resource metadata view at root."""

    async def test_get_returns_metadata(self):
        """Test GET returns protected resource metadata."""
        request = Mock()
        request.headers = {}
        request.url.origin.return_value = "https://homeassistant.local"

        view = MCPProtectedResourceMetadataView(Mock())
        response = await view.get(request)

        assert response.status == 200
        assert response.content_type == "application/json"

        body = json.loads(response.body)
        assert body["resource"] == "https://homeassistant.local/api/mcp"
        assert body["authorization_servers"] == ["https://homeassistant.local/oidc"]

    async def test_get_with_forwarded_headers(self):
        """Test GET with X-Forwarded headers."""
        request = Mock()
        request.headers = {
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "example.com",
        }

        view = MCPProtectedResourceMetadataView(Mock())
        response = await view.get(request)

        body = json.loads(response.body)
        assert body["resource"] == "https://example.com/api/mcp"

    async def test_get_returns_404_when_oidc_unavailable(self):
        """Test GET returns 404 when OIDC provider is not installed."""
        request = Mock()

        view = MCPProtectedResourceMetadataView(Mock())
        with patch(
            "custom_components.mcp_server_http_transport.http._get_issuer",
            return_value=None,
        ):
            response = await view.get(request)

        assert response.status == 404


class TestMCPSubpathProtectedResourceMetadataView:
    """Test the MCP protected resource metadata view with /mcp suffix."""

    async def test_get_returns_metadata(self):
        """Test GET returns protected resource metadata."""
        request = Mock()
        request.headers = {}
        request.url.origin.return_value = "https://homeassistant.local"

        view = MCPSubpathProtectedResourceMetadataView(Mock())
        response = await view.get(request)

        assert response.status == 200
        body = json.loads(response.body)
        assert body["resource"] == "https://homeassistant.local/api/mcp"

    async def test_get_returns_404_when_oidc_unavailable(self):
        """Test GET returns 404 when OIDC provider is not installed."""
        request = Mock()

        view = MCPSubpathProtectedResourceMetadataView(Mock())
        with patch(
            "custom_components.mcp_server_http_transport.http._get_issuer",
            return_value=None,
        ):
            response = await view.get(request)

        assert response.status == 404


class TestMCPEndpointView:
    """Test the MCP endpoint view: auth, routing, and error handling."""

    @pytest.fixture
    def mock_server(self):
        """Create a mock MCP server."""
        return Mock()

    @pytest.fixture
    def mock_hass(self):
        """Create a mock Home Assistant instance."""
        hass = Mock()
        hass.states = Mock()
        hass.services = Mock()
        hass.data = {DOMAIN: {"mcp_sessions": {_TEST_SID: _new_session()}}}
        return hass

    @pytest.fixture
    def view(self, mock_hass, mock_server):
        """Create an MCPEndpointView instance."""
        return MCPEndpointView(mock_hass, mock_server)

    async def test_post_without_token_returns_401(self, view):
        """Test POST without Authorization header returns 401."""
        request = Mock()
        request.headers = {}
        request.url.origin.return_value = "https://homeassistant.local"

        response = await view.post(request)

        assert response.status == 401
        body = json.loads(response.body)
        assert body["error"] == "invalid_token"
        assert "WWW-Authenticate" in response.headers

    async def test_post_with_invalid_token_returns_401(self, view):
        """Test POST with invalid token returns 401."""
        request = Mock()
        request.headers = {"Authorization": "Bearer invalid_token"}
        request.url.origin.return_value = "https://homeassistant.local"

        with patch.object(view, "_validate_token", return_value=None):
            response = await view.post(request)

        assert response.status == 401
        body = json.loads(response.body)
        assert body["error"] == "invalid_token"

    async def test_post_initialize_request(self, view):
        """Test POST with initialize request."""
        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token"}
        request.json = AsyncMock(return_value={"jsonrpc": "2.0", "method": "initialize", "id": 1})

        with patch.object(view, "_validate_token", return_value={"sub": "user123"}):
            response = await view.post(request)

        assert response.status == 200
        body = json.loads(response.body)
        assert body["jsonrpc"] == "2.0"
        assert body["result"]["protocolVersion"] == "2025-03-26"
        assert body["result"]["serverInfo"]["name"] == "home-assistant-mcp-server"
        assert body["id"] == 1
        assert "Mcp-Session-Id" in response.headers
        assert response.headers["Mcp-Session-Id"] in view.hass.data[DOMAIN]["mcp_sessions"]

    async def test_post_initialize_advertises_capabilities(self, view):
        """Test POST initialize advertises resources and prompts capabilities."""
        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token"}
        request.json = AsyncMock(return_value={"jsonrpc": "2.0", "method": "initialize", "id": 21})

        with patch.object(view, "_validate_token", return_value={"sub": "user123"}):
            response = await view.post(request)

        body = json.loads(response.body)
        capabilities = body["result"]["capabilities"]
        assert "tools" in capabilities
        assert capabilities["resources"] == {"subscribe": True}
        assert "prompts" in capabilities

    async def test_post_tools_list_request(self, view):
        """Test POST with tools/list request."""
        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token", "Mcp-Session-Id": _TEST_SID}
        request.json = AsyncMock(return_value={"jsonrpc": "2.0", "method": "tools/list", "id": 2})

        with patch.object(view, "_validate_token", return_value={"sub": "user123"}):
            response = await view.post(request)

        assert response.status == 200
        body = json.loads(response.body)
        assert body["jsonrpc"] == "2.0"
        assert "tools" in body["result"]
        assert len(body["result"]["tools"]) == 74
        tool_names = [t["name"] for t in body["result"]["tools"]]
        assert "get_state" in tool_names
        assert "call_service" in tool_names
        assert "knx_recent_telegrams" in tool_names
        assert "knx_create_entity" in tool_names
        assert "list_entities" in tool_names
        assert "get_error_log" in tool_names
        assert "restart_ha" in tool_names
        assert "get_system_status" in tool_names
        assert "get_statistics" in tool_names
        assert "get_camera_image" in tool_names
        assert "get_image_file" in tool_names
        assert "list_labels" in tool_names
        assert "batch_get_state" in tool_names

    async def test_post_unknown_method_returns_error(self, view):
        """Test POST with unknown method returns error."""
        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token", "Mcp-Session-Id": _TEST_SID}
        request.json = AsyncMock(
            return_value={"jsonrpc": "2.0", "method": "unknown_method", "id": 9}
        )

        with patch.object(view, "_validate_token", return_value={"sub": "user123"}):
            response = await view.post(request)

        assert response.status == 200
        body = json.loads(response.body)
        assert "error" in body
        assert body["error"]["code"] == -32601
        assert "Method not found" in body["error"]["message"]

    async def test_post_notification_returns_202(self, view):
        """Test POST with notification (no id) returns 202."""
        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token", "Mcp-Session-Id": _TEST_SID}
        request.json = AsyncMock(return_value={"jsonrpc": "2.0", "method": "some_notification"})

        with patch.object(view, "_validate_token", return_value={"sub": "user123"}):
            response = await view.post(request)

        assert response.status == 202

    async def test_validate_token_without_bearer_prefix(self, view):
        """Test _validate_token without Bearer prefix returns None."""
        request = Mock()
        request.headers = {"Authorization": "invalid_format"}

        result = await view._validate_token(request)

        assert result is None

    async def test_post_tools_call_unknown_tool(self, view):
        """Test POST with tools/call for unknown tool."""
        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token", "Mcp-Session-Id": _TEST_SID}
        request.json = AsyncMock(
            return_value={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "unknown_tool", "arguments": {}},
                "id": 10,
            }
        )
        request.url.origin.return_value = "https://homeassistant.local"

        with patch.object(view, "_validate_token", return_value={"sub": "user123"}):
            response = await view.post(request)

        assert response.status == 500
        body = json.loads(response.body)
        assert "error" in body
        assert "Unknown tool" in body["error"]["message"]

    async def test_post_without_session_header_returns_400(self, view):
        """Non-initialize POST with no Mcp-Session-Id header is a bad request (spec)."""
        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token"}
        request.json = AsyncMock(return_value={"jsonrpc": "2.0", "method": "tools/list", "id": 3})

        with patch.object(view, "_validate_token", return_value={"sub": "user123"}):
            response = await view.post(request)

        assert response.status == 400

    async def test_post_unknown_session_returns_404(self, view):
        """Non-initialize POST with an unknown Mcp-Session-Id is 404 (client re-inits)."""
        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token", "Mcp-Session-Id": "nope"}
        request.json = AsyncMock(return_value={"jsonrpc": "2.0", "method": "tools/list", "id": 3})

        with patch.object(view, "_validate_token", return_value={"sub": "user123"}):
            response = await view.post(request)

        assert response.status == 404

    async def test_post_subscribe_tracks_uri(self, view):
        """resources/subscribe adds the URI to the session's subscription set."""
        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token", "Mcp-Session-Id": _TEST_SID}
        request.json = AsyncMock(
            return_value={
                "jsonrpc": "2.0",
                "method": "resources/subscribe",
                "params": {"uri": "hass://x"},
                "id": 4,
            }
        )

        with patch.object(view, "_validate_token", return_value={"sub": "user123"}):
            response = await view.post(request)

        assert response.status == 200
        sessions = view.hass.data[DOMAIN]["mcp_sessions"]
        assert "hass://x" in sessions[_TEST_SID]["uris"]

    async def test_post_unsubscribe_drops_uri(self, view):
        """resources/unsubscribe removes the URI from the session's set."""
        view.hass.data[DOMAIN]["mcp_sessions"][_TEST_SID]["uris"].add("hass://x")
        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token", "Mcp-Session-Id": _TEST_SID}
        request.json = AsyncMock(
            return_value={
                "jsonrpc": "2.0",
                "method": "resources/unsubscribe",
                "params": {"uri": "hass://x"},
                "id": 5,
            }
        )

        with patch.object(view, "_validate_token", return_value={"sub": "user123"}):
            response = await view.post(request)

        assert response.status == 200
        sessions = view.hass.data[DOMAIN]["mcp_sessions"]
        assert "hass://x" not in sessions[_TEST_SID]["uris"]

    async def test_post_subscribe_invalid_uri_returns_error(self, view):
        """resources/subscribe without a uri returns -32602 and subscribes nothing."""
        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token", "Mcp-Session-Id": _TEST_SID}
        request.json = AsyncMock(
            return_value={"jsonrpc": "2.0", "method": "resources/subscribe", "params": {}, "id": 6}
        )

        with patch.object(view, "_validate_token", return_value={"sub": "user123"}):
            response = await view.post(request)

        assert response.status == 200
        body = json.loads(response.body)
        assert body["error"]["code"] == -32602
        assert view.hass.data[DOMAIN]["mcp_sessions"][_TEST_SID]["uris"] == set()

    async def test_post_unsubscribe_invalid_uri_returns_error(self, view):
        """resources/unsubscribe with a non-string uri returns -32602."""
        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token", "Mcp-Session-Id": _TEST_SID}
        request.json = AsyncMock(
            return_value={
                "jsonrpc": "2.0",
                "method": "resources/unsubscribe",
                "params": {"uri": 123},
                "id": 7,
            }
        )

        with patch.object(view, "_validate_token", return_value={"sub": "user123"}):
            response = await view.post(request)

        assert response.status == 200
        body = json.loads(response.body)
        assert body["error"]["code"] == -32602

    async def test_delete_terminates_session(self, view):
        """DELETE drops the session and signals any live stream to close."""
        sessions = view.hass.data[DOMAIN]["mcp_sessions"]
        session = sessions[_TEST_SID]
        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token", "Mcp-Session-Id": _TEST_SID}

        with patch.object(view, "_validate_token", return_value={"sub": "user123"}):
            response = await view.delete(request)

        assert response.status == 200
        assert _TEST_SID not in sessions
        assert session["closing"] is True

    async def test_delete_unknown_session_returns_404(self, view):
        """DELETE for an unknown session id is rejected."""
        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token", "Mcp-Session-Id": "nope"}

        with patch.object(view, "_validate_token", return_value={"sub": "user123"}):
            response = await view.delete(request)

        assert response.status == 404

    async def test_initialize_sweeps_idle_sessions(self, view):
        """initialize reclaims sessions idle past the retention window; fresh ones survive."""
        sessions = view.hass.data[DOMAIN]["mcp_sessions"]
        sessions.clear()
        stale = _new_session()
        stale["last_seen"] = time.monotonic() - 10 * _SESSION_RETENTION_SECONDS
        sessions["stale"] = stale
        sessions["fresh"] = _new_session()

        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token"}
        request.json = AsyncMock(return_value={"jsonrpc": "2.0", "method": "initialize", "id": 1})
        with patch.object(view, "_validate_token", return_value={"sub": "user123"}):
            response = await view.post(request)
        new_sid = response.headers["Mcp-Session-Id"]

        assert "stale" not in sessions  # reclaimed
        assert stale["closing"] is True  # and signaled to close
        assert "fresh" in sessions  # kept
        assert new_sid in sessions  # created

    async def test_get_without_session_returns_404(self, view):
        """GET without a known Mcp-Session-Id is rejected."""
        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token"}

        with patch.object(view, "_validate_token", return_value={"sub": "user123"}):
            response = await view.get(request)

        assert response.status == 404

    async def test_notify_resource_updated_buffers_for_subscribers_only(self, view):
        """notify_resource_updated marks pending only for sessions subscribed to the URI."""
        sessions = view.hass.data[DOMAIN]["mcp_sessions"]
        sessions[_TEST_SID]["uris"].add("hass://x")

        notify_resource_updated(view.hass, "hass://x")
        assert "hass://x" in sessions[_TEST_SID]["pending"]

        notify_resource_updated(view.hass, "hass://y")
        assert "hass://y" not in sessions[_TEST_SID]["pending"]

    async def test_notify_resource_updated_coalesces_by_uri(self, view):
        """Repeated updates to one uri collapse to a single pending marker."""
        session = view.hass.data[DOMAIN]["mcp_sessions"][_TEST_SID]
        session["uris"].add("hass://x")

        notify_resource_updated(view.hass, "hass://x")
        notify_resource_updated(view.hass, "hass://x")
        notify_resource_updated(view.hass, "hass://x")

        assert list(session["pending"]) == ["hass://x"]


class TestIntegrationDisabledGate:
    """Regression for #37: views return 503 when the integration is unloaded.

    HA's HTTP stack keeps registered views alive across config entry unloads,
    so we gate on `hass.data[DOMAIN]` — which async_unload_entry clears.
    """

    async def test_endpoint_view_returns_503_when_domain_missing(self):
        hass = Mock()
        hass.data = {}
        view = MCPEndpointView(hass, Mock())

        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token"}

        response = await view.post(request)

        assert response.status == 503
        body = json.loads(response.body)
        assert body["error"] == "service_unavailable"

    async def test_endpoint_view_returns_503_when_domain_cleared(self):
        hass = Mock()
        hass.data = {"mcp_server_http_transport": {}}  # matches async_unload_entry.clear()
        view = MCPEndpointView(hass, Mock())

        request = Mock()
        request.headers = {"Authorization": "Bearer valid_token"}

        response = await view.post(request)

        assert response.status == 503

    async def test_metadata_view_returns_503_when_unloaded(self):
        hass = Mock()
        hass.data = {}
        view = MCPProtectedResourceMetadataView(hass)

        response = await view.get(Mock())

        assert response.status == 503

    async def test_subpath_metadata_view_returns_503_when_unloaded(self):
        hass = Mock()
        hass.data = {}
        view = MCPSubpathProtectedResourceMetadataView(hass)

        response = await view.get(Mock())

        assert response.status == 503

    async def test_endpoint_view_serves_when_domain_populated(self):
        hass = Mock()
        hass.data = {"mcp_server_http_transport": {"server": Mock()}}
        view = MCPEndpointView(hass, Mock())

        request = Mock()
        request.headers = {}  # no token → 401, not 503

        response = await view.post(request)

        assert response.status == 401


class TestNativeAuth:
    """Test native HA authentication (Long-Lived Access Tokens)."""

    @pytest.fixture
    def mock_hass(self):
        """Create a mock Home Assistant instance with auth."""
        hass = Mock()
        hass.states = Mock()
        hass.services = Mock()
        hass.auth = Mock()
        hass.auth.async_validate_access_token = Mock(return_value=None)
        hass.data = {DOMAIN: {"mcp_sessions": {}}}
        return hass

    @pytest.fixture
    def view(self, mock_hass):
        """Create an MCPEndpointView with native auth enabled."""
        return MCPEndpointView(mock_hass, Mock(), native_auth_enabled=True)

    @pytest.fixture
    def view_disabled(self, mock_hass):
        """Create an MCPEndpointView with native auth disabled."""
        return MCPEndpointView(mock_hass, Mock(), native_auth_enabled=False)

    async def test_llat_validates_when_enabled(self, view, mock_hass):
        """Test that a valid LLAT is accepted when native auth is enabled."""
        mock_refresh_token = Mock()
        mock_refresh_token.user.id = "user_abc"
        mock_hass.auth.async_validate_access_token.return_value = mock_refresh_token

        request = Mock()
        request.headers = {"Authorization": "Bearer valid_llat"}

        result = await view._validate_token(request)

        assert result == {"sub": "user_abc"}
        mock_hass.auth.async_validate_access_token.assert_called_once_with("valid_llat")

    async def test_llat_rejected_when_disabled(self, view_disabled, mock_hass):
        """Test that LLAT is not tried when native auth is disabled."""
        request = Mock()
        request.headers = {"Authorization": "Bearer some_token"}

        result = await view_disabled._validate_token(request)

        assert result is None
        mock_hass.auth.async_validate_access_token.assert_not_called()

    async def test_invalid_llat_returns_none(self, view, mock_hass):
        """Test that an invalid LLAT returns None."""
        mock_hass.auth.async_validate_access_token.return_value = None

        request = Mock()
        request.headers = {"Authorization": "Bearer bad_token"}

        result = await view._validate_token(request)

        assert result is None

    async def test_oidc_tried_before_llat(self, view, mock_hass):
        """Test that OIDC validation is attempted before LLAT."""
        import sys

        request = Mock()
        request.headers = {"Authorization": "Bearer oidc_token"}

        mock_validator = sys.modules["custom_components.oidc_provider.token_validator"]
        original = mock_validator.validate_access_token.return_value
        mock_validator.validate_access_token.return_value = {"sub": "oidc_user"}
        try:
            result = await view._validate_token(request)
        finally:
            mock_validator.validate_access_token.return_value = original

        assert result == {"sub": "oidc_user"}
        mock_hass.auth.async_validate_access_token.assert_not_called()

    async def test_llat_fallback_after_oidc_fails(self, view, mock_hass):
        """Test LLAT is tried as fallback when OIDC validation returns None."""
        mock_refresh_token = Mock()
        mock_refresh_token.user.id = "ha_user"
        mock_hass.auth.async_validate_access_token.return_value = mock_refresh_token

        request = Mock()
        request.headers = {"Authorization": "Bearer llat_token"}

        # OIDC will fail (ImportError from conftest mock returning None)
        result = await view._validate_token(request)

        assert result == {"sub": "ha_user"}

    async def test_validate_token_import_error_falls_through(self, view, mock_hass):
        """Test _validate_token handles ImportError from OIDC and falls through to LLAT."""
        import sys

        mock_refresh_token = Mock()
        mock_refresh_token.user.id = "fallback_user"
        mock_hass.auth.async_validate_access_token.return_value = mock_refresh_token

        request = Mock()
        request.headers = {"Authorization": "Bearer some_token"}

        saved = sys.modules.pop("custom_components.oidc_provider.token_validator", None)
        saved_parent = sys.modules.pop("custom_components.oidc_provider", None)
        try:
            result = await view._validate_token(request)
        finally:
            if saved is not None:
                sys.modules["custom_components.oidc_provider.token_validator"] = saved
            if saved_parent is not None:
                sys.modules["custom_components.oidc_provider"] = saved_parent

        assert result == {"sub": "fallback_user"}

    async def test_401_without_oidc_metadata_when_oidc_unavailable(self, view, mock_hass):
        """Test 401 response uses plain Bearer when OIDC is not available."""
        mock_hass.auth.async_validate_access_token.return_value = None

        request = Mock()
        request.headers = {"Authorization": "Bearer bad_token"}
        request.url.origin.return_value = "http://localhost:8123"

        with patch(
            "custom_components.mcp_server_http_transport.http._get_issuer",
            return_value=None,
        ):
            response = await view.post(request)

        assert response.status == 401
        assert 'realm="Home Assistant MCP Server"' in response.headers["WWW-Authenticate"]
        assert "resource_metadata" not in response.headers["WWW-Authenticate"]

    async def test_native_auth_full_request(self, view, mock_hass):
        """Test a full request with native auth from token to response."""
        mock_refresh_token = Mock()
        mock_refresh_token.user.id = "user_xyz"
        mock_hass.auth.async_validate_access_token.return_value = mock_refresh_token

        request = Mock()
        request.headers = {"Authorization": "Bearer my_llat"}
        request.json = AsyncMock(return_value={"jsonrpc": "2.0", "method": "initialize", "id": 1})

        response = await view.post(request)

        assert response.status == 200
        body = json.loads(response.body)
        assert body["result"]["protocolVersion"] == "2025-03-26"


class TestOidcAudienceBinding:
    """Test that OIDC validation binds the token audience to this resource."""

    @pytest.fixture
    def view(self):
        """Create an MCPEndpointView (native auth disabled)."""
        hass = Mock()
        return MCPEndpointView(hass, Mock(), native_auth_enabled=False)

    async def test_passes_expected_audience_to_validator(self, view):
        """_validate_token derives the resource URI and passes it as audience."""
        import sys

        request = Mock()
        request.headers = {
            "Authorization": "Bearer t",
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "ha.example.com",
        }

        mv = sys.modules["custom_components.oidc_provider.token_validator"]
        mv.validate_access_token.reset_mock()
        mv.validate_access_token.return_value = {"sub": "u"}
        try:
            result = await view._validate_token(request)
        finally:
            mv.validate_access_token.return_value = None

        assert result == {"sub": "u"}
        _, kwargs = mv.validate_access_token.call_args
        assert kwargs.get("expected_audience") == "https://ha.example.com/api/mcp"

    async def test_falls_back_to_legacy_signature_on_type_error(self, view):
        """An older OIDC provider without expected_audience still works."""
        import sys

        request = Mock()
        request.headers = {
            "Authorization": "Bearer t",
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "ha.example.com",
        }

        def side_effect(*args, **kwargs):
            if "expected_audience" in kwargs:
                raise TypeError("unexpected keyword argument 'expected_audience'")
            return {"sub": "legacy"}

        mv = sys.modules["custom_components.oidc_provider.token_validator"]
        mv.validate_access_token.reset_mock()
        mv.validate_access_token.side_effect = side_effect
        try:
            result = await view._validate_token(request)
        finally:
            mv.validate_access_token.side_effect = None
            mv.validate_access_token.return_value = None

        assert result == {"sub": "legacy"}


@asynccontextmanager
async def _sse_app_client(view):
    """Run view.get behind a real aiohttp test server for genuine SSE streaming."""
    app = aiohttp_web.Application()
    app.router.add_get("/api/mcp", view.get)
    async with TestClient(TestServer(app)) as client:
        yield client


async def _read_frame(content):
    """Read one SSE frame (terminated by a blank line) with a safety timeout."""
    return await asyncio.wait_for(content.readuntil(b"\n\n"), timeout=2)


class TestSSEStream:
    """Real end-to-end SSE: prepare()/framing, keepalive, native auth, unload teardown."""

    @pytest.fixture
    def mock_hass(self):
        """HA mock whose native auth accepts any LLAT and holds one live session."""
        hass = Mock()
        hass.auth = Mock()
        refresh = Mock()
        refresh.user.id = "user-1"
        hass.auth.async_validate_access_token = Mock(return_value=refresh)
        hass.data = {DOMAIN: {"mcp_sessions": {_TEST_SID: _new_session()}}}
        return hass

    @pytest.fixture
    def view(self, mock_hass):
        """MCP endpoint with native auth enabled (real _validate_token path)."""
        return MCPEndpointView(mock_hass, Mock(), native_auth_enabled=True)

    async def test_streams_keepalive_then_data_over_real_socket(
        self, view, mock_hass, socket_enabled
    ):
        """Real prepare()/SSE framing: a keepalive fires, then a buffered notification ships."""
        session = mock_hass.data[DOMAIN]["mcp_sessions"][_TEST_SID]
        session["uris"].add("hass://x")
        with patch("custom_components.mcp_server_http_transport.http._SSE_KEEPALIVE_SECONDS", 0.05):
            async with _sse_app_client(view) as client:
                resp = await client.get(
                    "/api/mcp",
                    headers={"Authorization": "Bearer llat", "Mcp-Session-Id": _TEST_SID},
                )
                assert resp.status == 200
                assert resp.headers["Content-Type"] == "text/event-stream"

                # Empty buffer -> wait_for times out -> keepalive branch.
                assert await _read_frame(resp.content) == b": keepalive\n\n"

                # A notification for a subscribed uri is delivered as an SSE data line.
                notify_resource_updated(mock_hass, "hass://x")
                frame = b""
                for _ in range(50):
                    frame = await _read_frame(resp.content)
                    if b"notifications/resources/updated" in frame:
                        break
                assert frame.startswith(b"data: ")
                assert b"hass://x" in frame

                # Closing the session ends the stream cleanly.
                _close_session(session)
                await asyncio.wait_for(resp.content.read(), timeout=2)
                assert resp.content.at_eof()

        # The session survives the stream disconnect (a reconnect would resume it).
        assert _TEST_SID in mock_hass.data[DOMAIN]["mcp_sessions"]
        assert mock_hass.data[DOMAIN]["mcp_sessions"][_TEST_SID]["stream_attached"] is False

    async def test_native_auth_rejected_without_token(self, view):
        """GET runs the real _validate_token/_unauthorized path: no bearer -> 401 challenge."""
        request = Mock()
        request.headers = {}
        request.url.origin.return_value = "https://homeassistant.local"

        response = await view.get(request)

        assert response.status == 401
        assert "WWW-Authenticate" in response.headers

    async def test_unload_sentinel_closes_live_stream(self, view, mock_hass, socket_enabled):
        """async_unload_entry signals every session to close, breaking an open GET loop."""
        with patch("custom_components.mcp_server_http_transport.http._SSE_KEEPALIVE_SECONDS", 0.05):
            async with _sse_app_client(view) as client:
                resp = await client.get(
                    "/api/mcp",
                    headers={"Authorization": "Bearer llat", "Mcp-Session-Id": _TEST_SID},
                )
                assert resp.status == 200
                assert await _read_frame(resp.content) == b": keepalive\n\n"

                await async_unload_entry(mock_hass, Mock())

                await asyncio.wait_for(resp.content.read(), timeout=2)
                assert resp.content.at_eof()
