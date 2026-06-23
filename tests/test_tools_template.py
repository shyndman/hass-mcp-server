"""Tests for template entity CRUD tools."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.data_entry_flow import FlowResultType

from custom_components.mcp_server_http_transport.const import DOMAIN
from custom_components.mcp_server_http_transport.http import MCPEndpointView
from custom_components.mcp_server_http_transport.tools.template import (
    create_template_entity,
    delete_template_entity,
    get_template_entity,
    list_template_entities,
    update_template_entity,
)

_TEST_SID = "test-session-id"


def _make_entry(entry_id: str, name: str, template_type: str, **fields) -> Mock:
    """Create a mock template config entry."""
    entry = Mock()
    entry.entry_id = entry_id
    entry.domain = "template"
    entry.title = name
    entry.options = {"name": name, "template_type": template_type, **fields}
    return entry


def _make_hass() -> Mock:
    """Create a mock hass with config-entries flow plumbing."""
    hass = Mock()
    hass.async_block_till_done = AsyncMock()
    hass.config_entries = Mock()
    hass.config_entries.flow = Mock()
    hass.config_entries.flow.async_init = AsyncMock()
    hass.config_entries.flow.async_configure = AsyncMock()
    hass.config_entries.options = Mock()
    hass.config_entries.options.async_init = AsyncMock()
    hass.config_entries.options.async_configure = AsyncMock()
    hass.config_entries.async_entries = Mock(return_value=[])
    hass.config_entries.async_get_entry = Mock(return_value=None)
    hass.config_entries.async_remove = AsyncMock()
    return hass


class TestCreateTemplateEntity:
    """Tests for create_template_entity."""

    async def test_create_success_returns_entity_id(self):
        """Drives menu -> form -> create and returns the resolved entity_id."""
        hass = _make_hass()
        created = Mock(entry_id="ce1", title="My Temp")
        hass.config_entries.flow.async_init.return_value = {
            "type": FlowResultType.MENU,
            "flow_id": "f1",
        }
        hass.config_entries.flow.async_configure.side_effect = [
            {"type": FlowResultType.FORM, "flow_id": "f1", "step_id": "sensor"},
            {"type": FlowResultType.CREATE_ENTRY, "result": created},
        ]

        reg = Mock()
        with (
            patch(
                "homeassistant.helpers.entity_registry.async_get",
                return_value=reg,
            ),
            patch(
                "homeassistant.helpers.entity_registry.async_entries_for_config_entry",
                return_value=[Mock(entity_id="sensor.my_temp")],
            ),
        ):
            result = await create_template_entity(
                hass,
                {
                    "template_type": "sensor",
                    "config": {"name": "My Temp", "state": "{{ 1 }}"},
                },
            )

        # Menu selection then the platform form.
        first, second = hass.config_entries.flow.async_configure.call_args_list
        assert first.args[1] == {"next_step_id": "sensor"}
        assert second.args[1] == {"name": "My Temp", "state": "{{ 1 }}"}
        text = result["content"][0]["text"]
        assert "sensor.my_temp" in text
        assert "ce1" in text

    async def test_create_unknown_type_skips_flow(self):
        """Unknown template type is rejected before any flow call."""
        hass = _make_hass()

        result = await create_template_entity(
            hass, {"template_type": "frobnicate", "config": {"name": "x"}}
        )

        text = result["content"][0]["text"]
        assert "Unknown template type" in text
        assert "sensor" in text  # supported list shown
        hass.config_entries.flow.async_init.assert_not_called()

    async def test_create_validation_error_reports_form_errors(self):
        """A FORM result on final configure surfaces the validation errors."""
        hass = _make_hass()
        hass.config_entries.flow.async_init.return_value = {
            "type": FlowResultType.MENU,
            "flow_id": "f1",
        }
        hass.config_entries.flow.async_configure.side_effect = [
            {"type": FlowResultType.FORM, "flow_id": "f1", "step_id": "sensor"},
            {"type": FlowResultType.FORM, "errors": {"state": "invalid_template"}},
        ]

        result = await create_template_entity(
            hass, {"template_type": "sensor", "config": {"name": "Bad"}}
        )

        text = result["content"][0]["text"]
        assert "Invalid template config" in text
        assert "invalid_template" in text

    async def test_create_with_category_writes_entity_category(self):
        """Category name is resolved and written to the created entity."""
        hass = _make_hass()
        created = Mock(entry_id="ce1", title="My Temp")
        hass.config_entries.flow.async_init.return_value = {
            "type": FlowResultType.MENU,
            "flow_id": "f1",
        }
        hass.config_entries.flow.async_configure.side_effect = [
            {"type": FlowResultType.FORM, "flow_id": "f1", "step_id": "sensor"},
            {"type": FlowResultType.CREATE_ENTRY, "result": created},
        ]

        cat = Mock(category_id="c1")
        cat.name = "Climate"
        cr_reg = Mock()
        cr_reg.async_list_categories.return_value = [cat]

        ent_reg = Mock()
        ent_reg.async_get.return_value = Mock(categories={})

        with (
            patch(
                "custom_components.mcp_server_http_transport.tools.categories.cr.async_get",
                return_value=cr_reg,
            ),
            patch(
                "homeassistant.helpers.entity_registry.async_get",
                return_value=ent_reg,
            ),
            patch(
                "homeassistant.helpers.entity_registry.async_entries_for_config_entry",
                return_value=[Mock(entity_id="sensor.my_temp")],
            ),
        ):
            result = await create_template_entity(
                hass,
                {
                    "template_type": "sensor",
                    "config": {"name": "My Temp", "state": "{{ 1 }}"},
                    "category": "Climate",
                },
            )

        _, kwargs = ent_reg.async_update_entity.call_args
        assert kwargs["categories"] == {"entity": "c1"}
        assert "in category 'Climate'" in result["content"][0]["text"]


class TestListTemplateEntities:
    """Tests for list_template_entities."""

    async def test_filter_by_template_type(self):
        """The template_type filter returns only matching entries."""
        hass = _make_hass()
        hass.config_entries.async_entries.return_value = [
            _make_entry("ce1", "Temp", "sensor"),
            _make_entry("ce2", "Door", "binary_sensor"),
        ]

        with (
            patch("homeassistant.helpers.entity_registry.async_get", return_value=Mock()),
            patch(
                "homeassistant.helpers.entity_registry.async_entries_for_config_entry",
                return_value=[],
            ),
        ):
            result = await list_template_entities(hass, {"template_type": "binary_sensor"})

        body = json.loads(result["content"][0]["text"])
        assert len(body) == 1
        assert body[0]["entry_id"] == "ce2"
        assert body[0]["template_type"] == "binary_sensor"


class TestGetTemplateEntity:
    """Tests for get_template_entity."""

    async def test_get_returns_options(self):
        """get returns the stored options for a template entity."""
        hass = _make_hass()
        entry = _make_entry("ce1", "Temp", "sensor", state="{{ 1 }}")
        hass.config_entries.async_get_entry.return_value = entry

        reg = Mock()
        reg.async_get.return_value = Mock(config_entry_id="ce1")
        with patch("homeassistant.helpers.entity_registry.async_get", return_value=reg):
            result = await get_template_entity(hass, {"entity_id": "sensor.temp"})

        body = json.loads(result["content"][0]["text"])
        assert body["entry_id"] == "ce1"
        assert body["template_type"] == "sensor"
        assert body["options"]["state"] == "{{ 1 }}"

    async def test_get_non_template_entity_errors(self):
        """Resolving an entity backed by a non-template entry errors."""
        hass = _make_hass()
        hass.config_entries.async_get_entry.return_value = Mock(domain="zha")

        reg = Mock()
        reg.async_get.return_value = Mock(config_entry_id="other")
        with patch("homeassistant.helpers.entity_registry.async_get", return_value=reg):
            result = await get_template_entity(hass, {"entity_id": "sensor.zha_thing"})

        assert "not a template entity" in result["content"][0]["text"]


class TestUpdateTemplateEntity:
    """Tests for update_template_entity."""

    async def test_update_drives_options_flow(self):
        """Update opens the options flow and configures it with the new config."""
        hass = _make_hass()
        entry = _make_entry("ce1", "Temp", "sensor")
        hass.config_entries.async_get_entry.return_value = entry
        hass.config_entries.options.async_init.return_value = {
            "type": FlowResultType.FORM,
            "flow_id": "o1",
        }
        hass.config_entries.options.async_configure.return_value = {
            "type": FlowResultType.CREATE_ENTRY,
            "result": entry,
        }

        reg = Mock()
        reg.async_get.return_value = Mock(config_entry_id="ce1")
        with patch("homeassistant.helpers.entity_registry.async_get", return_value=reg):
            result = await update_template_entity(
                hass, {"entity_id": "sensor.temp", "config": {"state": "{{ 2 }}"}}
            )

        hass.config_entries.options.async_init.assert_called_once_with("ce1")
        _, args = hass.config_entries.options.async_configure.call_args.args
        assert args == {"state": "{{ 2 }}"}
        assert "Successfully updated template entity" in result["content"][0]["text"]

    async def test_update_rejects_immutable_keys(self):
        """name/template_type in config are rejected before driving the flow."""
        hass = _make_hass()

        result = await update_template_entity(
            hass, {"entity_id": "sensor.temp", "config": {"name": "New", "state": "{{ 2 }}"}}
        )

        text = result["content"][0]["text"]
        assert "Cannot change name" in text
        hass.config_entries.options.async_init.assert_not_called()


class TestDeleteTemplateEntity:
    """Tests for delete_template_entity."""

    async def test_delete_removes_config_entry(self):
        """Delete resolves the entity and removes its config entry."""
        hass = _make_hass()
        entry = _make_entry("ce1", "Temp", "sensor")
        hass.config_entries.async_get_entry.return_value = entry

        reg = Mock()
        reg.async_get.return_value = Mock(config_entry_id="ce1")
        with patch("homeassistant.helpers.entity_registry.async_get", return_value=reg):
            result = await delete_template_entity(hass, {"entity_id": "sensor.temp"})

        hass.config_entries.async_remove.assert_called_once_with("ce1")
        assert "Successfully deleted template entity" in result["content"][0]["text"]


class TestTemplateToolsViaHTTP:
    """Integration-style test: template tools appear in tools/list."""

    @pytest.fixture
    def mock_hass(self):
        hass = Mock()
        hass.data = {
            DOMAIN: {"mcp_sessions": {_TEST_SID: {"queue": asyncio.Queue(), "uris": set()}}}
        }
        hass.states = Mock()
        hass.services = Mock()
        return hass

    @pytest.fixture
    def view(self, mock_hass):
        return MCPEndpointView(mock_hass, Mock())

    def _make_request(self, method: str, params: dict, request_id: int = 1) -> Mock:
        request = Mock()
        request.headers = {
            "Authorization": "Bearer valid_token",
            "Mcp-Session-Id": _TEST_SID,
        }
        request.json = AsyncMock(
            return_value={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
        return request

    async def test_template_tools_appear_in_tools_list(self, view):
        """All five template tools are registered."""
        request = self._make_request("tools/list", {})

        with patch.object(view, "_validate_token", return_value={"sub": "user123"}):
            response = await view.post(request)

        assert response.status == 200
        body = json.loads(response.body)
        tool_names = [t["name"] for t in body["result"]["tools"]]
        for name in (
            "create_template_entity",
            "list_template_entities",
            "get_template_entity",
            "update_template_entity",
            "delete_template_entity",
        ):
            assert name in tool_names
