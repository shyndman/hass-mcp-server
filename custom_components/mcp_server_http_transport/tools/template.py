"""Template entity CRUD tools (config-entry template helpers, state-based)."""

import json
from typing import Any

from homeassistant.config_entries import SOURCE_USER, ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import entity_registry as er

from . import _HAJSONEncoder, register_tool
from .categories import resolve_category_id, write_entity_category

_DOMAIN = "template"

# Mirrors TEMPLATE_TYPES in homeassistant/components/template/config_flow.py.
# Kept local (like helpers.HELPER_DOMAINS) instead of importing core's private list.
TEMPLATE_TYPES = frozenset(
    {
        "alarm_control_panel",
        "binary_sensor",
        "button",
        "cover",
        "device_tracker",
        "event",
        "fan",
        "image",
        "light",
        "lock",
        "number",
        "select",
        "sensor",
        "switch",
        "update",
        "vacuum",
        "weather",
    }
)


def _text(text: str) -> dict[str, Any]:
    """Wrap text in the MCP tool content envelope."""
    return {"content": [{"type": "text", "text": text}]}


def _resolve_template_entry(hass: HomeAssistant, entity_id: str) -> ConfigEntry:
    """Resolve an entity_id to its template config entry, or raise."""
    reg = er.async_get(hass)
    entry = reg.async_get(entity_id)
    if entry is None:
        raise ValueError(f"Entity '{entity_id}' not found in entity registry")
    if entry.config_entry_id is None:
        raise ValueError(f"'{entity_id}' is not backed by a config entry")
    ce = hass.config_entries.async_get_entry(entry.config_entry_id)
    if ce is None or ce.domain != _DOMAIN:
        raise ValueError(f"'{entity_id}' is not a template entity")
    return ce


def _entity_id_for_entry(hass: HomeAssistant, entry_id: str) -> str | None:
    """Return the single entity_id produced by a template config entry, if registered."""
    entries = er.async_entries_for_config_entry(er.async_get(hass), entry_id)
    return entries[0].entity_id if entries else None


@register_tool(
    name="create_template_entity",
    description=(
        "Create a state-based template entity via Home Assistant's UI config-entry helper flow. "
        "Supported types: alarm_control_panel, binary_sensor, button, cover, device_tracker, "
        "event, fan, image, light, lock, number, select, sensor, switch, update, vacuum, weather. "
        "Trigger-based templates are not supported here (use the YAML config-file tools)"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "template_type": {
                "type": "string",
                "description": (
                    "The template platform to create, e.g. 'sensor', 'binary_sensor', 'switch'"
                ),
            },
            "config": {
                "type": "object",
                "description": (
                    "Template configuration. Must include 'name' plus the fields for the chosen "
                    "type, e.g. sensor needs 'state'; switch needs 'turn_on'/'turn_off'; number "
                    "needs 'state','min','max','step','set_value'. Do NOT include 'template_type'"
                ),
            },
            "category": {
                "type": ["string", "null"],
                "description": (
                    "Category name to assign this entity to. "
                    "Must already exist (use create_category)."
                ),
            },
        },
        "required": ["template_type", "config"],
    },
)
async def create_template_entity(hass: HomeAssistant, arguments: dict[str, Any]) -> dict[str, Any]:
    """Create a template entity by driving the config-entry flow."""
    template_type = arguments["template_type"]
    if template_type not in TEMPLATE_TYPES:
        return _text(
            f"Unknown template type '{template_type}'. "
            f"Supported: {', '.join(sorted(TEMPLATE_TYPES))}"
        )

    try:
        category = arguments.get("category")
        category_id = (
            resolve_category_id(hass, "entity", category) if category is not None else None
        )

        result = await hass.config_entries.flow.async_init(_DOMAIN, context={"source": SOURCE_USER})
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"next_step_id": template_type}
        )
        if result["type"] is not FlowResultType.FORM:
            return _text(f"Could not start template flow: {result.get('reason', result['type'])}")

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], arguments["config"]
        )
        if result["type"] is FlowResultType.FORM:
            return _text(f"Invalid template config: {result.get('errors')}")
        if result["type"] is not FlowResultType.CREATE_ENTRY:
            return _text(f"Template creation aborted: {result.get('reason')}")

        entry = result["result"]
        await hass.async_block_till_done()
        entity_id = _entity_id_for_entry(hass, entry.entry_id)

        if category_id is not None:
            if entity_id is None:
                return _text(
                    f"Successfully created {template_type} template entity "
                    f"(entry {entry.entry_id}), but its entity is not registered yet; "
                    "category not applied."
                )
            write_entity_category(hass, entity_id, "entity", category_id)

        suffix = f" in category '{category}'" if category else ""
        return _text(
            f"Successfully created {template_type} template entity "
            f"'{entity_id or entry.title}' (entry {entry.entry_id}){suffix}"
        )
    except Exception as e:
        return _text(f"Error creating template entity: {str(e)}")


@register_tool(
    name="list_template_entities",
    description=(
        "List all state-based template entities created via the UI config-entry helper flow, "
        "optionally filtered by template type"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "template_type": {
                "type": "string",
                "description": (
                    "Filter by template type, e.g. 'sensor' or 'binary_sensor'. "
                    "Omit to list all types"
                ),
            }
        },
    },
)
async def list_template_entities(hass: HomeAssistant, arguments: dict[str, Any]) -> dict[str, Any]:
    """List template config entries."""
    try:
        type_filter = arguments.get("template_type")
        entities = []
        for entry in hass.config_entries.async_entries(_DOMAIN):
            ttype = entry.options.get("template_type")
            if type_filter and ttype != type_filter:
                continue
            entities.append(
                {
                    "entry_id": entry.entry_id,
                    "entity_id": _entity_id_for_entry(hass, entry.entry_id),
                    "name": entry.title,
                    "template_type": ttype,
                    "options": dict(entry.options),
                }
            )
        return _text(json.dumps(entities, indent=2, cls=_HAJSONEncoder))
    except Exception as e:
        return _text(f"Error listing template entities: {str(e)}")


@register_tool(
    name="get_template_entity",
    description="Get the stored configuration of a template entity by its entity ID",
    input_schema={
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": "The template entity ID (e.g., sensor.my_template)",
            }
        },
        "required": ["entity_id"],
    },
)
async def get_template_entity(hass: HomeAssistant, arguments: dict[str, Any]) -> dict[str, Any]:
    """Get a template entity's stored config."""
    try:
        entity_id = arguments["entity_id"]
        entry = _resolve_template_entry(hass, entity_id)
        return _text(
            json.dumps(
                {
                    "entry_id": entry.entry_id,
                    "entity_id": entity_id,
                    "name": entry.title,
                    "template_type": entry.options.get("template_type"),
                    "options": dict(entry.options),
                },
                indent=2,
                cls=_HAJSONEncoder,
            )
        )
    except Exception as e:
        return _text(f"Error getting template entity: {str(e)}")


@register_tool(
    name="update_template_entity",
    description=(
        "Update an existing template entity by its entity ID. "
        "Changes the template fields only; 'name' and 'template_type' are immutable here"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": "The template entity ID (e.g., sensor.my_template)",
            },
            "config": {
                "type": "object",
                "description": (
                    "Updated template fields (all of them, not just changed ones). "
                    "Do NOT include 'name' or 'template_type' — those are immutable here"
                ),
            },
            "category": {
                "type": ["string", "null"],
                "description": (
                    "Category name to assign this entity to. "
                    "Must already exist (use create_category). "
                    "Pass null to remove the category."
                ),
            },
        },
        "required": ["entity_id", "config"],
    },
)
async def update_template_entity(hass: HomeAssistant, arguments: dict[str, Any]) -> dict[str, Any]:
    """Update a template entity by driving its options flow."""
    try:
        has_cat = "category" in arguments
        category = arguments.get("category")
        category_id = (
            resolve_category_id(hass, "entity", category)
            if (has_cat and category is not None)
            else None
        )
        config = arguments["config"]
        immutable = {"name", "template_type"} & config.keys()
        if immutable:
            return _text(
                f"Cannot change {', '.join(sorted(immutable))} via update_template_entity; "
                "these are fixed at creation. Remove them from config."
            )
        entity_id = arguments["entity_id"]
        entry = _resolve_template_entry(hass, entity_id)

        result = await hass.config_entries.options.async_init(entry.entry_id)
        if result["type"] is not FlowResultType.FORM:
            return _text(f"Could not open options flow: {result.get('reason', result['type'])}")

        result = await hass.config_entries.options.async_configure(result["flow_id"], config)
        if result["type"] is FlowResultType.FORM:
            return _text(f"Invalid template config: {result.get('errors')}")
        if result["type"] is not FlowResultType.CREATE_ENTRY:
            return _text(f"Template update aborted: {result.get('reason')}")

        await hass.async_block_till_done()
        if has_cat:
            write_entity_category(hass, entity_id, "entity", category_id)
        return _text("Successfully updated template entity")
    except Exception as e:
        return _text(f"Error updating template entity: {str(e)}")


@register_tool(
    name="delete_template_entity",
    description="Delete a template entity from Home Assistant by its entity ID",
    input_schema={
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": "The template entity ID (e.g., sensor.my_template)",
            }
        },
        "required": ["entity_id"],
    },
)
async def delete_template_entity(hass: HomeAssistant, arguments: dict[str, Any]) -> dict[str, Any]:
    """Delete a template entity by removing its config entry."""
    try:
        entry = _resolve_template_entry(hass, arguments["entity_id"])
        await hass.config_entries.async_remove(entry.entry_id)
        return _text("Successfully deleted template entity")
    except Exception as e:
        return _text(f"Error deleting template entity: {str(e)}")
