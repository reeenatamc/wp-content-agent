import json
import re
from pathlib import Path
from typing import Any

import pytest

from wp_content_agent.api.app import _KEY_PATTERN, WEBHOOK_HEADER
from wp_content_agent.schemas import Brief, RunStatus

WORKFLOW = Path(__file__).parents[2] / "examples" / "n8n" / "brief-to-draft.json"
TERMINAL = {"awaiting_approval", "draft_created", "rejected", "failed"}


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(WORKFLOW.read_text())
    return data


def _node(workflow: dict[str, Any], name: str) -> dict[str, Any]:
    return next(node for node in workflow["nodes"] if node["name"] == name)


def test_connections_point_at_existing_nodes(workflow: dict[str, Any]) -> None:
    names = {node["name"] for node in workflow["nodes"]}

    targets = {
        link["node"]
        for outputs in workflow["connections"].values()
        for branch in outputs["main"]
        for link in branch
    }

    assert set(workflow["connections"]) <= names
    assert targets <= names


def test_each_request_uses_the_right_credential(workflow: dict[str, Any]) -> None:
    start = _node(workflow, "Start run")
    get = _node(workflow, "Get run")

    assert start["parameters"]["url"].endswith("/webhooks/n8n")
    assert start["credentials"]["httpHeaderAuth"]["name"] == "wpagent webhook secret"
    assert get["credentials"]["httpHeaderAuth"]["name"] == "wpagent API token"


def test_idempotency_key_matches_the_api_pattern(workflow: dict[str, Any]) -> None:
    code = _node(workflow, "Build brief")["parameters"]["jsCode"]
    template = re.search(r"requestKey = `(.+?)`", code)

    assert template is not None
    example = re.sub(r"\$\{[^}]+\}", "123", template.group(1))
    assert re.fullmatch(_KEY_PATTERN, example)


def test_form_fields_are_brief_fields(workflow: dict[str, Any]) -> None:
    fields = _node(workflow, "Brief form")["parameters"]["formFields"]["values"]

    assert {f["fieldLabel"] for f in fields} <= set(Brief.model_fields)


def test_poll_stops_on_terminal_statuses(workflow: dict[str, Any]) -> None:
    conditions = _node(workflow, "Still running?")["parameters"]["conditions"]["conditions"]
    branches = workflow["connections"]["Still running?"]["main"]

    listed = set(re.findall(r"'([a-z_]+)'", conditions[0]["leftValue"]))
    assert listed == TERMINAL
    assert {status.value for status in RunStatus} >= TERMINAL
    assert "$runIndex" in conditions[1]["leftValue"]
    assert (branches[0][0]["node"], branches[1][0]["node"]) == ("Wait", "Result")


def test_no_secrets_in_the_export(workflow: dict[str, Any]) -> None:
    credentials = [n["credentials"] for n in workflow["nodes"] if "credentials" in n]
    headers = [
        header
        for node in workflow["nodes"]
        for header in node["parameters"].get("headerParameters", {}).get("parameters", [])
    ]

    assert all(c["httpHeaderAuth"]["id"] == "" for c in credentials)
    assert all(h["value"].startswith("=") for h in headers)
    assert all(h["name"] not in {"Authorization", WEBHOOK_HEADER} for h in headers)
