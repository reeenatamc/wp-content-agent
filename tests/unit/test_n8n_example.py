import json
from pathlib import Path
from typing import Any

import pytest

WORKFLOW = Path(__file__).parents[2] / "examples" / "n8n" / "brief-to-draft.json"


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


def test_start_run_calls_the_webhook_with_secret_and_idempotency_key(
    workflow: dict[str, Any],
) -> None:
    start = _node(workflow, "Start run")["parameters"]

    headers = {h["name"]: h["value"] for h in start["headerParameters"]["parameters"]}

    assert start["url"].endswith("/webhooks/n8n")
    assert start["genericAuthType"] == "httpHeaderAuth"
    assert "$execution.id" in headers["Idempotency-Key"]


def test_poll_loops_back_while_the_run_is_working(workflow: dict[str, Any]) -> None:
    branches = workflow["connections"]["Still running?"]["main"]

    assert branches[0][0]["node"] == "Wait"
    assert branches[1][0]["node"] == "Result"


def test_no_secrets_in_the_export(workflow: dict[str, Any]) -> None:
    text = WORKFLOW.read_text()

    assert all(
        node["credentials"]["httpHeaderAuth"]["id"] == ""
        for node in workflow["nodes"]
        if "credentials" in node
    )
    assert "Bearer " not in text
