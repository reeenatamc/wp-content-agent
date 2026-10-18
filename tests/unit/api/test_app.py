import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from tests.fake_graph import fake_runner
from wp_content_agent.api import create_app
from wp_content_agent.api.service import RunService
from wp_content_agent.errors import LLMError
from wp_content_agent.graph import AgentState
from wp_content_agent.runner import AgentRunner
from wp_content_agent.settings import Settings

BRIEF = {"topic": "Caching in Django", "audience": "Backend developers"}
SECRET = "s3cret-value"


def _app(runner: AgentRunner | None = None, **settings: Any) -> FastAPI:
    runner = runner or fake_runner()

    @asynccontextmanager
    async def open_runner(settings: Settings) -> AsyncIterator[AgentRunner]:
        yield runner

    config = Settings(_env_file=None, **{"webhook_secret": SECRET, **settings})  # type: ignore[call-arg]
    return create_app(config, open_runner=open_runner)


@asynccontextmanager
async def _client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as client:
            yield client


def _service(app: FastAPI) -> RunService:
    service: RunService = app.state.service
    return service


async def _started(client: httpx.AsyncClient, app: FastAPI) -> str:
    response = await client.post("/runs", json=BRIEF)
    run_id: str = response.json()["run_id"]
    await _service(app).wait(run_id)
    return run_id


async def test_post_runs_returns_accepted_with_run_id() -> None:
    app = _app()
    async with _client(app) as client:
        response = await client.post("/runs", json=BRIEF)

    assert response.status_code == 202
    assert response.json()["status"] == "planning"
    assert len(response.json()["run_id"]) == 32


async def test_run_reaches_approval_with_the_draft() -> None:
    app = _app()
    async with _client(app) as client:
        run_id = await _started(client, app)
        body = (await client.get(f"/runs/{run_id}")).json()

    assert body["status"] == "awaiting_approval"
    assert body["approval"]["title"] == "Caching Django querysets"
    assert body["total_tokens"] == 15


async def test_approve_creates_the_draft() -> None:
    app = _app()
    async with _client(app) as client:
        run_id = await _started(client, app)
        accepted = await client.post(f"/runs/{run_id}/approve", json={"note": "ok"})
        await _service(app).wait(run_id)
        body = (await client.get(f"/runs/{run_id}")).json()

    assert accepted.status_code == 202
    assert accepted.json()["status"] == "publishing"
    assert body["status"] == "draft_created"
    assert body["edit_url"].endswith("post=8&action=edit")


async def test_reject_without_body_ends_the_run() -> None:
    app = _app()
    async with _client(app) as client:
        run_id = await _started(client, app)
        await client.post(f"/runs/{run_id}/reject")
        await _service(app).wait(run_id)
        body = (await client.get(f"/runs/{run_id}")).json()

    assert body["status"] == "rejected"
    assert body["wp_post_id"] is None


async def test_second_approve_conflicts() -> None:
    app = _app()
    async with _client(app) as client:
        run_id = await _started(client, app)
        first = await client.post(f"/runs/{run_id}/approve")
        second = await client.post(f"/runs/{run_id}/approve")

    assert first.status_code == 202
    assert second.status_code == 409


async def test_unknown_run_is_404() -> None:
    app = _app()
    async with _client(app) as client:
        get = await client.get("/runs/missing")
        approve = await client.post("/runs/missing/approve")

    assert (get.status_code, approve.status_code) == (404, 404)


async def test_invalid_brief_is_422() -> None:
    app = _app()
    async with _client(app) as client:
        response = await client.post("/runs", json={"topic": "x"})

    assert response.status_code == 422


async def test_failed_run_reports_error() -> None:
    def write(state: AgentState) -> dict[str, Any]:
        raise LLMError("write failed")

    app = _app(fake_runner(write=write))
    async with _client(app) as client:
        run_id = await _started(client, app)
        body = (await client.get(f"/runs/{run_id}")).json()

    assert body["status"] == "failed"
    assert body["error"] == "write failed"


@pytest.mark.parametrize(
    ("headers", "expected"),
    [({"X-Webhook-Secret": SECRET}, 202), ({"X-Webhook-Secret": "wrong"}, 401), ({}, 401)],
)
async def test_webhook_checks_the_shared_secret(headers: dict[str, str], expected: int) -> None:
    app = _app()
    async with _client(app) as client:
        response = await client.post("/webhooks/n8n", json=BRIEF, headers=headers)

    assert response.status_code == expected


async def test_webhook_is_disabled_without_a_secret() -> None:
    app = _app(webhook_secret="")
    async with _client(app) as client:
        response = await client.post("/webhooks/n8n", json=BRIEF, headers={"X-Webhook-Secret": ""})

    assert response.status_code == 503


async def test_shutdown_marks_running_runs_failed() -> None:
    started = asyncio.Event()

    async def write(state: AgentState) -> dict[str, Any]:
        started.set()
        await asyncio.Event().wait()
        return {}

    runner = fake_runner(write=write)
    app = _app(runner)
    async with _client(app) as client:
        run_id = (await client.post("/runs", json=BRIEF)).json()["run_id"]
        await started.wait()

    assert (await runner.get(run_id)).status == "failed"
