import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from tests.fake_graph import fake_runner, flaky_publish
from wp_content_agent.api import create_app
from wp_content_agent.api.service import RunService
from wp_content_agent.errors import LLMError, ToolError
from wp_content_agent.graph import AgentState
from wp_content_agent.runner import AgentRunner
from wp_content_agent.schemas import RunStatus
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
    assert (accepted.json()["status"], accepted.json()["approval"]) == ("publishing", None)
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


async def test_concurrent_approvals_get_one_conflict() -> None:
    app = _app()
    async with _client(app) as client:
        run_id = await _started(client, app)
        responses = await asyncio.gather(
            client.post(f"/runs/{run_id}/approve"), client.post(f"/runs/{run_id}/reject")
        )
        await _service(app).wait(run_id)
        final = (await client.get(f"/runs/{run_id}")).json()

    assert sorted(r.status_code for r in responses) == [202, 409]
    assert final["status"] in {"draft_created", "rejected"}


async def test_idempotency_key_returns_the_same_run() -> None:
    app = _app()
    headers = {"Idempotency-Key": "brief-2026-10-18"}
    async with _client(app) as client:
        first = await client.post("/runs", json=BRIEF, headers=headers)
        await _service(app).wait("brief-2026-10-18")
        second = await client.post("/runs", json=BRIEF, headers=headers)

    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json()["run_id"] == "brief-2026-10-18"
    assert second.json()["status"] == "awaiting_approval"


async def test_invalid_idempotency_key_is_422() -> None:
    app = _app()
    async with _client(app) as client:
        response = await client.post("/runs", json=BRIEF, headers={"Idempotency-Key": "a/b"})

    assert response.status_code == 422


async def test_too_many_runs_is_429() -> None:
    gate = asyncio.Event()

    async def plan(state: AgentState) -> dict[str, Any]:
        await gate.wait()
        return {}

    app = _app(fake_runner(plan=plan), max_concurrent_runs=1)
    async with _client(app) as client:
        first = await client.post("/runs", json=BRIEF)
        second = await client.post("/runs", json=BRIEF)
        gate.set()
        await _service(app).wait(first.json()["run_id"])

    assert (first.status_code, second.status_code) == (202, 429)


async def _failed_publish(client: httpx.AsyncClient, app: FastAPI) -> str:
    run_id = await _started(client, app)
    await client.post(f"/runs/{run_id}/approve")
    await _service(app).wait(run_id)
    return run_id


async def test_retry_creates_the_draft_after_a_publish_failure() -> None:
    app = _app(fake_runner(publish_draft=flaky_publish()))
    async with _client(app) as client:
        run_id = await _failed_publish(client, app)
        failed = (await client.get(f"/runs/{run_id}")).json()
        accepted = await client.post(f"/runs/{run_id}/retry")
        await _service(app).wait(run_id)
        body = (await client.get(f"/runs/{run_id}")).json()

    assert failed["can_retry"] is True
    assert accepted.status_code == 202
    assert (accepted.json()["status"], accepted.json()["error"]) == ("publishing", None)
    assert accepted.json()["can_retry"] is False
    assert (body["status"], body["wp_post_id"], body["can_retry"]) == ("draft_created", 8, False)


async def test_retry_unknown_run_is_404() -> None:
    app = _app()
    async with _client(app) as client:
        response = await client.post("/runs/missing/retry")

    assert response.status_code == 404


async def test_retry_of_a_run_that_did_not_fail_in_publish_is_409() -> None:
    app = _app()
    async with _client(app) as client:
        run_id = await _started(client, app)
        response = await client.post(f"/runs/{run_id}/retry")

    assert response.status_code == 409
    assert "cannot be retried" in response.json()["detail"]


async def test_retry_while_running_is_409() -> None:
    gate = asyncio.Event()
    calls = 0

    async def publish_draft(state: AgentState) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ToolError("create_draft_post failed: timeout")
        await gate.wait()
        return {"status": RunStatus.DRAFT_CREATED, "wp_post_id": 8}

    app = _app(fake_runner(publish_draft=publish_draft))
    async with _client(app) as client:
        run_id = await _failed_publish(client, app)
        first = await client.post(f"/runs/{run_id}/retry")
        second = await client.post(f"/runs/{run_id}/retry")
        during = (await client.get(f"/runs/{run_id}")).json()
        gate.set()
        await _service(app).wait(run_id)

    assert (first.status_code, second.status_code) == (202, 409)
    assert "still running" in second.json()["detail"]
    assert (during["status"], during["can_retry"]) == ("publishing", False)


async def test_retry_when_all_run_slots_are_busy_is_429() -> None:
    gate = asyncio.Event()
    blocking = False

    async def plan(state: AgentState) -> dict[str, Any]:
        if blocking:
            await gate.wait()
        return {}

    app = _app(fake_runner(plan=plan, publish_draft=flaky_publish()), max_concurrent_runs=1)
    async with _client(app) as client:
        failed_id = await _failed_publish(client, app)
        blocking = True
        busy = await client.post("/runs", json=BRIEF)
        response = await client.post(f"/runs/{failed_id}/retry")
        gate.set()
        await _service(app).wait(busy.json()["run_id"])

    assert response.status_code == 429


async def test_retry_requires_the_api_token() -> None:
    app = _app(api_token="tok-123")
    async with _client(app) as client:
        response = await client.post("/runs/missing/retry")

    assert response.status_code == 401


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Authorization": "Bearer tok-123"}, 404),
        ({"Authorization": "Bearer nope"}, 401),
        ({}, 401),
    ],
)
async def test_api_token_protects_run_endpoints(headers: dict[str, str], expected: int) -> None:
    app = _app(api_token="tok-123")
    async with _client(app) as client:
        response = await client.get("/runs/missing", headers=headers)

    assert response.status_code == expected


async def test_shutdown_before_a_run_starts_marks_it_failed() -> None:
    runner = fake_runner()
    app = _app(runner)
    async with _client(app) as client:
        run_id = (await client.post("/runs", json=BRIEF)).json()["run_id"]

    snapshot = await runner.get(run_id)
    assert snapshot.status == "failed"
    assert snapshot.error == "the server stopped during the run"


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


async def test_revise_sends_the_draft_back_and_waits_for_approval_again() -> None:
    app = _app()
    async with _client(app) as client:
        run_id = await _started(client, app)
        accepted = await client.post(f"/runs/{run_id}/revise", json={"note": "shorter intro"})
        await _service(app).wait(run_id)
        body = (await client.get(f"/runs/{run_id}")).json()

    assert (accepted.status_code, accepted.json()["status"]) == (202, "writing")
    assert body["status"] == "awaiting_approval"
    assert body["approval"]["human_revisions_left"] == 1


@pytest.mark.parametrize("body", [{}, {"note": ""}, {"note": "   "}, {"note": "x" * 2001}])
async def test_revise_with_a_missing_or_blank_note_is_422(body: dict[str, str]) -> None:
    app = _app()
    async with _client(app) as client:
        run_id = await _started(client, app)
        response = await client.post(f"/runs/{run_id}/revise", json=body)

    assert response.status_code == 422


async def test_revise_with_no_revisions_left_conflicts() -> None:
    settings = Settings(_env_file=None, max_human_revisions=1)  # type: ignore[call-arg]
    app = _app(fake_runner(settings))
    async with _client(app) as client:
        run_id = await _started(client, app)
        await client.post(f"/runs/{run_id}/revise", json={"note": "one"})
        await _service(app).wait(run_id)
        response = await client.post(f"/runs/{run_id}/revise", json={"note": "two"})

    assert response.status_code == 409
    assert "no human revisions left" in response.json()["detail"]
