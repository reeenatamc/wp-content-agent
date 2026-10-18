"""HTTP API: start runs, read them, approve, reject or revise, and an n8n webhook."""

import asyncio
import json
import secrets
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, SecretStr, ValidationError

from wp_content_agent.api.service import RunEvent, RunService
from wp_content_agent.errors import RunLimitError, RunNotFoundError, RunStateError
from wp_content_agent.runner import AgentRunner
from wp_content_agent.schemas import ApprovalDecision, Brief, RunSnapshot
from wp_content_agent.settings import Settings, get_settings

WEBHOOK_HEADER = "X-Webhook-Secret"
# A retried request with the same key returns the existing run instead of a new one.
IDEMPOTENCY_HEADER = "Idempotency-Key"
_KEY_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"

OpenRunner = Callable[[Settings], AbstractAsyncContextManager[AgentRunner]]
IdempotencyKey = Annotated[str | None, Header(alias=IDEMPOTENCY_HEADER, pattern=_KEY_PATTERN)]


class DecisionNote(BaseModel):
    note: str = Field(default="", max_length=2000)


class RevisionNote(BaseModel):
    note: str = Field(min_length=1, max_length=2000)


def create_app(
    settings: Settings | None = None, *, open_runner: OpenRunner = AgentRunner.open
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with open_runner(settings) as runner:
            service = RunService(runner, settings.max_concurrent_runs)
            app.state.service = service
            try:
                yield
            finally:
                await service.aclose()

    app = FastAPI(title="wp-content-agent", version="0.1.0", lifespan=lifespan)

    for error, code in (
        (RunNotFoundError, status.HTTP_404_NOT_FOUND),
        (RunStateError, status.HTTP_409_CONFLICT),
        (RunLimitError, status.HTTP_429_TOO_MANY_REQUESTS),
    ):
        app.add_exception_handler(error, _handler(code))

    def service(request: Request) -> RunService:
        run_service: RunService = request.app.state.service
        return run_service

    Service = Annotated[RunService, Depends(service)]

    def check_token(authorization: Annotated[str | None, Header()] = None) -> None:
        if settings.api_token is None:
            return
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not _same(token, settings.api_token):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def check_secret(
        secret: Annotated[str | None, Header(alias=WEBHOOK_HEADER)] = None,
    ) -> None:
        expected = settings.webhook_secret
        if expected is None or not expected.get_secret_value():
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "webhook is disabled")
        if not _same(secret or "", expected):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid webhook secret")

    async def start(
        brief: Brief, runs: RunService, response: Response, key: str | None
    ) -> RunSnapshot:
        snapshot, created = await runs.start(brief, run_id=key)
        response.status_code = status.HTTP_202_ACCEPTED if created else status.HTTP_200_OK
        return snapshot

    async def decide(
        runs: RunService, run_id: str, approved: bool, body: DecisionNote | None
    ) -> RunSnapshot:
        decision = ApprovalDecision(approved=approved, note=body.note if body else "")
        return await runs.resume(run_id, decision)

    router = APIRouter(prefix="/runs", dependencies=[Depends(check_token)])

    @router.post("", status_code=status.HTTP_202_ACCEPTED)
    async def start_run(
        brief: Brief, runs: Service, response: Response, key: IdempotencyKey = None
    ) -> RunSnapshot:
        return await start(brief, runs, response, key)

    @router.get("/{run_id}")
    async def get_run(run_id: str, runs: Service) -> RunSnapshot:
        return await runs.get(run_id)

    @router.get("/{run_id}/events")
    async def run_events(run_id: str, runs: Service) -> StreamingResponse:
        events = runs.events(run_id)
        # The first event is read here so an unknown run is a 404, not an empty stream.
        first = await anext(events)
        return StreamingResponse(
            _sse(first, events, settings.sse_keepalive_s),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/{run_id}/approve", status_code=status.HTTP_202_ACCEPTED)
    async def approve_run(
        run_id: str, runs: Service, body: DecisionNote | None = None
    ) -> RunSnapshot:
        return await decide(runs, run_id, True, body)

    @router.post("/{run_id}/reject", status_code=status.HTTP_202_ACCEPTED)
    async def reject_run(
        run_id: str, runs: Service, body: DecisionNote | None = None
    ) -> RunSnapshot:
        return await decide(runs, run_id, False, body)

    @router.post("/{run_id}/retry", status_code=status.HTTP_202_ACCEPTED)
    async def retry_run(run_id: str, runs: Service) -> RunSnapshot:
        return await runs.retry(run_id)

    @router.post("/{run_id}/revise", status_code=status.HTTP_202_ACCEPTED)
    async def revise_run(run_id: str, body: RevisionNote, runs: Service) -> RunSnapshot:
        try:
            decision = ApprovalDecision(approved=False, note=body.note, revise=True)
        except ValidationError as exc:
            # A note of only spaces passes the length check but says nothing to change.
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, "note must not be blank"
            ) from exc
        return await runs.resume(run_id, decision)

    app.include_router(router)

    @app.post(
        "/webhooks/n8n",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(check_secret)],
    )
    async def n8n_webhook(
        brief: Brief, runs: Service, response: Response, key: IdempotencyKey = None
    ) -> RunSnapshot:
        return await start(brief, runs, response, key)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


async def _sse(
    first: RunEvent, rest: AsyncGenerator[RunEvent], keepalive_s: float
) -> AsyncIterator[str]:
    """Server-sent events, with a comment line whenever nothing was sent for keepalive_s."""
    pending: asyncio.Task[RunEvent] | None = None
    try:
        yield _frame(first)
        while True:
            # A timed-out wait must not cancel the read, or the event it was about to
            # deliver would be lost; the same task is awaited again.
            pending = pending or asyncio.ensure_future(anext(rest))
            done, _ = await asyncio.wait({pending}, timeout=keepalive_s)
            if not done:
                yield ": keepalive\n\n"
                continue
            task, pending = pending, None
            try:
                yield _frame(task.result())
            except StopAsyncIteration:
                return
    finally:
        if pending is not None:
            pending.cancel()
            # The generator cannot be closed while that read is still running.
            await asyncio.gather(pending, return_exceptions=True)
        await rest.aclose()


def _frame(event: RunEvent) -> str:
    return f"event: {event.type}\ndata: {json.dumps(event.data)}\n\n"


def _same(given: str, expected: SecretStr) -> bool:
    return secrets.compare_digest(given.encode(), expected.get_secret_value().encode())


def _handler(code: int) -> Callable[[Request, Exception], JSONResponse]:
    def handle(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=code)

    return handle
