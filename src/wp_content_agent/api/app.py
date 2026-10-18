"""HTTP API: start runs, read them, approve or reject, and an n8n webhook."""

import secrets
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from wp_content_agent.api.service import RunService
from wp_content_agent.errors import RunNotFoundError, RunStateError
from wp_content_agent.runner import AgentRunner
from wp_content_agent.schemas import ApprovalDecision, Brief, RunSnapshot
from wp_content_agent.settings import Settings, get_settings

WEBHOOK_HEADER = "X-Webhook-Secret"

OpenRunner = Callable[[Settings], AbstractAsyncContextManager[AgentRunner]]


class DecisionNote(BaseModel):
    note: str = Field(default="", max_length=2000)


def create_app(
    settings: Settings | None = None, *, open_runner: OpenRunner = AgentRunner.open
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with open_runner(settings) as runner:
            service = RunService(runner)
            app.state.service = service
            try:
                yield
            finally:
                await service.aclose()

    app = FastAPI(title="wp-content-agent", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RunNotFoundError)
    async def not_found(request: Request, exc: RunNotFoundError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=status.HTTP_404_NOT_FOUND)

    @app.exception_handler(RunStateError)
    async def conflict(request: Request, exc: RunStateError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=status.HTTP_409_CONFLICT)

    def service(request: Request) -> RunService:
        run_service: RunService = request.app.state.service
        return run_service

    Service = Annotated[RunService, Depends(service)]

    def check_secret(
        secret: Annotated[str | None, Header(alias=WEBHOOK_HEADER)] = None,
    ) -> None:
        expected = settings.webhook_secret
        if expected is None or not expected.get_secret_value():
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "webhook is disabled")
        given = (secret or "").encode()
        if not secrets.compare_digest(given, expected.get_secret_value().encode()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid webhook secret")

    @app.post("/runs", status_code=status.HTTP_202_ACCEPTED)
    async def start_run(brief: Brief, runs: Service) -> RunSnapshot:
        return runs.start(brief)

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str, runs: Service) -> RunSnapshot:
        return await runs.get(run_id)

    @app.post("/runs/{run_id}/approve", status_code=status.HTTP_202_ACCEPTED)
    async def approve_run(
        run_id: str, runs: Service, body: DecisionNote | None = None
    ) -> RunSnapshot:
        note = body.note if body else ""
        return await runs.resume(run_id, ApprovalDecision(approved=True, note=note))

    @app.post("/runs/{run_id}/reject", status_code=status.HTTP_202_ACCEPTED)
    async def reject_run(
        run_id: str, runs: Service, body: DecisionNote | None = None
    ) -> RunSnapshot:
        note = body.note if body else ""
        return await runs.resume(run_id, ApprovalDecision(approved=False, note=note))

    @app.post(
        "/webhooks/n8n",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(check_secret)],
    )
    async def n8n_webhook(brief: Brief, runs: Service) -> RunSnapshot:
        return runs.start(brief)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
