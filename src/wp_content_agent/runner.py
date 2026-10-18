"""Runs the graph: start, resume after approval, and report where a run stands."""

import logging
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import httpx
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from wp_content_agent.errors import AgentError, LLMError, RunNotFoundError, RunStateError
from wp_content_agent.graph import AgentContext, build_graph
from wp_content_agent.llm import build_model_client
from wp_content_agent.nodes import NODES
from wp_content_agent.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    Brief,
    RunSnapshot,
    RunStatus,
)
from wp_content_agent.settings import Settings
from wp_content_agent.tools.mcp_gateway import connect_wordpress

logger = logging.getLogger(__name__)

# Called after each node finishes, with the node name and the state update it returned.
ProgressCallback = Callable[[str, Mapping[str, Any]], None]


class AgentRunner:
    def __init__(self, graph: Any, context: AgentContext) -> None:
        self._graph = graph
        self._context = context

    @classmethod
    @asynccontextmanager
    async def open(cls, settings: Settings) -> AsyncIterator["AgentRunner"]:
        """Open the checkpoint database, the MCP session and an HTTP client for fetches."""
        async with AsyncExitStack() as stack:
            saver = await stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db))
            )
            wordpress = await stack.enter_async_context(connect_wordpress(settings))
            http = await stack.enter_async_context(
                httpx.AsyncClient(timeout=settings.fetch_timeout_s)
            )
            context = AgentContext(
                settings=settings,
                writer=build_model_client(settings, "writer"),
                reviewer=build_model_client(settings, "reviewer"),
                wordpress=wordpress,
                http=http,
            )
            yield cls(build_graph(NODES, saver), context)

    async def start(
        self,
        brief: Brief,
        *,
        run_id: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> RunSnapshot:
        run_id = run_id or uuid.uuid4().hex
        if await self._exists(run_id):
            raise RunStateError(f"run {run_id} already exists")
        logger.info("starting run %s: %s", run_id, brief.topic)
        initial = {"brief": brief, "status": RunStatus.PLANNING}
        return await self._drive(initial, run_id, on_progress)

    async def resume(
        self,
        run_id: str,
        decision: ApprovalDecision,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> RunSnapshot:
        current = await self.get(run_id)
        if current.status != RunStatus.AWAITING_APPROVAL or current.approval is None:
            raise RunStateError(f"run {run_id} is {current.status}, not awaiting approval")
        logger.info("resuming run %s: approved=%s", run_id, decision.approved)
        return await self._drive(Command(resume=decision.model_dump()), run_id, on_progress)

    async def get(self, run_id: str) -> RunSnapshot:
        state = await self._graph.aget_state(_config(run_id))
        if not state.values:
            raise RunNotFoundError(f"no run with id {run_id}")
        values = state.values
        interrupts = [i for task in state.tasks for i in task.interrupts]
        approval = ApprovalRequest.model_validate(interrupts[0].value) if interrupts else None
        review = values.get("review")
        return RunSnapshot(
            run_id=run_id,
            status=values.get("status", RunStatus.PLANNING),
            approval=approval,
            wp_post_id=values.get("wp_post_id"),
            edit_url=values.get("edit_url"),
            review_passed=review.passed if review else None,
            revision_count=values.get("revision_count", 0),
            warnings=values.get("warnings", []),
            error=values.get("error"),
            usage=values.get("usage", []),
        )

    async def _exists(self, run_id: str) -> bool:
        state = await self._graph.aget_state(_config(run_id))
        return bool(state.values)

    async def _drive(
        self, payload: Any, run_id: str, on_progress: ProgressCallback | None
    ) -> RunSnapshot:
        config = _config(run_id)
        try:
            async for chunk in self._graph.astream(
                payload, config, context=self._context, stream_mode="updates"
            ):
                for node, update in chunk.items():
                    if node != "__interrupt__" and on_progress is not None:
                        on_progress(node, update or {})
        except Exception as exc:
            await self._mark_failed(config, exc)
        return await self.get(run_id)

    async def _mark_failed(self, config: Any, exc: Exception) -> None:
        if isinstance(exc, AgentError):
            logger.error("run %s failed: %s", config["configurable"]["thread_id"], exc)
            message = str(exc)
        else:
            logger.exception("run %s crashed", config["configurable"]["thread_id"])
            message = f"internal error: {type(exc).__name__}: {exc}"
        values: dict[str, Any] = {"status": RunStatus.FAILED, "error": message}
        if isinstance(exc, LLMError) and exc.usage is not None:
            values["usage"] = [exc.usage]
        await self._graph.aupdate_state(config, values)


def _config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}
