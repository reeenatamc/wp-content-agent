"""Runs the graph: start, resume after approval, and report where a run stands."""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import httpx
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command, StateSnapshot

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
    """Drives runs in one process. Each run id has a lock, so a double approve cannot
    publish twice; several processes sharing one checkpoint database are not covered."""

    def __init__(self, graph: Any, context: AgentContext) -> None:
        self._graph = graph
        self._context = context
        self._locks: dict[str, asyncio.Lock] = {}

    @classmethod
    @asynccontextmanager
    async def open(cls, settings: Settings) -> AsyncIterator["AgentRunner"]:
        """Open the checkpoint database, the MCP session and an HTTP client for fetches."""
        # Model clients first: a missing API key should fail before anything is opened.
        writer = build_model_client(settings, "writer")
        reviewer = build_model_client(settings, "reviewer")
        settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
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
                writer=writer,
                reviewer=reviewer,
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
        async with self._lock(run_id):
            if (await self._state(run_id)).values:
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
        async with self._lock(run_id):
            current = _snapshot(run_id, await self._state(run_id))
            if current.approval is None:
                raise RunStateError(f"run {run_id} is {current.status}, not awaiting approval")
            logger.info("resuming run %s: approved=%s", run_id, decision.approved)
            payload: Command[Any] = Command(resume=decision.model_dump())
            return await self._drive(payload, run_id, on_progress)

    async def get(self, run_id: str) -> RunSnapshot:
        return _snapshot(run_id, await self._state(run_id))

    def _lock(self, run_id: str) -> asyncio.Lock:
        return self._locks.setdefault(run_id, asyncio.Lock())

    async def _state(self, run_id: str) -> StateSnapshot:
        state: StateSnapshot = await self._graph.aget_state(_config(run_id))
        return state

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
                        _notify(on_progress, node, update or {})
        except BaseException as exc:
            # Cancellation and Ctrl-C also end the run, so it never stays half-done.
            await self._mark_failed(config, exc)
            if not isinstance(exc, Exception):
                raise
        return await self.get(run_id)

    async def _mark_failed(self, config: Any, exc: BaseException) -> None:
        run_id = config["configurable"]["thread_id"]
        if isinstance(exc, AgentError):
            logger.error("run %s failed: %s", run_id, exc)
            message = str(exc)
        elif isinstance(exc, Exception):
            # Details stay in the log: an exception message can carry URLs or paths.
            logger.error("run %s crashed", run_id, exc_info=exc)
            message = f"internal error ({type(exc).__name__}); see the logs"
        else:
            logger.warning("run %s stopped: %s", run_id, type(exc).__name__)
            message = f"run stopped ({type(exc).__name__})"
        values: dict[str, Any] = {"status": RunStatus.FAILED, "error": message}
        if isinstance(exc, LLMError) and exc.usage is not None:
            values["usage"] = [exc.usage]
        try:
            # Written as the last node, so the run has no pending tasks and cannot resume.
            await self._graph.aupdate_state(config, values, as_node="publish_draft")
        except Exception:
            logger.exception("could not mark run %s as failed", run_id)


def _snapshot(run_id: str, state: StateSnapshot) -> RunSnapshot:
    if not state.values:
        raise RunNotFoundError(f"no run with id {run_id}")
    values = state.values
    status = values.get("status", RunStatus.PLANNING)
    interrupts = [i for task in state.tasks for i in task.interrupts]
    approval = None
    if interrupts and status == RunStatus.AWAITING_APPROVAL:
        approval = ApprovalRequest.model_validate(interrupts[0].value)
    review = values.get("review")
    return RunSnapshot(
        run_id=run_id,
        status=status,
        approval=approval,
        wp_post_id=values.get("wp_post_id"),
        edit_url=values.get("edit_url"),
        review_passed=review.passed if review else None,
        revision_count=values.get("revision_count", 0),
        warnings=values.get("warnings", []),
        error=values.get("error"),
        usage=values.get("usage", []),
    )


def _notify(callback: ProgressCallback, node: str, update: Mapping[str, Any]) -> None:
    # A broken terminal or socket must not fail the run.
    try:
        callback(node, update)
    except Exception:
        logger.warning("progress callback failed for node %s", node, exc_info=True)


def _config(run_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}
