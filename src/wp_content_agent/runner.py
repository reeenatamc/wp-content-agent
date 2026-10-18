"""Runs the graph: start, resume after approval, and report where a run stands."""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import aiosqlite
import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command, StateSnapshot

from wp_content_agent.errors import (
    AgentError,
    LLMError,
    RunNotFoundError,
    RunStateError,
    RunStoppedError,
)
from wp_content_agent.graph import AgentContext, build_graph
from wp_content_agent.graph.state import checkpoint_types
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


def checkpoint_serializer() -> JsonPlusSerializer:
    """A serializer that loads the state's own models back from checkpoints.

    LangGraph warns on unregistered types today and will refuse them later, which
    would make every waiting run impossible to resume.
    """
    return JsonPlusSerializer(allowed_msgpack_modules=checkpoint_types())


class AgentRunner:
    """Drives runs in one process. Each run id has a lock, so a double approve cannot
    publish twice; several processes sharing one checkpoint database are not covered."""

    def __init__(self, graph: Any, context: AgentContext) -> None:
        self._graph = graph
        self._context = context
        self._locks: dict[str, asyncio.Lock] = {}

    @classmethod
    @asynccontextmanager
    async def open(cls, settings: Settings, *, models: bool = True) -> AsyncIterator["AgentRunner"]:
        """Open the checkpoint database, the MCP session and an HTTP client for fetches.

        models=False is for commands that never call a model: they then work without
        an API key, and a model call fails when it is made.
        """
        # Model clients first: when models are needed, a missing API key should fail
        # before anything is opened.
        writer = build_model_client(settings, "writer", required=models)
        reviewer = build_model_client(settings, "reviewer", required=models)
        settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncExitStack() as stack:
            conn = await stack.enter_async_context(aiosqlite.connect(str(settings.checkpoint_db)))
            saver = AsyncSqliteSaver(conn, serde=checkpoint_serializer())
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
            ensure_can_resume(_snapshot(run_id, await self._state(run_id)), decision)
            logger.info("resuming run %s: %s", run_id, _decision_kind(decision))
            payload: Command[Any] = Command(resume=decision.model_dump())
            return await self._drive(payload, run_id, on_progress)

    async def retry(
        self, run_id: str, *, on_progress: ProgressCallback | None = None
    ) -> RunSnapshot:
        """Run the node that failed again, from the checkpoint taken just before it."""
        async with self._lock(run_id):
            latest = await self._state(run_id)
            ensure_can_retry(_snapshot(run_id, latest))
            failed_node = latest.values["failed_node"]
            logger.info("retrying %s for run %s", failed_node, run_id)
            before = await self._checkpoint_before(run_id, failed_node)
            if before is None:
                raise RunStateError(f"run {run_id} has no checkpoint before {failed_node}")
            # Billed by the failed attempt but absent from the fork's lineage.
            billed = latest.values.get("usage", [])
            missing = billed[len(before.values.get("usage", [])) :]
            update: dict[str, Any] = {"failed_node": failed_node, "usage": missing}
            # Streaming from an old checkpoint forks the history. The update is applied as
            # input, so no edge is evaluated and the pending node stays the one that failed.
            return await self._drive(
                Command(update=update),
                run_id,
                on_progress,
                start_from=before.config,
                retrying=failed_node,
            )

    async def get(self, run_id: str) -> RunSnapshot:
        return _snapshot(run_id, await self._state(run_id))

    async def mark_failed(self, run_id: str, message: str) -> None:
        """Record a run that was stopped from outside, such as a server shutdown."""
        await self._mark_failed(_config(run_id), RunStoppedError(message))

    def _lock(self, run_id: str) -> asyncio.Lock:
        return self._locks.setdefault(run_id, asyncio.Lock())

    async def _state(self, run_id: str) -> StateSnapshot:
        state: StateSnapshot = await self._graph.aget_state(_config(run_id))
        return state

    async def _checkpoint_before(self, run_id: str, node: str) -> StateSnapshot | None:
        """The newest checkpoint whose pending node is `node`: where it last started from."""
        async for state in self._graph.aget_state_history(_config(run_id)):
            if state.next == (node,):
                found: StateSnapshot = state
                return found
        return None

    async def _drive(
        self,
        payload: Any,
        run_id: str,
        on_progress: ProgressCallback | None,
        *,
        start_from: RunnableConfig | None = None,
        retrying: str = "",
    ) -> RunSnapshot:
        config = _config(run_id)
        try:
            async for chunk in self._graph.astream(
                payload,
                config if start_from is None else start_from,
                context=self._context,
                stream_mode="updates",
            ):
                for node, update in chunk.items():
                    if node != "__interrupt__" and on_progress is not None:
                        _notify(on_progress, node, update or {})
        except BaseException as exc:
            # Cancellation and Ctrl-C also end the run, so it never stays half-done.
            await self._mark_failed(config, exc, retrying)
            if not isinstance(exc, Exception):
                raise
        return await self.get(run_id)

    async def _mark_failed(self, config: Any, exc: BaseException, retrying: str = "") -> None:
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
            # The node that raised is still pending; retry restarts from just before it.
            pending = (await self._graph.aget_state(config)).next
        except Exception:
            # Marking the run failed matters more than knowing where it stopped.
            logger.warning("could not read where run %s stopped", run_id, exc_info=True)
            pending = ()
        # Always written, so a failure elsewhere never inherits an earlier one. A retry that
        # fails before its fork is written still points at the node it was retrying.
        values["failed_node"] = pending[0] if pending else retrying
        try:
            # Written as the last node, so the run has no pending tasks and cannot resume.
            await self._graph.aupdate_state(config, values, as_node="publish_draft")
        except Exception:
            logger.exception("could not mark run %s as failed", run_id)


def ensure_can_retry(snapshot: RunSnapshot) -> None:
    """Raise RunStateError unless the run failed in a node. Nothing has run yet."""
    if not snapshot.can_retry:
        raise RunStateError(
            f"run {snapshot.run_id} cannot be retried; only a run that failed in a node"
            " can be retried"
        )


def ensure_can_resume(snapshot: RunSnapshot, decision: ApprovalDecision) -> None:
    """Raise RunStateError when the run cannot take this decision. Nothing has run yet."""
    if snapshot.approval is None:
        raise RunStateError(f"run {snapshot.run_id} is {snapshot.status}, not awaiting approval")
    if decision.revise and snapshot.approval.human_revisions_left == 0:
        raise RunStateError(
            f"run {snapshot.run_id} has no human revisions left; approve or reject the draft"
        )


def _decision_kind(decision: ApprovalDecision) -> str:
    if decision.revise:
        return "revise"
    return "approve" if decision.approved else "reject"


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
    failed_node = values.get("failed_node") or None
    # Failed, or a retry that died mid-way: the newest checkpoint is still the fork the
    # retry wrote. Once the node runs, failed_node stays set, but the next checkpoint
    # comes from the loop, so a later pass through the same node is not retryable.
    died_mid_retry = (state.metadata or {}).get("source") == "fork" and state.next == (failed_node,)
    can_retry = (
        failed_node is not None
        and not values.get("wp_post_id")
        and approval is None
        and (status == RunStatus.FAILED or died_mid_retry)
    )
    return RunSnapshot(
        run_id=run_id,
        status=status,
        approval=approval,
        wp_post_id=values.get("wp_post_id"),
        edit_url=values.get("edit_url"),
        review_passed=review.passed if review else None,
        revision_count=values.get("revision_count", 0),
        warnings=values.get("warnings", []),
        error=values.get("error") or None,
        can_retry=can_retry,
        failed_node=failed_node,
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
