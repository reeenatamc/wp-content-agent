"""Runs graph work in background tasks so HTTP requests return right away."""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, Coroutine, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from wp_content_agent.errors import RunLimitError, RunNotFoundError, RunStateError
from wp_content_agent.runner import AgentRunner, ensure_can_resume, ensure_can_retry
from wp_content_agent.schemas import ApprovalDecision, Brief, RunSnapshot, RunStatus

logger = logging.getLogger(__name__)

_UNFINISHED = {
    RunStatus.PLANNING,
    RunStatus.GATHERING,
    RunStatus.WRITING,
    RunStatus.REVIEWING,
    RunStatus.PUBLISHING,
}

# What a run reports while the node that failed runs again.
_RETRY_STATUS = {
    "plan": RunStatus.PLANNING,
    "gather_context": RunStatus.GATHERING,
    "write": RunStatus.WRITING,
    "review": RunStatus.REVIEWING,
    "human_approval": RunStatus.AWAITING_APPROVAL,
    "publish_draft": RunStatus.PUBLISHING,
}

# Events a subscriber may fall behind by. A reader slower than this is dropped instead of
# growing memory without bound; it reconnects and gets a fresh snapshot.
SUBSCRIBER_QUEUE_SIZE = 1000


@dataclass(frozen=True)
class RunEvent:
    """One thing that happened in a run. `end` is the last event of a stream, unless the
    reader fell too far behind and was dropped; the stream then just stops."""

    type: Literal["snapshot", "node", "token", "end"]
    data: dict[str, Any] = field(default_factory=dict)


# Put on a queue to tell its reader that no more events follow.
_CLOSED = object()


class RunService:
    """One background task per run id at a time, and at most `max_runs` at once."""

    def __init__(self, runner: AgentRunner, max_runs: int) -> None:
        self._runner = runner
        self._max_runs = max_runs
        self._tasks: dict[str, asyncio.Task[RunSnapshot]] = {}
        # Claimed synchronously, before any await, so two requests cannot both pass a check.
        self._busy: set[str] = set()
        self._subscribers: dict[str, set[asyncio.Queue[Any]]] = {}
        self._closers: set[asyncio.Task[None]] = set()
        # Set by aclose: subscribers then get `end` only after their run is marked failed.
        self._closing = False
        self._ending: dict[str, set[asyncio.Queue[Any]]] = {}

    async def start(self, brief: Brief, run_id: str | None = None) -> tuple[RunSnapshot, bool]:
        """Start a run. With a known run_id, return that run instead (False = not new)."""
        run_id = run_id or uuid.uuid4().hex
        if run_id in self._busy:
            return await self.get(run_id), False
        if len(self._busy) >= self._max_runs:
            raise RunLimitError(f"{self._max_runs} runs are already in progress")
        self._busy.add(run_id)
        try:
            existing = await self._runner.get(run_id)
        except RunNotFoundError:
            self._spawn(run_id, self._runner.start(brief, run_id=run_id, **self._hooks(run_id)))
            return RunSnapshot(run_id=run_id, status=RunStatus.PLANNING), True
        except BaseException:
            self._release(run_id)
            raise
        self._release(run_id)
        return existing, False

    async def resume(self, run_id: str, decision: ApprovalDecision) -> RunSnapshot:
        # Checked here so the caller gets 404 or 409 now, not a failed background task.
        if run_id in self._busy:
            raise RunStateError(f"run {run_id} is still running")
        self._busy.add(run_id)
        try:
            snapshot = await self._runner.get(run_id)
            ensure_can_resume(snapshot, decision)
        except BaseException:
            self._release(run_id)
            raise
        self._spawn(run_id, self._runner.resume(run_id, decision, **self._hooks(run_id)))
        return snapshot.model_copy(update={"approval": None, "status": _status_after(decision)})

    async def retry(self, run_id: str) -> RunSnapshot:
        if run_id in self._busy:
            raise RunStateError(f"run {run_id} is still running")
        if len(self._busy) >= self._max_runs:
            raise RunLimitError(f"{self._max_runs} runs are already in progress")
        self._busy.add(run_id)
        try:
            ensure_can_retry(snapshot := await self._runner.get(run_id))
        except BaseException:
            self._release(run_id)
            raise
        self._spawn(run_id, self._runner.retry(run_id, **self._hooks(run_id)))
        return _retrying(snapshot)

    async def get(self, run_id: str) -> RunSnapshot:
        try:
            snapshot = await self._runner.get(run_id)
        except RunNotFoundError:
            # Started a moment ago: the first checkpoint is not written yet.
            if run_id in self._busy:
                return RunSnapshot(run_id=run_id, status=RunStatus.PLANNING)
            raise
        if run_id in self._busy and snapshot.can_retry:
            # A retry that has not written its first update yet, or is running now.
            return _retrying(snapshot)
        return snapshot

    async def events(self, run_id: str) -> AsyncGenerator[RunEvent]:
        """A snapshot, then node and token events while the run works, then `end`.

        Subscribes before reading the snapshot, so nothing between the two is lost.
        Raises RunNotFoundError for an unknown run when iteration starts.
        """
        queue: asyncio.Queue[Any] = asyncio.Queue(SUBSCRIBER_QUEUE_SIZE)
        self._subscribers.setdefault(run_id, set()).add(queue)
        try:
            # Read before the snapshot: a run that is idle now cannot change under it, and
            # one that finishes during the read still sends `end` to the queue.
            running = run_id in self._busy
            snapshot = await self.get(run_id)
            yield RunEvent("snapshot", _dump(snapshot))
            if not running:
                yield RunEvent("end", _dump(snapshot))
                return
            while (event := await queue.get()) is not _CLOSED:
                yield event
                if event.type == "end":
                    return
        finally:
            subscribers = self._subscribers.get(run_id)
            if subscribers is not None:
                subscribers.discard(queue)
                if not subscribers:
                    del self._subscribers[run_id]

    async def wait(self, run_id: str) -> None:
        """Wait for the background task of a run, if there is one. Used by tests."""
        task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.wait({task})

    async def aclose(self) -> None:
        self._closing = True
        tasks = dict(self._tasks)
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        # A task cancelled before it started never reached the runner's own handling.
        for run_id in tasks:
            try:
                status = (await self._runner.get(run_id)).status
            except RunNotFoundError:
                status = RunStatus.PLANNING
            if status in _UNFINISHED:
                await self._runner.mark_failed(run_id, "the server stopped during the run")
        for run_id, subscribers in self._ending.items():
            await self._publish_end(run_id, subscribers)
        await asyncio.gather(*self._closers)

    def _hooks(self, run_id: str) -> dict[str, Any]:
        """Keyword arguments for the runner that publish progress to the run's subscribers."""

        def on_progress(node: str, update: Mapping[str, Any]) -> None:
            self._publish(run_id, RunEvent("node", {"node": node, "status": _status_of(update)}))

        def on_token(node: str, text: str) -> None:
            self._publish(run_id, RunEvent("token", {"node": node, "text": text}))

        return {"on_progress": on_progress, "on_token": on_token}

    def _publish(self, run_id: str, event: RunEvent) -> None:
        for queue in list(self._subscribers.get(run_id, ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                self._drop(run_id, queue)

    def _drop(self, run_id: str, queue: asyncio.Queue[Any]) -> None:
        """End a subscriber's stream: make room for the close marker, which is always last."""
        self._subscribers.get(run_id, set()).discard(queue)
        while True:
            try:
                queue.put_nowait(_CLOSED)
                return
            except asyncio.QueueFull:
                queue.get_nowait()

    async def _publish_end(self, run_id: str, subscribers: set[asyncio.Queue[Any]]) -> None:
        try:
            end = RunEvent("end", _dump(await self.get(run_id)))
        except Exception:
            logger.warning("could not read the final state of run %s", run_id, exc_info=True)
            for queue in subscribers:
                self._drop(run_id, queue)
            return
        for queue in subscribers:
            try:
                queue.put_nowait(end)
            except asyncio.QueueFull:
                self._drop(run_id, queue)

    def _release(self, run_id: str) -> None:
        """Free a claimed run id, and end the streams of anyone who saw it as busy."""
        self._busy.discard(run_id)
        if run_id in self._tasks:
            return
        # Taken now, so a run started again under the same id gets its own subscribers.
        if not (subscribers := self._subscribers.pop(run_id, set())):
            return
        if self._closing:
            self._ending.setdefault(run_id, set()).update(subscribers)
            return
        closer = asyncio.create_task(self._publish_end(run_id, subscribers))
        self._closers.add(closer)
        closer.add_done_callback(self._closers.discard)

    def _spawn(self, run_id: str, work: Coroutine[Any, Any, RunSnapshot]) -> None:
        task = asyncio.create_task(work, name=f"run-{run_id}")
        self._tasks[run_id] = task
        task.add_done_callback(lambda done: self._finished(run_id, done))

    def _finished(self, run_id: str, task: asyncio.Task[RunSnapshot]) -> None:
        if self._tasks.get(run_id) is task:
            del self._tasks[run_id]
            self._release(run_id)
        if task.cancelled():
            return
        if (exc := task.exception()) is not None:
            logger.error("background work for run %s raised", run_id, exc_info=exc)


def _retrying(snapshot: RunSnapshot) -> RunSnapshot:
    status = _RETRY_STATUS.get(snapshot.failed_node or "", RunStatus.PLANNING)
    return snapshot.model_copy(update={"status": status, "error": None, "can_retry": False})


def _status_after(decision: ApprovalDecision) -> RunStatus:
    if decision.revise:
        return RunStatus.WRITING
    return RunStatus.PUBLISHING if decision.approved else RunStatus.REJECTED


def _dump(snapshot: RunSnapshot) -> dict[str, Any]:
    return snapshot.model_dump(mode="json")


def _status_of(update: Mapping[str, Any]) -> str | None:
    status = update.get("status")
    return None if status is None else str(status)
