"""Runs graph work in background tasks so HTTP requests return right away."""

import asyncio
import logging
import uuid
from collections.abc import Coroutine
from typing import Any

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


class RunService:
    """One background task per run id at a time, and at most `max_runs` at once."""

    def __init__(self, runner: AgentRunner, max_runs: int) -> None:
        self._runner = runner
        self._max_runs = max_runs
        self._tasks: dict[str, asyncio.Task[RunSnapshot]] = {}
        # Claimed synchronously, before any await, so two requests cannot both pass a check.
        self._busy: set[str] = set()

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
            self._spawn(run_id, self._runner.start(brief, run_id=run_id))
            return RunSnapshot(run_id=run_id, status=RunStatus.PLANNING), True
        except BaseException:
            self._busy.discard(run_id)
            raise
        self._busy.discard(run_id)
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
            self._busy.discard(run_id)
            raise
        self._spawn(run_id, self._runner.resume(run_id, decision))
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
            self._busy.discard(run_id)
            raise
        self._spawn(run_id, self._runner.retry(run_id))
        return snapshot.model_copy(
            update={"status": RunStatus.PUBLISHING, "error": None, "can_retry": False}
        )

    async def get(self, run_id: str) -> RunSnapshot:
        try:
            snapshot = await self._runner.get(run_id)
        except RunNotFoundError:
            # Started a moment ago: the first checkpoint is not written yet.
            if run_id in self._busy:
                return RunSnapshot(run_id=run_id, status=RunStatus.PLANNING)
            raise
        if run_id in self._busy and snapshot.can_retry:
            # A retry that has not written its first update yet, or is publishing now.
            return snapshot.model_copy(
                update={"status": RunStatus.PUBLISHING, "error": None, "can_retry": False}
            )
        return snapshot

    async def wait(self, run_id: str) -> None:
        """Wait for the background task of a run, if there is one. Used by tests."""
        task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.wait({task})

    async def aclose(self) -> None:
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

    def _spawn(self, run_id: str, work: Coroutine[Any, Any, RunSnapshot]) -> None:
        task = asyncio.create_task(work, name=f"run-{run_id}")
        self._tasks[run_id] = task
        task.add_done_callback(lambda done: self._finished(run_id, done))

    def _finished(self, run_id: str, task: asyncio.Task[RunSnapshot]) -> None:
        if self._tasks.get(run_id) is task:
            del self._tasks[run_id]
            self._busy.discard(run_id)
        if task.cancelled():
            return
        if (exc := task.exception()) is not None:
            logger.error("background work for run %s raised", run_id, exc_info=exc)


def _status_after(decision: ApprovalDecision) -> RunStatus:
    if decision.revise:
        return RunStatus.WRITING
    return RunStatus.PUBLISHING if decision.approved else RunStatus.REJECTED
