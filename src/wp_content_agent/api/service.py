"""Runs graph work in background tasks so HTTP requests return right away."""

import asyncio
import logging
import uuid
from collections.abc import Coroutine
from typing import Any

from wp_content_agent.errors import RunNotFoundError, RunStateError
from wp_content_agent.runner import AgentRunner
from wp_content_agent.schemas import ApprovalDecision, Brief, RunSnapshot, RunStatus

logger = logging.getLogger(__name__)


class RunService:
    def __init__(self, runner: AgentRunner) -> None:
        self._runner = runner
        self._tasks: dict[str, asyncio.Task[RunSnapshot]] = {}

    def start(self, brief: Brief) -> RunSnapshot:
        run_id = uuid.uuid4().hex
        self._spawn(run_id, self._runner.start(brief, run_id=run_id))
        return RunSnapshot(run_id=run_id, status=RunStatus.PLANNING)

    async def resume(self, run_id: str, decision: ApprovalDecision) -> RunSnapshot:
        # Checked here so the caller gets 404 or 409 now, not a failed background task.
        if run_id in self._tasks:
            raise RunStateError(f"run {run_id} is still running")
        snapshot = await self._runner.get(run_id)
        if snapshot.approval is None:
            raise RunStateError(f"run {run_id} is {snapshot.status}, not awaiting approval")
        self._spawn(run_id, self._runner.resume(run_id, decision))
        status = RunStatus.PUBLISHING if decision.approved else RunStatus.REJECTED
        return snapshot.model_copy(update={"status": status, "approval": None})

    async def get(self, run_id: str) -> RunSnapshot:
        try:
            return await self._runner.get(run_id)
        except RunNotFoundError:
            # Started a moment ago: the first checkpoint is not written yet.
            if run_id in self._tasks:
                return RunSnapshot(run_id=run_id, status=RunStatus.PLANNING)
            raise

    async def wait(self, run_id: str) -> None:
        """Wait for the background task of a run, if there is one. Used by tests."""
        task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.wait({task})

    async def aclose(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        # The runner marks cancelled runs as failed before the task ends.
        await asyncio.gather(*tasks, return_exceptions=True)

    def _spawn(self, run_id: str, work: Coroutine[Any, Any, RunSnapshot]) -> None:
        task = asyncio.create_task(work, name=f"run-{run_id}")
        self._tasks[run_id] = task
        task.add_done_callback(lambda done: self._finished(run_id, done))

    def _finished(self, run_id: str, task: asyncio.Task[RunSnapshot]) -> None:
        self._tasks.pop(run_id, None)
        if task.cancelled():
            return
        if (exc := task.exception()) is not None:
            logger.error("background work for run %s raised", run_id, exc_info=exc)
