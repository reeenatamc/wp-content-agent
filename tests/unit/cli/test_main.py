import io
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from tests.fake_graph import fake_runner
from wp_content_agent.cli.main import EXIT_FAILED, EXIT_OK, main
from wp_content_agent.errors import LLMError
from wp_content_agent.graph import AgentState
from wp_content_agent.runner import AgentRunner
from wp_content_agent.schemas import CheckResult, Review, RunStatus
from wp_content_agent.settings import Settings

BRIEF_YAML = """\
topic: Caching in Django
audience: Backend developers
keywords: [django]
"""


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # no .env from the repo
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")


@pytest.fixture
def brief(tmp_path: Path) -> Path:
    path = tmp_path / "brief.yaml"
    path.write_text(BRIEF_YAML)
    return path


def _opener(runner: AgentRunner) -> Any:
    @asynccontextmanager
    async def open_runner(settings: Settings) -> AsyncIterator[AgentRunner]:
        yield runner

    return open_runner


def _answers(*values: str) -> Any:
    replies: Iterator[str] = iter(values)
    return lambda prompt: next(replies)


def _cli(argv: list[str], runner: AgentRunner, *answers: str, tty: bool = True) -> tuple[int, str]:
    out = io.StringIO()
    ask = _answers(*answers) if tty else None
    code = main(argv, open_runner=_opener(runner), ask=ask, out=out)
    return code, out.getvalue()


def test_run_approved_creates_draft(brief: Path) -> None:
    code, output = _cli(["run", "--brief", str(brief), "--run-id", "r1"], fake_runner(), "a", "")

    assert code == EXIT_OK
    assert "status: draft_created" in output
    assert "post.php?post=8&action=edit" in output


def test_run_streams_each_node(brief: Path) -> None:
    _, output = _cli(["run", "--brief", str(brief)], fake_runner(), "a", "")

    for node in ("plan", "gather_context", "write", "review", "human_approval", "publish_draft"):
        assert f"  {node}" in output


def test_run_shows_approval_details(brief: Path) -> None:
    _, output = _cli(["run", "--brief", str(brief)], fake_runner(), "r", "off topic")

    assert "title:     Caching Django querysets" in output
    assert "tags:      django" in output
    assert "status: rejected" in output


def test_view_prints_html_then_asks_again(brief: Path) -> None:
    _, output = _cli(["run", "--brief", str(brief)], fake_runner(), "v", "x", "a", "")

    assert "<p>x</p>" in output
    assert "answer a, r, v or l" in output
    assert "status: draft_created" in output


def test_later_leaves_run_pending_and_approve_finishes_it(brief: Path) -> None:
    runner = fake_runner()

    code, output = _cli(["run", "--brief", str(brief), "--run-id", "r1"], runner, "l")
    assert code == EXIT_OK
    assert "wpagent approve r1" in output

    code, output = _cli(["approve", "r1", "--note", "ok"], runner)
    assert code == EXIT_OK
    assert "status: draft_created" in output


def test_without_terminal_run_waits_for_approval(brief: Path) -> None:
    runner = fake_runner()

    _, output = _cli(["run", "--brief", str(brief), "--run-id", "r1"], runner, tty=False)
    _, status = _cli(["status", "r1"], runner)

    assert "waiting for approval" in output
    assert "status: awaiting_approval" in status


def test_yes_approves_without_asking(brief: Path) -> None:
    code, output = _cli(["run", "--brief", str(brief), "--yes"], fake_runner(), tty=False)

    assert code == EXIT_OK
    assert "approved with --yes" in output
    assert "status: draft_created" in output


def test_failed_run_exits_nonzero(brief: Path) -> None:
    def write(state: AgentState) -> dict[str, Any]:
        raise LLMError("write failed")

    code, output = _cli(["run", "--brief", str(brief)], fake_runner(write=write))

    assert code == EXIT_FAILED
    assert "error:  write failed" in output


def test_invalid_brief_exits_with_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("topic: x\n")

    code, _ = _cli(["run", "--brief", str(path)], fake_runner())

    assert code == EXIT_FAILED
    assert "not a valid brief" in capsys.readouterr().err


def test_reject_unknown_run_exits_with_error(capsys: pytest.CaptureFixture[str]) -> None:
    code, _ = _cli(["reject", "missing"], fake_runner())

    assert code == EXIT_FAILED
    assert "no run with id missing" in capsys.readouterr().err


def test_yes_does_not_approve_a_failed_review(brief: Path) -> None:
    def review(state: AgentState) -> dict[str, Any]:
        check = CheckResult(name="structure", passed=False, note="no conclusion")
        done = state.get("revision_count", 0) >= 2
        status = RunStatus.AWAITING_APPROVAL if done else RunStatus.WRITING
        return {"review": Review(passed=False, checks=[check], fixes=[]), "status": status}

    code, output = _cli(["run", "--brief", str(brief), "--yes"], fake_runner(review=review))

    assert code == EXIT_OK
    assert "--yes does not approve it" in output
    assert "waiting for approval" in output


def test_end_of_input_leaves_run_pending(brief: Path) -> None:
    def closed(prompt: str) -> str:
        raise EOFError

    out = io.StringIO()
    code = main(
        ["run", "--brief", str(brief)], open_runner=_opener(fake_runner()), ask=closed, out=out
    )

    assert code == EXIT_OK
    assert "waiting for approval" in out.getvalue()


def test_unknown_log_level_is_a_settings_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = fake_runner()
    monkeypatch.setenv("LOG_LEVEL", "verbose")

    code, _ = _cli(["status", "r1"], runner)

    assert code == EXIT_FAILED
    assert "unknown log level" in capsys.readouterr().err


@pytest.mark.parametrize("host", ["0.0.0.0", "example.com"])
def test_serve_refuses_public_host_without_token(
    host: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _cli(["serve", "--host", host], fake_runner())

    assert code == EXIT_FAILED
    assert "without API_TOKEN" in capsys.readouterr().err


def test_serve_passes_host_port_and_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: calls.append(kwargs))

    code, _ = _cli(["serve", "--port", "0"], fake_runner())

    assert code == EXIT_OK
    assert (calls[0]["host"], calls[0]["port"]) == ("127.0.0.1", 0)
