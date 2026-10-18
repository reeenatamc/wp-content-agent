"""Command line entry point: `wpagent run --brief brief.yaml` and friends."""

import argparse
import asyncio
import html
import ipaddress
import logging
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, Protocol, TextIO

from pydantic import ValidationError

from wp_content_agent.briefs import load_brief
from wp_content_agent.errors import AgentError
from wp_content_agent.runner import AgentRunner, ProgressCallback, TokenCallback
from wp_content_agent.schemas import ApprovalDecision, ApprovalRequest, RunSnapshot, RunStatus
from wp_content_agent.settings import Settings


class OpenRunner(Protocol):
    """AgentRunner.open, or a stand-in in tests. models=False is for commands that never
    call a model, so they work without an API key."""

    def __call__(
        self, settings: Settings, *, models: bool = True
    ) -> AbstractAsyncContextManager[AgentRunner]: ...


# Reads one line of input, like input(). Commands get None when there is no terminal.
Ask = Callable[[str], str]

EXIT_OK, EXIT_FAILED = 0, 1
PREVIEW_CHARS = 600

_HIDDEN = re.compile(r"<(script|style)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")


def main(
    argv: Sequence[str] | None = None,
    *,
    open_runner: OpenRunner = AgentRunner.open,
    ask: Ask | None = None,
    out: TextIO | None = None,
) -> int:
    args = _parser().parse_args(argv)
    out = out or sys.stdout
    if ask is None and sys.stdin.isatty():
        ask = input
    try:
        settings = Settings()
    except ValidationError as exc:
        print(f"error: invalid settings: {exc}", file=sys.stderr)
        return EXIT_FAILED
    logging.basicConfig(
        level=settings.log_level,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    command: Callable[..., Any] = args.command
    if command is _serve:
        return _serve(args, settings, open_runner)
    try:
        snapshot: RunSnapshot = asyncio.run(command(args, settings, open_runner, ask, out))
    except AgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    return EXIT_FAILED if snapshot.status == RunStatus.FAILED else EXIT_OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wpagent", description=__doc__)
    commands = parser.add_subparsers(required=True, metavar="COMMAND")

    run = commands.add_parser("run", help="run a brief through the graph")
    run.add_argument("--brief", type=Path, required=True, help="brief file, YAML or JSON")
    run.add_argument("--run-id", help="id for the run; generated when omitted")
    run.add_argument(
        "--yes", action="store_true", help="approve the draft without asking (still a draft)"
    )
    run.set_defaults(command=_run)

    status = commands.add_parser("status", help="show where a run stands")
    status.add_argument("run_id")
    status.set_defaults(command=_status)

    for name, approved in (("approve", True), ("reject", False)):
        decide = commands.add_parser(name, help=f"{name} a run that is awaiting approval")
        decide.add_argument("run_id")
        decide.add_argument("--note", default="", help="note stored with the decision")
        decide.set_defaults(command=_decide, approved=approved)

    retry = commands.add_parser(
        "retry", help="run the failed step again for a run that failed in a node"
    )
    retry.add_argument("run_id")
    retry.set_defaults(command=_retry)

    revise = commands.add_parser("revise", help="send a waiting draft back to the writer")
    revise.add_argument("run_id")
    revise.add_argument("--note", required=True, help="what the writer should change")
    revise.set_defaults(command=_revise)
    serve = commands.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--host", help="defaults to API_HOST")
    serve.add_argument("--port", type=int, help="defaults to API_PORT")
    serve.set_defaults(command=_serve)
    return parser


def _serve(args: argparse.Namespace, settings: Settings, open_runner: OpenRunner) -> int:
    import uvicorn

    from wp_content_agent.api import create_app

    host = args.host or settings.api_host
    port = settings.api_port if args.port is None else args.port
    if not 0 <= port <= 65535:
        print(f"error: invalid port {port}", file=sys.stderr)
        return EXIT_FAILED
    if settings.api_token is None and not _is_loopback(host):
        print(
            f"error: refusing to listen on {host} without API_TOKEN; the run endpoints"
            " would be open to anyone who can reach the port",
            file=sys.stderr,
        )
        return EXIT_FAILED
    app = create_app(settings, open_runner=open_runner)
    try:
        # uvicorn reports startup failures (missing API key, MCP server down) itself.
        uvicorn.run(app, host=host, port=port, log_level=settings.log_level.lower())
    except KeyboardInterrupt:
        return 130
    return EXIT_OK


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


async def _run(
    args: argparse.Namespace,
    settings: Settings,
    open_runner: OpenRunner,
    ask: Ask | None,
    out: TextIO,
) -> RunSnapshot:
    brief = load_brief(args.brief)
    print(f"brief: {brief.topic}", file=out)
    progress, tokens = _reporters(out)
    async with open_runner(settings) as runner:
        snapshot = await runner.start(
            brief, run_id=args.run_id, on_progress=progress, on_token=tokens
        )
        print(f"run id: {snapshot.run_id}", file=out)
        # A revision sends the run back to the writer and it stops at approval again.
        while snapshot.approval is not None:
            decision = _decision(snapshot.approval, args.yes, ask, out)
            if decision is None:
                print(
                    f"waiting for approval: wpagent approve {snapshot.run_id}"
                    f" (or wpagent reject {snapshot.run_id}, wpagent revise {snapshot.run_id}"
                    " --note TEXT)",
                    file=out,
                )
                return snapshot
            snapshot = await runner.resume(
                snapshot.run_id, decision, on_progress=progress, on_token=tokens
            )
    _summary(snapshot, out)
    return snapshot


async def _status(
    args: argparse.Namespace,
    settings: Settings,
    open_runner: OpenRunner,
    ask: Ask | None,
    out: TextIO,
) -> RunSnapshot:
    async with open_runner(settings, models=False) as runner:
        snapshot = await runner.get(args.run_id)
    if snapshot.approval is not None:
        _show_approval(snapshot.approval, out)
    _summary(snapshot, out)
    return snapshot


async def _decide(
    args: argparse.Namespace,
    settings: Settings,
    open_runner: OpenRunner,
    ask: Ask | None,
    out: TextIO,
) -> RunSnapshot:
    decision = ApprovalDecision(approved=args.approved, note=args.note)
    async with open_runner(settings, models=False) as runner:
        progress, tokens = _reporters(out)
        snapshot = await runner.resume(args.run_id, decision, on_progress=progress, on_token=tokens)
    _summary(snapshot, out)
    return snapshot


async def _revise(
    args: argparse.Namespace,
    settings: Settings,
    open_runner: OpenRunner,
    ask: Ask | None,
    out: TextIO,
) -> RunSnapshot:
    try:
        decision = ApprovalDecision(approved=False, note=args.note, revise=True)
    except ValidationError as exc:
        raise AgentError("the note must say what to change") from exc
    async with open_runner(settings) as runner:
        progress, tokens = _reporters(out)
        snapshot = await runner.resume(args.run_id, decision, on_progress=progress, on_token=tokens)
    _summary(snapshot, out)
    return snapshot


async def _retry(
    args: argparse.Namespace,
    settings: Settings,
    open_runner: OpenRunner,
    ask: Ask | None,
    out: TextIO,
) -> RunSnapshot:
    async with open_runner(settings) as runner:
        progress, tokens = _reporters(out)
        snapshot = await runner.retry(args.run_id, on_progress=progress, on_token=tokens)
    _summary(snapshot, out)
    return snapshot


def _decision(
    request: ApprovalRequest, auto_approve: bool, ask: Ask | None, out: TextIO
) -> ApprovalDecision | None:
    _show_approval(request, out)
    if auto_approve:
        if not request.review_passed:
            # --yes is for unattended demos; a draft that failed review needs a person.
            print("review failed, so --yes does not approve it", file=out)
            return None
        print("approved with --yes", file=out)
        return ApprovalDecision(approved=True)
    if ask is None:
        return None
    try:
        return _ask_decision(request, ask, out)
    except EOFError:
        print(file=out)
        return None


def _ask_decision(request: ApprovalRequest, ask: Ask, out: TextIO) -> ApprovalDecision | None:
    while True:
        answer = ask(
            "create the draft? [a]pprove, [r]eject, [c]hange, [v]iew HTML, [l]ater: "
        ).strip()
        match answer.lower()[:1]:
            case "a":
                return ApprovalDecision(approved=True, note=ask("note (optional): ").strip())
            case "r":
                return ApprovalDecision(approved=False, note=ask("reason (optional): ").strip())
            case "c" if request.human_revisions_left == 0:
                print("no human revisions left; approve or reject", file=out)
            case "c":
                # A blank note goes back to the menu.
                if note := ask("what should change (blank to go back): ").strip():
                    return ApprovalDecision(approved=False, note=note, revise=True)
            case "v":
                print(request.content_html, file=out)
            case "l":
                return None
            case _:
                print("answer a, r, c, v or l", file=out)


def _show_approval(request: ApprovalRequest, out: TextIO) -> None:
    review = "passed" if request.review_passed else "failed"
    lines = [
        "",
        f"title:     {request.title}",
        f"slug:      {request.slug}",
        f"meta:      {request.meta_description} ({len(request.meta_description)} chars)",
        f"category:  {', '.join(request.category_names) or '-'}",
        f"tags:      {', '.join(request.tag_names) or '-'}",
        f"review:    {review} after {request.revision_count} revision(s)",
        *(f"  - {check}" for check in request.failed_checks),
        f"changes:   {request.human_revisions_left} human revision(s) left",
        "",
        _preview(request.content_html),
        "",
    ]
    print("\n".join(lines), file=out)


def _preview(content_html: str) -> str:
    text = html.unescape(_TAG.sub(" ", _HIDDEN.sub(" ", content_html)))
    text = " ".join(text.split())
    return text if len(text) <= PREVIEW_CHARS else f"{text[:PREVIEW_CHARS].rstrip()}..."


def _reporters(out: TextIO) -> tuple[ProgressCallback, TokenCallback]:
    """Progress lines and the writer's streamed text, which share the terminal.

    The writer returns structured output, so the text is the draft as JSON.
    """
    mid_line = False

    def report(node: str, update: Mapping[str, Any]) -> None:
        nonlocal mid_line
        if mid_line:
            # Streamed text has no newline; progress lines start on their own line.
            print(file=out)
            mid_line = False
        cost = ""
        if entries := update.get("usage", []):
            tokens = sum(entry.input_tokens + entry.output_tokens for entry in entries)
            cached = sum(entry.cache_read_tokens for entry in entries)
            cost = f"  {tokens:>7,} tok  ${sum(entry.cost_usd for entry in entries):.4f}"
            if cached:
                cost += f"  cache read {cached:,}"
        print(f"  {node:<15} {_detail(node, update)}{cost}", file=out, flush=True)

    def stream(node: str, text: str) -> None:
        nonlocal mid_line
        out.write(text)
        out.flush()
        mid_line = True

    return report, stream


def _detail(node: str, update: Mapping[str, Any]) -> str:
    match node:
        case "plan" if "outline" in update:
            return f"{len(update['outline'].sections)} sections"
        case "gather_context" if "context" in update:
            notes = update["context"]
            return f"{len(notes.sources)} sources, {len(notes.related_posts)} related posts"
        case "write" if "draft" in update:
            revision = update.get("revision_count", 0)
            suffix = f" (revision {revision})" if revision else ""
            return f"{update['draft'].title}{suffix}"
        case "review" if "review" in update:
            review = update["review"]
            if review.passed:
                return "passed"
            return "failed: " + ", ".join(check.name for check in review.failed_checks)
        case "human_approval":
            return str(update.get("status", ""))
        case "publish_draft" if "wp_post_id" in update:
            return f"post {update['wp_post_id']}"
    return "done"


def _summary(snapshot: RunSnapshot, out: TextIO) -> None:
    lines = [f"status: {snapshot.status}"]
    if snapshot.edit_url:
        lines.append(f"edit:   {snapshot.edit_url}")
    if snapshot.error:
        lines.append(f"error:  {snapshot.error}")
    if snapshot.can_retry:
        hint = f"retry:  wpagent retry {snapshot.run_id} (runs {snapshot.failed_node} again"
        if snapshot.failed_node == "publish_draft":
            # A request that timed out may still have created the post on the site.
            hint += "; check WordPress for the draft first, a timeout may have created it"
        lines.append(hint + ")")
    lines.extend(f"warning: {warning}" for warning in snapshot.warnings)
    lines.append(f"usage:  {snapshot.total_tokens:,} tokens, ${snapshot.total_cost_usd:.4f}")
    print("\n".join(lines), file=out)


if __name__ == "__main__":
    sys.exit(main())
