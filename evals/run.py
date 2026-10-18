"""Run every brief in evals/briefs through the graph and report quality, tokens and cost.

Runs stop at human approval, so no drafts are created in WordPress. They need a model
API key and a reachable WordPress (scripts/wp-up.sh), because gather_context searches it.

    uv run python evals/run.py
"""

import argparse
import asyncio
import html
import json
import re
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from wp_content_agent.briefs import load_brief
from wp_content_agent.runner import AgentRunner
from wp_content_agent.schemas import RunSnapshot
from wp_content_agent.settings import Settings

HERE = Path(__file__).parent
_TAG = re.compile(r"<[^>]+>")


@dataclass
class EvalResult:
    brief: str
    status: str
    review_passed: bool
    revisions: int
    failed_checks: list[str] = field(default_factory=list)
    words: int = 0
    meta_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    error: str | None = None


def result_from(name: str, snapshot: RunSnapshot, seconds: float) -> EvalResult:
    approval = snapshot.approval
    usage = snapshot.usage
    return EvalResult(
        brief=name,
        status=str(snapshot.status),
        review_passed=bool(snapshot.review_passed),
        revisions=snapshot.revision_count,
        failed_checks=[c.split(":", 1)[0] for c in approval.failed_checks] if approval else [],
        words=_words(approval.content_html) if approval else 0,
        meta_chars=len(approval.meta_description) if approval else 0,
        input_tokens=sum(u.input_tokens for u in usage),
        output_tokens=sum(u.output_tokens for u in usage),
        cache_read_tokens=sum(u.cache_read_tokens for u in usage),
        cache_write_tokens=sum(u.cache_write_tokens for u in usage),
        cost_usd=snapshot.total_cost_usd,
        seconds=round(seconds, 1),
        error=snapshot.error,
    )


def render_markdown(results: list[EvalResult], settings: Settings, when: datetime) -> str:
    passed = sum(r.review_passed for r in results)
    total_input = sum(r.input_tokens for r in results)
    cache_read = sum(r.cache_read_tokens for r in results)
    lines = [
        "# Eval results",
        "",
        f"Run on {when:%Y-%m-%d %H:%M} UTC with `{settings.writer}` writing and "
        f"`{settings.reviewer}` reviewing, max {settings.max_revisions} revisions. "
        "Runs stop at human approval. Produced by `uv run python evals/run.py`.",
        "",
        "| Brief | Review | Revisions | Failed checks | Words | Meta | Tokens in / out "
        "| Cache read | Cost (USD) | Time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        review = "pass" if r.review_passed else ("error" if r.error else "fail")
        lines.append(
            f"| {r.brief} | {review} | {r.revisions} | {', '.join(r.failed_checks) or '-'} "
            f"| {r.words} | {r.meta_chars} | {r.input_tokens:,} / {r.output_tokens:,} "
            f"| {r.cache_read_tokens:,} | {r.cost_usd:.4f} | {r.seconds} |"
        )
    share = cache_read / total_input if total_input else 0.0
    count = len(results) or 1
    lines += [
        "",
        f"Pass rate: {passed}/{len(results)}. "
        f"Average revisions: {sum(r.revisions for r in results) / count:.1f}. "
        f"Total cost: ${sum(r.cost_usd for r in results):.4f}. "
        f"Cache reads: {cache_read:,} of {total_input:,} input tokens ({share:.0%}).",
    ]
    errors = [r for r in results if r.error]
    if errors:
        lines += ["", "Errors:", ""]
        lines += [f"- {r.brief}: {r.error}" for r in errors]
    return "\n".join(lines) + "\n"


async def run_all(briefs: list[Path], settings: Settings) -> list[EvalResult]:
    results: list[EvalResult] = []
    async with AgentRunner.open(settings) as runner:
        # One after another, so later briefs can reuse the cached prompt prefixes.
        for path in briefs:
            brief = load_brief(path)
            print(f"{path.stem}: {brief.topic}", flush=True)
            started = time.monotonic()
            run_id = f"eval-{path.stem}-{uuid.uuid4().hex[:6]}"
            snapshot = await runner.start(brief, run_id=run_id)
            result = result_from(path.stem, snapshot, time.monotonic() - started)
            print(f"  {result.status}, review passed: {result.review_passed}", flush=True)
            results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--briefs", type=Path, default=HERE / "briefs")
    parser.add_argument("--out", type=Path, default=HERE / "results.md")
    args = parser.parse_args()

    briefs = sorted(args.briefs.glob("*.yaml"))
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(checkpoint_db=Path(tmp) / "evals.sqlite")
        results = asyncio.run(run_all(briefs, settings))
    args.out.write_text(render_markdown(results, settings, datetime.now(UTC)))
    raw = [asdict(r) for r in results]
    args.out.with_suffix(".json").write_text(json.dumps(raw, indent=2) + "\n")
    print(f"wrote {args.out}")


def _words(content_html: str) -> int:
    return len(html.unescape(_TAG.sub(" ", content_html)).split())


if __name__ == "__main__":
    main()
