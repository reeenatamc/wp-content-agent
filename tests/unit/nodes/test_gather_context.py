import httpx
import pytest
from langchain_core.messages import ToolMessage

from tests.fakes import FakeChatModel, ai, fake_client
from tests.unit.nodes.fake_wordpress import FakeGateway
from tests.unit.nodes.helpers import BRIEF, runtime
from tests.unit.nodes.test_plan import OUTLINE
from wp_content_agent.errors import LLMError
from wp_content_agent.nodes.gather_context import ResearchFindings, gather_context
from wp_content_agent.schemas import RelatedPost, RunStatus, SourceNote

SOURCE = "https://example.com/django-cache"
PAGE = (
    "<html><title>Cache guide</title><body><p>Per-view caching cut load by 40%.</p></body></html>"
)


def _http() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=PAGE, headers={"content-type": "text/html"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _call(name: str, args: dict[str, object], call_id: str) -> dict[str, object]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


FINDINGS = ResearchFindings(
    sources=[
        SourceNote(url=SOURCE, title="Cache guide", facts=["Per-view caching cut load by 40%."])
    ],
    related_posts=[
        RelatedPost(id=3, title="Django caching basics", link="https://wp.test/caching")
    ],
)


async def test_gather_context_runs_tools_and_merges_terms() -> None:
    writer = fake_client(
        ai(
            tool_calls=[
                _call("fetch_url", {"url": SOURCE}, "1"),
                _call("list_posts", {"search": "django"}, "2"),
            ]
        ),
        ai("done"),
        FINDINGS,
    )
    gateway = FakeGateway()

    update = await gather_context(
        {"brief": BRIEF, "outline": OUTLINE},
        runtime(writer=writer, wordpress=gateway, http=_http()),
    )

    notes = update["context"]
    assert notes.sources[0].facts == ["Per-view caching cut load by 40%."]
    assert notes.related_posts[0].id == 3
    assert [c.name for c in notes.categories] == ["Backend"]
    assert [t.name for t in notes.tags] == ["django"]
    assert update["status"] == RunStatus.WRITING
    assert ("list_posts", {"search": "django"}) in gateway.calls
    assert [u.node for u in update["usage"]] == ["gather_context", "gather_context"]


async def test_gather_context_feeds_fetched_text_back_to_the_model() -> None:
    writer = fake_client(
        ai(tool_calls=[_call("fetch_url", {"url": SOURCE}, "1")]), ai("done"), FINDINGS
    )

    await gather_context(
        {"brief": BRIEF, "outline": OUTLINE},
        runtime(writer=writer, wordpress=FakeGateway(), http=_http()),
    )

    assert isinstance(writer.chat_model, FakeChatModel)
    tool_messages = [m for m in writer.chat_model.calls[1] if isinstance(m, ToolMessage)]
    assert "cut load by 40%" in str(tool_messages[0].content)


async def test_gather_context_drops_entries_not_seen_in_tool_output() -> None:
    invented = ResearchFindings(
        sources=[SourceNote(url="https://elsewhere.test", title="X", facts=["90% of devs agree"])],
        related_posts=[RelatedPost(id=99, title="Ghost", link="https://wp.test/ghost")],
    )
    writer = fake_client(ai("done"), invented)

    update = await gather_context(
        {"brief": BRIEF, "outline": OUTLINE}, runtime(writer=writer, wordpress=FakeGateway())
    )

    assert update["context"].sources == []
    assert update["context"].related_posts == []


async def test_gather_context_stops_after_max_tool_iterations() -> None:
    loop = [ai(tool_calls=[_call("list_posts", {"search": "x"}, str(i))]) for i in range(2)]
    writer = fake_client(*loop, ResearchFindings())
    gateway = FakeGateway()

    await gather_context(
        {"brief": BRIEF, "outline": OUTLINE},
        runtime(writer=writer, wordpress=gateway, max_tool_iterations=2),
    )

    assert sum(1 for name, _ in gateway.calls if name == "list_posts") == 2


async def test_gather_context_without_sources_binds_no_fetch_tool() -> None:
    brief = BRIEF.model_copy(update={"source_urls": []})
    writer = fake_client(
        ai(tool_calls=[_call("fetch_url", {"url": SOURCE}, "1")]), ai("done"), ResearchFindings()
    )

    await gather_context(
        {"brief": brief, "outline": OUTLINE}, runtime(writer=writer, wordpress=FakeGateway())
    )

    assert isinstance(writer.chat_model, FakeChatModel)
    tool_message = next(m for m in writer.chat_model.calls[1] if isinstance(m, ToolMessage))
    assert "unknown tool fetch_url" in str(tool_message.content)


async def test_gather_context_propagates_llm_errors() -> None:
    writer = fake_client(ai("done"), ai("not json"))

    with pytest.raises(LLMError):
        await gather_context(
            {"brief": BRIEF, "outline": OUTLINE}, runtime(writer=writer, wordpress=FakeGateway())
        )
