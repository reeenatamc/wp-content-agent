import anthropic
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
    assert [u.node for u in update["usage"]] == ["gather_context"]
    assert update["usage"][0].input_tokens == 30


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


async def test_notes_call_gets_results_as_text_without_tool_blocks() -> None:
    writer = fake_client(
        ai(tool_calls=[_call("fetch_url", {"url": SOURCE}, "1")]), ai("done"), FINDINGS
    )

    await gather_context(
        {"brief": BRIEF, "outline": OUTLINE},
        runtime(writer=writer, wordpress=FakeGateway(), http=_http()),
    )

    assert isinstance(writer.chat_model, FakeChatModel)
    notes_call = writer.chat_model.calls[-1]
    assert not any(isinstance(m, ToolMessage) for m in notes_call)
    assert '<tool_result name="fetch_url"' in str(notes_call[-1].content)


async def test_source_that_failed_to_fetch_is_dropped() -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="gone")

    http = httpx.AsyncClient(transport=httpx.MockTransport(failing))
    writer = fake_client(
        ai(tool_calls=[_call("fetch_url", {"url": SOURCE}, "1")]), ai("done"), FINDINGS
    )

    update = await gather_context(
        {"brief": BRIEF, "outline": OUTLINE},
        runtime(writer=writer, wordpress=FakeGateway(), http=http),
    )

    assert update["context"].sources == []


async def test_facts_with_numbers_missing_from_the_page_are_dropped() -> None:
    facts = ["Per-view caching cut load by 40%.", "Redis is 75% faster.", "Caching helps."]
    findings = ResearchFindings(sources=[SourceNote(url=SOURCE, title="Guide", facts=facts)])
    writer = fake_client(
        ai(tool_calls=[_call("fetch_url", {"url": SOURCE}, "1")]), ai("done"), findings
    )

    update = await gather_context(
        {"brief": BRIEF, "outline": OUTLINE},
        runtime(writer=writer, wordpress=FakeGateway(), http=_http()),
    )

    assert update["context"].sources[0].facts == [facts[0], facts[2]]


async def test_related_posts_take_id_title_and_link_from_wordpress() -> None:
    wrong = ResearchFindings(
        related_posts=[RelatedPost(id=3, title="Made up", link="https://wp.test/caching-advanced")]
    )
    writer = fake_client(
        ai(tool_calls=[_call("list_posts", {"search": "django"}, "1")]), ai("done"), wrong
    )

    update = await gather_context(
        {"brief": BRIEF, "outline": OUTLINE}, runtime(writer=writer, wordpress=FakeGateway())
    )

    assert update["context"].related_posts == [
        RelatedPost(id=3, title="Django caching basics", link="https://wp.test/caching")
    ]


async def test_local_tool_error_is_returned_to_the_model_as_text() -> None:
    writer = fake_client(
        ai(tool_calls=[_call("fetch_url", {}, "1")]), ai("done"), ResearchFindings()
    )

    await gather_context(
        {"brief": BRIEF, "outline": OUTLINE},
        runtime(writer=writer, wordpress=FakeGateway(), http=_http()),
    )

    assert isinstance(writer.chat_model, FakeChatModel)
    tool_message = next(m for m in writer.chat_model.calls[1] if isinstance(m, ToolMessage))
    assert str(tool_message.content).startswith("error: fetch_url failed")


async def test_llm_error_on_notes_carries_loop_usage() -> None:
    writer = fake_client(
        ai(tool_calls=[_call("list_posts", {"search": "x"}, "1")]), ai("done"), ai("bad")
    )

    with pytest.raises(LLMError) as info:
        await gather_context(
            {"brief": BRIEF, "outline": OUTLINE}, runtime(writer=writer, wordpress=FakeGateway())
        )

    assert info.value.usage is not None
    assert info.value.usage.input_tokens == 30


async def test_llm_error_mid_loop_carries_usage_of_earlier_replies() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    writer = fake_client(
        ai(tool_calls=[_call("list_posts", {"search": "x"}, "1")]),
        anthropic.APIConnectionError(request=request),
    )

    with pytest.raises(LLMError, match="gather_context: model call failed") as info:
        await gather_context(
            {"brief": BRIEF, "outline": OUTLINE}, runtime(writer=writer, wordpress=FakeGateway())
        )

    assert info.value.usage is not None
    assert (info.value.usage.input_tokens, info.value.usage.output_tokens) == (10, 5)


async def test_llm_error_on_first_call_has_no_usage() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    writer = fake_client(anthropic.APIConnectionError(request=request))

    with pytest.raises(LLMError, match="gather_context: model call failed") as info:
        await gather_context(
            {"brief": BRIEF, "outline": OUTLINE}, runtime(writer=writer, wordpress=FakeGateway())
        )

    assert info.value.usage is None
