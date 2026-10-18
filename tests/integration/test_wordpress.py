import uuid

from langgraph.checkpoint.memory import InMemorySaver

from tests.fakes import ai, fake_client
from tests.integration.conftest import Cleanup
from tests.unit.nodes.test_plan import OUTLINE
from wp_content_agent.graph import AgentContext, build_graph
from wp_content_agent.nodes import NODES
from wp_content_agent.nodes.gather_context import ResearchFindings
from wp_content_agent.nodes.review import MODEL_CHECKS
from wp_content_agent.runner import AgentRunner
from wp_content_agent.schemas import (
    ApprovalDecision,
    Brief,
    CheckResult,
    Draft,
    RelatedPost,
    ReviewVerdict,
    RunStatus,
)
from wp_content_agent.settings import Settings
from wp_content_agent.tools.mcp_gateway import connect_wordpress
from wp_mcp.client import WordPressClient
from wp_mcp.models import CreateDraftInput, PostSummary

META = (
    "Cache Django querysets where reads dominate, pick timeouts on purpose, and clear "
    "entries from signals so readers never see stale data on a busy page."
)


async def _seeded_post(wordpress: WordPressClient, title: str) -> PostSummary:
    """The post scripts/wp-up.sh seeds, matched by exact title among published posts."""
    posts = await wordpress.list_posts(search=title)
    matches = [post for post in posts if post.title == title and post.status == "publish"]
    assert matches, f"seeded post {title!r} is missing; run scripts/wp-up.sh"
    return matches[0]


async def test_client_creates_a_draft_with_meta(
    wordpress: WordPressClient, cleanup: Cleanup
) -> None:
    draft = CreateDraftInput(
        title="Integration draft",
        content_html="<!-- wp:paragraph --><p>Body</p><!-- /wp:paragraph -->",
        meta_description=META,
    )

    created = await wordpress.create_draft_post(draft)
    cleanup.posts.append(created.id)
    post = await wordpress.get_post(created.id)

    assert created.status == "draft"
    assert created.meta_description_saved is True
    assert post.status == "draft"


async def test_create_draft_twice_returns_the_same_post(
    wordpress: WordPressClient, cleanup: Cleanup
) -> None:
    draft = CreateDraftInput(
        title="Idempotent integration draft",
        slug=f"idempotent-{uuid.uuid4().hex[:8]}",
        content_html="<!-- wp:paragraph --><p>Body</p><!-- /wp:paragraph -->",
    )

    first = await wordpress.create_draft_post(draft)
    cleanup.posts.append(first.id)
    second = await wordpress.create_draft_post(draft)

    assert first.reused is False
    assert second.reused is True
    assert second.id == first.id


async def test_create_tag_twice_returns_the_same_tag(
    wordpress: WordPressClient, cleanup: Cleanup
) -> None:
    name = f"it-{uuid.uuid4().hex[:8]}"

    first = await wordpress.create_tag(name)
    cleanup.tags.append(first.id)
    second = await wordpress.create_tag(name)

    assert first.id == second.id


async def test_seeded_content_is_listed(wordpress: WordPressClient) -> None:
    categories = await wordpress.list_categories()
    posts = await wordpress.list_posts(search="caching")

    assert "Backend" in {category.name for category in categories.terms}
    assert "Django caching basics" in {post.title for post in posts}


async def test_mcp_gateway_lists_research_tools_and_terms() -> None:
    async with connect_wordpress(Settings()) as gateway:
        tools = {tool.name for tool in gateway.research_tools}
        tags = await gateway.list_tags()
        output = await gateway.run_research_tool("list_posts", {"search": "django"})

    assert tools == {"list_posts", "list_tags", "get_post"}
    assert "django" in {tag.name for tag in tags.terms}
    assert "Django caching basics" in output


async def test_full_graph_creates_a_gutenberg_draft(
    wordpress: WordPressClient, cleanup: Cleanup
) -> None:
    """The real graph, MCP server and WordPress, with scripted model replies."""
    seeded = await _seeded_post(wordpress, "Django caching basics")
    new_tag = f"querysets-{uuid.uuid4().hex[:6]}"
    draft = Draft(
        title="Cache Django querysets without stale data",
        slug=f"cache-querysets-{uuid.uuid4().hex[:6]}",
        excerpt="Where caching pays off and how to invalidate it.",
        content_html=(
            "<h2>When caching helps</h2><p>Read-heavy views gain the most.</p>"
            "<h2>Invalidation</h2><ul><li>Clear entries from signals.</li></ul>"
            f'<p>See <a href="{seeded.link}">Django caching basics</a>.</p>'
        ),
        category_ids=[],
        tag_names=["django", new_tag],
        meta_description=META,
    )
    writer = fake_client(
        OUTLINE,
        ai(tool_calls=[{"name": "list_posts", "args": {"search": "django"}, "id": "c1"}]),
        ai("done"),
        ResearchFindings(
            related_posts=[RelatedPost(id=seeded.id, title=seeded.title, link=seeded.link)]
        ),
        draft,
    )
    reviewer = fake_client(
        ReviewVerdict(checks=[CheckResult(name=n, passed=True) for n in MODEL_CHECKS], fixes=[])
    )
    settings = Settings()
    brief = Brief(topic="Caching Django querysets", audience="Backend developers")

    async with connect_wordpress(settings) as gateway:
        context = AgentContext(
            settings=settings, writer=writer, reviewer=reviewer, wordpress=gateway
        )
        runner = AgentRunner(build_graph(NODES, InMemorySaver()), context)
        pending = await runner.start(brief, run_id="it")
        done = await runner.resume("it", ApprovalDecision(approved=True))
        tags = await wordpress.list_tags(search=new_tag)
        cleanup.tags.extend(tag.id for tag in tags.terms)
        if done.wp_post_id is not None:
            cleanup.posts.append(done.wp_post_id)

    assert done.wp_post_id is not None, done.error
    post = await wordpress.get_post(done.wp_post_id)
    raw = await wordpress._http.get(f"/posts/{done.wp_post_id}", params={"context": "edit"})
    content = raw.json()["content"]["raw"]

    assert pending.status == RunStatus.AWAITING_APPROVAL
    assert done.status == RunStatus.DRAFT_CREATED
    assert done.warnings == []
    assert post.status == "draft"
    assert "<!-- wp:heading -->" in content
    assert "<!-- wp:list -->" in content
    assert [tag.name for tag in tags] == [new_tag]
