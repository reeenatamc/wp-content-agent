import uuid

from langgraph.checkpoint.memory import InMemorySaver

from tests.fakes import ai, fake_client
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
from wp_mcp.models import CreateDraftInput

META = (
    "Cache Django querysets where reads dominate, pick timeouts on purpose, and clear "
    "entries from signals so readers never see stale data on a busy page."
)


async def test_client_creates_a_draft_with_meta(
    wordpress: WordPressClient, cleanup: list[int]
) -> None:
    draft = CreateDraftInput(
        title="Integration draft",
        content_html="<!-- wp:paragraph --><p>Body</p><!-- /wp:paragraph -->",
        meta_description=META,
    )

    created = await wordpress.create_draft_post(draft)
    cleanup.append(created.id)
    post = await wordpress.get_post(created.id)

    assert created.status == "draft"
    assert created.meta_description_saved is True
    assert post.status == "draft"


async def test_create_tag_twice_returns_the_same_tag(wordpress: WordPressClient) -> None:
    name = f"it-{uuid.uuid4().hex[:8]}"

    first = await wordpress.create_tag(name)
    second = await wordpress.create_tag(name)
    await wordpress._http.delete(f"/tags/{first.id}", params={"force": "true"})

    assert first.id == second.id


async def test_seeded_content_is_listed(wordpress: WordPressClient) -> None:
    categories = await wordpress.list_categories()
    posts = await wordpress.list_posts(search="caching")

    assert "Backend" in {category.name for category in categories}
    assert "Django caching basics" in {post.title for post in posts}


async def test_mcp_gateway_lists_research_tools_and_terms() -> None:
    async with connect_wordpress(Settings()) as gateway:
        tools = {tool.name for tool in gateway.research_tools}
        tags = await gateway.list_tags()
        output = await gateway.run_research_tool("list_posts", {"search": "django"})

    assert tools == {"list_posts", "list_tags", "get_post"}
    assert "django" in {tag.name for tag in tags}
    assert "Django caching basics" in output


async def test_full_graph_creates_a_gutenberg_draft(
    wordpress: WordPressClient, cleanup: list[int]
) -> None:
    """The real graph, MCP server and WordPress, with scripted model replies."""
    seeded = (await wordpress.list_posts(search="Django caching basics"))[0]
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
    assert done.wp_post_id is not None
    cleanup.append(done.wp_post_id)
    post = await wordpress.get_post(done.wp_post_id)
    raw = await wordpress._http.get(f"/posts/{done.wp_post_id}", params={"context": "edit"})
    content = raw.json()["content"]["raw"]
    tags = await wordpress.list_tags(search=new_tag)
    for tag in tags:
        await wordpress._http.delete(f"/tags/{tag.id}", params={"force": "true"})

    assert pending.status == RunStatus.AWAITING_APPROVAL
    assert done.status == RunStatus.DRAFT_CREATED
    assert done.warnings == []
    assert post.status == "draft"
    assert "<!-- wp:heading -->" in content
    assert "<!-- wp:list -->" in content
    assert [tag.name for tag in tags] == [new_tag]
