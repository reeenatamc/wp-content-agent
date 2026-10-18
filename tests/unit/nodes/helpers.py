from typing import Any

from langgraph.runtime import Runtime

from tests.fakes import fake_client
from wp_content_agent.graph import AgentContext
from wp_content_agent.llm import ModelClient
from wp_content_agent.schemas import Brief
from wp_content_agent.settings import Settings

BRIEF = Brief(
    topic="Caching Django querysets",
    audience="backend developers",
    tone="practical",
    target_words=600,
    keywords=["django", "caching"],
    source_urls=["https://example.com/django-cache"],  # type: ignore[list-item]
)


def runtime(
    *,
    writer: ModelClient | None = None,
    reviewer: ModelClient | None = None,
    wordpress: Any = None,
    **settings: Any,
) -> Runtime[AgentContext]:
    context = AgentContext(
        settings=Settings(_env_file=None, **settings),  # type: ignore[call-arg]
        writer=writer or fake_client(),
        reviewer=reviewer or fake_client(),
        wordpress=wordpress or object(),
    )
    return Runtime(context=context)
