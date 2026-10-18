from typing import Any

import httpx
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
    http: httpx.AsyncClient | None = None,
    **settings: Any,
) -> Runtime[AgentContext]:
    context = AgentContext(
        settings=Settings(_env_file=None, **settings),  # type: ignore[call-arg]
        writer=fake_client() if writer is None else writer,
        reviewer=fake_client() if reviewer is None else reviewer,
        wordpress=object() if wordpress is None else wordpress,
        http=http,
    )
    return Runtime(context=context)
