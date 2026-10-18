from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from wp_content_agent.graph.context import AgentContext
from wp_content_agent.graph.state import AgentState
from wp_content_agent.nodes.render import render_brief
from wp_content_agent.schemas import Outline, RunStatus

SYSTEM_PROMPT = """You plan blog posts for a WordPress site.

Turn the brief into an outline a writer can follow. Pick one clear angle that serves the \
stated audience, then lay out sections in reading order. Use H2 for main sections and H3 \
only for subsections under an H2. Size the outline for the target length: roughly one H2 \
per 200 to 300 words. Each section's key points say what the section must cover, not \
how to phrase it. Work the keywords into headings only where they read naturally.

Do not invent facts or numbers at this stage; if a point needs evidence, say what kind \
of evidence ("example from the source on X") so research can look for it."""


async def plan(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(render_brief(state["brief"]))]
    outline, usage = await runtime.context.writer.structured(Outline, messages, node="plan")
    return {"outline": outline, "status": RunStatus.GATHERING, "usage": [usage]}
