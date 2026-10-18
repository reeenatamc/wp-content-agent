"""Test doubles shared across unit tests."""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, PrivateAttr

from wp_content_agent.llm import ModelClient
from wp_content_agent.settings import ModelPrice

PRICES = {"fake-model": ModelPrice(input=1.0, output=2.0)}


def ai(content: str = "", tokens: tuple[int, int] = (10, 5), **kwargs: Any) -> AIMessage:
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
            "total_tokens": sum(tokens),
        },
        **kwargs,
    )


class FakeChatModel(BaseChatModel):
    """Replays scripted replies. A reply is an AIMessage (chat) or a BaseModel (structured)."""

    _replies: list[AIMessage | BaseModel | Exception] = PrivateAttr(default_factory=list)
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @classmethod
    def scripted(cls, *replies: AIMessage | BaseModel | Exception) -> "FakeChatModel":
        model = cls()
        model._replies = list(replies)
        return model

    @property
    def calls(self) -> list[list[BaseMessage]]:
        return self._calls

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _next(self, messages: Sequence[BaseMessage]) -> AIMessage | BaseModel:
        self._calls.append(list(messages))
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        reply = self._next(messages)
        assert isinstance(reply, AIMessage), "expected a chat reply, got a structured one"
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, AIMessage]:
        return self  # type: ignore[return-value]

    def with_structured_output(  # type: ignore[override]
        self, schema: type[BaseModel], **kwargs: Any
    ) -> Runnable[Any, dict[str, Any]]:
        assert kwargs == {"method": "json_schema", "include_raw": True}, kwargs

        def run(messages: list[BaseMessage]) -> dict[str, Any]:
            reply = self._next(messages)
            if isinstance(reply, AIMessage):
                return {"raw": reply, "parsed": None, "parsing_error": ValueError("bad")}
            return {"raw": ai(), "parsed": reply, "parsing_error": None}

        return RunnableLambda(run)


def fake_client(*replies: AIMessage | BaseModel | Exception) -> ModelClient:
    return ModelClient(FakeChatModel.scripted(*replies), "fake-model", PRICES)
