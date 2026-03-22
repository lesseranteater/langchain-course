from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_ollama import ChatOllama as LangChainChatOllama
from langchain_openai import ChatOpenAI as LangChainChatOpenAI


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return str(content)


class ChatModel(ABC):
    @abstractmethod
    def chat(self, prompt: str) -> str:
        raise NotImplementedError


class ChatOpenAI(ChatModel):
    def __init__(self, model: str, temperature: float = 0, **kwargs: Any) -> None:
        self._client = LangChainChatOpenAI(
            model=model,
            temperature=temperature,
            **kwargs,
        )

    def chat(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        return _content_to_text(response.content)


class ChatOllama(ChatModel):
    def __init__(
        self,
        model: str,
        temperature: float = 0,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._client = LangChainChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
            **kwargs,
        )

    def chat(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        return _content_to_text(response.content)
