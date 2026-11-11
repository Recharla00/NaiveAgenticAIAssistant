"""Prepare the messages required by a selected conditional path."""

import json
import re
from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .retrieval import search_internal_knowledge_base
from .routing import Route
from .flow_logging import logger
from .execution_time import RunDeadlineExceeded, check_deadline
from .tool_errors import tool_error_text

_COMMON_WORDS = frozenset({
    "a", "about", "an", "and", "are", "can", "do", "does", "for", "how", "i",
    "in", "is", "it", "me", "my", "of", "on", "or", "our", "please", "the",
    "to", "what", "where", "which", "with", "you",
})
NO_DOCUMENTATION_ANSWER = "I could not find relevant internal documentation for that question."


@dataclass(frozen=True)
class PreparedRoute:
    messages: list[BaseMessage]
    answer: str | None = None


def _meaningful_words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.casefold())
            if len(word) > 1 and word not in _COMMON_WORDS}


def prepare_route_messages(route: Route, question: str) -> PreparedRoute:
    """Return input messages; technical questions always search first.

    The tool request/result pair follows the normal chat protocol so the
    documentation evidence can also be retained in session memory later.
    """
    if route not in {"billing", "technical", "general"}:
        raise ValueError(f"Unknown route: {route!r}")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    messages = [HumanMessage(content=question)]
    if route == "technical":
        call_id = f"routing-search-{uuid4().hex}"
        logger.info("Knowledge search started for technical route")
        logger.debug("Knowledge search query: %r", question)
        started = perf_counter()
        try:
            check_deadline()
            evidence = search_internal_knowledge_base.invoke({"query": question})
            check_deadline()
            if (not isinstance(evidence, dict) or
                    not isinstance(evidence.get("results"), list) or
                    not all(isinstance(result, dict) for result in evidence["results"])):
                raise ValueError("Knowledge search returned an invalid result")
        except RunDeadlineExceeded:
            raise
        except Exception as error:
            logger.info("Knowledge search failed after %.2fs: %s", perf_counter() - started, error)
            messages.extend([
                AIMessage(content="", tool_calls=[{
                    "name": "search_internal_knowledge_base", "args": {"query": question}, "id": call_id,
                }], additional_kwargs={"origin": "routing"}),
                ToolMessage(content=tool_error_text("search_internal_knowledge_base", error),
                            tool_call_id=call_id, name="search_internal_knowledge_base",
                            status="error"),
            ])
            return PreparedRoute(messages, "I could not search the internal documentation right now. Please try again.")
        candidates = evidence.get("results", [])
        query_words = _meaningful_words(question)
        relevant = [result for result in candidates
                    if query_words & _meaningful_words(
                        f"{result.get('source', '')} {result.get('text', '')}")]
        sources = sorted({str(result.get("source", "unknown")) for result in relevant})
        logger.info("Knowledge search finished: candidates=%d relevant=%d sources=%s duration=%.2fs",
                    len(candidates), len(relevant), ", ".join(sources) or "none", perf_counter() - started)
        evidence = {**evidence, "results": relevant, "count": len(relevant)}
        messages.extend([
            AIMessage(content="", tool_calls=[{
                "name": "search_internal_knowledge_base", "args": {"query": question}, "id": call_id,
            }], additional_kwargs={"origin": "routing"}),
            ToolMessage(content=json.dumps(evidence, ensure_ascii=False),
                        tool_call_id=call_id, name="search_internal_knowledge_base"),
        ])
        if not relevant:
            return PreparedRoute(messages, NO_DOCUMENTATION_ANSWER)
    return PreparedRoute(messages)
