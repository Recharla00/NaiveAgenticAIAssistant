"""Step 6.1: explainable, local question classification for one agent."""

from dataclasses import dataclass
import re
from typing import Literal


Route = Literal["billing", "technical", "general"]

BILLING_TERMS = (
    "bill", "billing", "invoice", "invoices", "charge", "charged", "charges",
    "payment", "payments", "refund", "refunds", "receipt", "receipts",
    "subscription", "subscriptions", "price", "pricing",
)
TECHNICAL_TERMS = (
    "api", "bug", "code", "configure", "configuration", "crash", "debug",
    "docs", "documentation", "endpoint", "error", "exception", "install",
    "installation", "integration", "server", "setup", "traceback",
)


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    reason: str
    billing_matches: tuple[str, ...] = ()
    technical_matches: tuple[str, ...] = ()


def _matches(question: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(term for term in terms
                 if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", question, re.IGNORECASE))


def route_question(question: str) -> RouteDecision:
    """Choose billing, technical, or general from explicit topic words.

    Mixed topics use the general route until a later step adds more context.
    This router does not call Ollama or run any tools.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")

    billing = _matches(question, BILLING_TERMS)
    technical = _matches(question, TECHNICAL_TERMS)
    if billing and technical:
        return RouteDecision("general", "Both billing and technical terms matched; topic is unclear.",
                             billing, technical)
    if billing:
        return RouteDecision("billing", "Billing terms matched.", billing, technical)
    if technical:
        return RouteDecision("technical", "Technical terms matched.", billing, technical)
    return RouteDecision("general", "No billing or technical terms matched.")
