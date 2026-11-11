"""Turn tool exceptions into readable results for the agent."""

from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from .flow_logging import logger


MAX_FAILED_TOOL_CALLS = 3  # Initial attempt plus at most two corrections.
RECOVERY_EXHAUSTED_ANSWER = (
    "I could not complete that request after several tool attempts. "
    "Please check the input or try again later."
)
APPROVAL_DENIED_ANSWER = "The unapproved change was not made."


@wrap_model_call
def limit_tool_recovery(request, handler):
    """Let the model correct a failed call, then stop repeated failures."""
    messages = request.state.get("messages", [])
    last_question = max((index for index, message in enumerate(messages)
                         if isinstance(message, HumanMessage)), default=-1)
    if any(isinstance(message, ToolMessage) and
           isinstance(message.artifact, dict) and message.artifact.get("approval_denied")
           for message in messages[last_question + 1:]):
        return AIMessage(content=APPROVAL_DENIED_ANSWER,
                         response_metadata={"run_failed": "approval_denied"})
    failures = sum(isinstance(message, ToolMessage) and message.status == "error"
                   for message in messages[last_question + 1:])
    if failures >= MAX_FAILED_TOOL_CALLS:
        logger.warning("Tool recovery limit reached: failures=%d limit=%d",
                       failures, MAX_FAILED_TOOL_CALLS)
        return AIMessage(content=RECOVERY_EXHAUSTED_ANSWER,
                         response_metadata={"run_failed": "tool_retries_exhausted"})
    if failures:
        logger.info("Tool recovery: failure=%d/%d; model may correct the request",
                    failures, MAX_FAILED_TOOL_CALLS)
    return handler(request)


def tool_error_text(name: str, error: Exception) -> str:
    """Give the model a useful next step without exposing a traceback."""
    if isinstance(error, ValidationError):
        return (f"Tool {name} received invalid arguments. Check the required "
                "argument names, types, and allowed values before trying again.")
    if isinstance(error, TimeoutError):
        return f"Tool {name} timed out. Try a simpler request or another approach."
    if isinstance(error, FileNotFoundError):
        return f"Tool {name} could not find its required data. {error}"
    return f"Tool {name} failed ({type(error).__name__}). Try another approach."


@wrap_tool_call
def return_tool_errors(request, handler):
    """Keep a failed tool call in the conversation for the model to inspect."""
    try:
        return handler(request)
    except Exception as error:
        name = request.tool_call["name"]
        logger.warning("Tool failed: %s call_id=%s error=%s: %s", name,
                       request.tool_call["id"], type(error).__name__, error)
        return ToolMessage(content=tool_error_text(name, error),
                           tool_call_id=request.tool_call["id"], name=name,
                           status="error")
