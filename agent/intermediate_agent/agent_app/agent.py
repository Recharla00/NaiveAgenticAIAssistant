"""Let LangChain construct and run the agent instead of a handwritten loop."""

import logging
from time import perf_counter

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from .config import Settings
from .approvals import ApprovalHandler, require_data_change_approval
from .execution_time import (DEFAULT_MAX_SECONDS, RunDeadlineExceeded,
                             check_deadline, enforce_model_deadline,
                             enforce_tool_deadline, question_time_limit)
from .flow_logging import logger
from .memory import MessageRecord, SessionStore
from .model import build_model
from .prompts import (BILLING_SYSTEM_PROMPT, DEMO_WRITE_INSTRUCTION,
                      GENERAL_SYSTEM_PROMPT,
                      RETRIEVAL_SYSTEM_PROMPT, SYSTEM_PROMPT,
                      TECHNICAL_SYSTEM_PROMPT, TOOL_RECOVERY_INSTRUCTION,
                      TOOL_SYSTEM_PROMPT)
from .route_paths import prepare_route_messages
from .routing import Route, route_question
from .tools import get_tools
from .tool_errors import limit_tool_recovery, return_tool_errors


def build_agent(settings: Settings, *, enable_tools=False, enable_retrieval=False,
                enable_demo_write=False, approval_handler: ApprovalHandler | None = None,
                max_model_calls: int = 6, max_tool_calls: int = 8):
    tools = (get_tools(include_search=enable_retrieval, include_demo_write=enable_demo_write)
             if enable_tools or enable_retrieval or enable_demo_write else [])
    middleware = _execution_middleware(bool(tools), max_model_calls, max_tool_calls,
                                       approval_handler=approval_handler)
    return create_agent(
        model=build_model(settings),
        tools=tools,
        system_prompt=(RETRIEVAL_SYSTEM_PROMPT if enable_retrieval else
                       (TOOL_SYSTEM_PROMPT if enable_tools or enable_demo_write else SYSTEM_PROMPT))
                      + (TOOL_RECOVERY_INSTRUCTION if tools else "")
                      + (DEMO_WRITE_INSTRUCTION if enable_demo_write else ""),
        middleware=middleware,
    )


def build_routed_agent(settings: Settings, route: Route, *, max_model_calls: int = 6,
                       max_tool_calls: int = 8, _tool_calls_used: int = 0,
                       approval_handler: ApprovalHandler | None = None):
    """Build one agent with the instructions and tools for a chosen path."""
    if route == "billing":
        prompt = BILLING_SYSTEM_PROMPT
        tools = get_tools(include_search=True)
    elif route == "technical":
        prompt = TECHNICAL_SYSTEM_PROMPT
        tools = get_tools(include_search=True, include_calculator=False)
    elif route == "general":
        prompt = GENERAL_SYSTEM_PROMPT
        tools = get_tools()
    else:
        raise ValueError(f"Unknown route: {route!r}")
    middleware = _execution_middleware(bool(tools), max_model_calls, max_tool_calls,
                                       tool_calls_used=_tool_calls_used,
                                       approval_handler=approval_handler)
    return create_agent(model=build_model(settings), tools=tools,
                        system_prompt=prompt + TOOL_RECOVERY_INSTRUCTION,
                        middleware=middleware)


def _execution_middleware(has_tools: bool, max_model_calls: int, max_tool_calls: int,
                          *, tool_calls_used: int = 0,
                          approval_handler: ApprovalHandler | None = None):
    _validate_call_limits(max_model_calls, max_tool_calls)
    if (isinstance(tool_calls_used, bool) or not isinstance(tool_calls_used, int) or
            not 0 <= tool_calls_used <= max_tool_calls):
        raise ValueError("tool_calls_used must fit within max_tool_calls")
    logger.info("Execution limits: model_calls=%d tool_calls=%d already_used=%d",
                max_model_calls, max_tool_calls, tool_calls_used)
    middleware = [ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="error"),
                  enforce_model_deadline]
    if has_tools:
        middleware.extend([
            ToolCallLimitMiddleware(run_limit=max_tool_calls - tool_calls_used,
                                    exit_behavior="error"),
            enforce_tool_deadline,
            require_data_change_approval(approval_handler),
            return_tool_errors,
            limit_tool_recovery,
        ])
    return middleware


def _validate_call_limits(max_model_calls: int, max_tool_calls: int) -> None:
    for name, value in (("max_model_calls", max_model_calls), ("max_tool_calls", max_tool_calls)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def run_session_agent(agent, question: str, session_id: str, *, store: SessionStore | None = None,
                      max_steps: int = 50, max_seconds: float = DEFAULT_MAX_SECONDS):
    """Run one turn with restored history, then persist the complete result.

    ``agent`` is a graph returned by ``build_agent``. A failed or incomplete
    turn leaves the previously saved session unchanged.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    _validate_max_steps(max_steps)
    store = store if store is not None else SessionStore()
    session = store.load(session_id)
    history = session.langchain_messages()
    logger.info("Session loaded: id=%r previous_messages=%d", session_id, len(history))
    _log_history(history)
    logger.debug("Current question: %s", _preview(question))
    logger.info("Agent started: mode=manual max_steps=%d", max_steps)
    started = perf_counter()
    with question_time_limit(max_seconds):
        try:
            state = agent.invoke(
                {"messages": [*history, HumanMessage(content=question)]},
                config={"recursion_limit": max_steps},
            )
            check_deadline()
            _log_turn(state, len(history), perf_counter() - started)
            _save_completed_state(state, session, store)
            return state
        except Exception as error:
            logger.info("Agent failed after %.2fs: %s: %s", perf_counter() - started,
                        type(error).__name__, error)
            answer = _run_failure_answer(error)
            if answer is None:
                raise
            logger.info("Session unchanged after failed run: id=%r", session_id)
            return _failed_state(history, [HumanMessage(content=question)], answer)


def run_routed_session_agent(settings: Settings, question: str, session_id: str, *,
                             store: SessionStore | None = None, max_steps: int = 50,
                             max_model_calls: int = 6, max_tool_calls: int = 8,
                             max_seconds: float = DEFAULT_MAX_SECONDS,
                             approval_handler: ApprovalHandler | None = None):
    """Select a route for this question and run it with saved conversation history."""
    decision = route_question(question)
    _validate_max_steps(max_steps)
    _validate_call_limits(max_model_calls, max_tool_calls)
    store = store if store is not None else SessionStore()
    session = store.load(session_id)
    history = session.langchain_messages()
    logger.info("Session loaded: id=%r previous_messages=%d", session_id, len(history))
    logger.info("Route selected: %s (%s)", decision.route, decision.reason)
    _log_history(history)
    logger.debug("Current question: %s", _preview(question))
    started = perf_counter()
    with question_time_limit(max_seconds):
        try:
            prepared = prepare_route_messages(decision.route, question)
            check_deadline()
            if prepared.answer is not None:
                search_failed = any(isinstance(message, ToolMessage) and message.status == "error"
                                    for message in prepared.messages)
                logger.info("Model skipped: %s", "knowledge search failed" if search_failed
                            else "no relevant internal documentation")
                state = {"messages": [*history, *prepared.messages,
                                      AIMessage(content=prepared.answer,
                                                response_metadata={"run_failed": "knowledge_search_failed"}
                                                if search_failed else {})]}
            else:
                agent = build_routed_agent(settings, decision.route,
                                           max_model_calls=max_model_calls,
                                           max_tool_calls=max_tool_calls,
                                           _tool_calls_used=1 if decision.route == "technical" else 0,
                                           approval_handler=approval_handler)
                logger.info("Agent started: route=%s max_steps=%d", decision.route, max_steps)
                state = agent.invoke(
                    {"messages": [*history, *prepared.messages]},
                    config={"recursion_limit": max_steps},
                )
                check_deadline()
                _log_turn(state, len(history) + len(prepared.messages), perf_counter() - started)
            check_deadline()
            _save_completed_state(state, session, store)
            return decision, state
        except Exception as error:
            logger.info("Agent failed after %.2fs: %s: %s", perf_counter() - started,
                        type(error).__name__, error)
            answer = _run_failure_answer(error)
            if answer is None:
                raise
            logger.info("Session unchanged after failed run: id=%r", session_id)
            return decision, _failed_state(history, [HumanMessage(content=question)], answer)


def _validate_max_steps(max_steps: int) -> None:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")


def _run_failure_answer(error: Exception) -> str | None:
    if isinstance(error, RunDeadlineExceeded):
        return "The question reached its time limit. Please try again."
    if isinstance(error, ModelCallLimitExceededError):
        return "The agent reached its model-call limit. Please try a simpler question."
    if isinstance(error, ToolCallLimitExceededError):
        return "The agent reached its tool-call limit. Please try a simpler question."
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return "The model took too long to respond. Please try again."
    if isinstance(error, httpx.ConnectError):
        return "I could not connect to the local model. Check that Ollama is running and try again."
    if isinstance(error, GraphRecursionError):
        return "The agent reached its step limit before finishing. Please try a simpler question."
    return None


def _failed_state(history, turn_messages, answer: str):
    return {"messages": [*history, *turn_messages,
                         AIMessage(content=answer, response_metadata={"run_failed": "model_error"})]}


def _save_completed_state(state, session, store: SessionStore) -> None:
    messages = state.get("messages") if isinstance(state, dict) else None
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], AIMessage):
        raise ValueError("The agent ended without a final model response")
    response = messages[-1]
    if response.tool_calls or not response.text.strip() or response.response_metadata.get("done_reason") == "length":
        raise ValueError("The agent did not produce a complete final answer")
    if response.response_metadata.get("run_failed"):
        logger.info("Session unchanged after failed run: id=%r reason=%s", session.session_id,
                    response.response_metadata["run_failed"])
        return
    # Preserve the whole graph state: tool requests, tool results, and source
    # evidence are needed when this session is resumed in another process.
    session.replace_messages([MessageRecord.from_langchain(message) for message in messages])
    path = store.save(session)
    logger.info("Session saved: messages=%d path=%s", len(session.messages), path)


def _log_history(history) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        for index, message in enumerate(history, 1):
            if isinstance(message, AIMessage) and message.tool_calls:
                calls = ", ".join(f"{call['name']}[{call['id']}]" for call in message.tool_calls)
                logger.debug("History %d: assistant requested %s", index, calls)
            elif isinstance(message, ToolMessage):
                logger.debug("History %d: tool %s[%s] %s", index, message.name or "unknown",
                             message.tool_call_id, _preview(message.text))
            else:
                logger.debug("History %d: %s %s", index, message.type, _preview(message.text))


def _log_turn(state, start_index: int, elapsed: float) -> None:
    if not logger.isEnabledFor(logging.INFO):
        return
    messages = state.get("messages", []) if isinstance(state, dict) else []
    current = messages[start_index:]
    model_messages = [message for message in current if isinstance(message, AIMessage)
                      and message.additional_kwargs.get("origin") != "routing"]
    tool_results = [message for message in current if isinstance(message, ToolMessage)]
    for message in current:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                logger.info("Model requested tool: %s call_id=%s", call["name"], call["id"])
                logger.debug("Tool arguments: %s", _preview(str(call["args"])))
            if message.text:
                logger.debug("Assistant message: %s", _preview(message.text))
        elif isinstance(message, ToolMessage):
            logger.info("Tool returned: %s call_id=%s status=%s",
                        message.name or "unknown", message.tool_call_id, message.status)
            logger.debug("Tool result preview: %s", _preview(message.text))
    logger.info("Agent finished: duration=%.2fs model_messages=%d tool_results=%d",
                elapsed, len(model_messages), len(tool_results))


def _preview(value: str, limit: int = 180) -> str:
    clean = value.replace("\r", " ").replace("\n", " ")
    return repr(clean[:limit] + ("..." if len(clean) > limit else ""))
