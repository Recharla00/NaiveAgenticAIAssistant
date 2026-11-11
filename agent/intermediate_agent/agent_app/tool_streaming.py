"""Observe LangChain's native tool loop; execution stays inside the framework."""

import json
from time import perf_counter

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from .execution_time import DEFAULT_MAX_SECONDS, check_deadline, question_time_limit


def stream_tool_run(agent, question, max_steps=50, *, initial_messages=None,
                    max_seconds=DEFAULT_MAX_SECONDS):
    with question_time_limit(max_seconds):
        state = _stream_tool_run(agent, question, max_steps,
                                 initial_messages=initial_messages)
        check_deadline()
        return state


def _stream_tool_run(agent, question, max_steps, *, initial_messages):
    started = perf_counter()
    first_text = None
    state = None
    seen = 0
    messages = initial_messages if initial_messages is not None else [{"role": "user", "content": question}]
    for mode, event in agent.stream(
        {"messages": messages},
        config={"recursion_limit": max_steps},
        stream_mode=["messages", "values"],
    ):
        check_deadline()
        if mode == "messages":
            chunk, _metadata = event
            if isinstance(chunk, (AIMessage, AIMessageChunk)) and chunk.text:
                if first_text is None:
                    first_text = perf_counter() - started
                print(chunk.text, end="", flush=True)
        else:
            state = event
            # Values contain the accumulated conversation. Print each new event once.
            for message in state["messages"][seen:]:
                if isinstance(message, AIMessage):
                    for call in message.tool_calls:
                        origin = "Routing search" if message.additional_kwargs.get("origin") == "routing" else "Tool request"
                        print(f"\n{origin} [{call['id']}]: {call['name']} {json.dumps(call['args'])}", flush=True)
                elif isinstance(message, ToolMessage):
                    print(f"Tool result [{message.tool_call_id}]: {message.content}", flush=True)
            seen = len(state["messages"])
    print(flush=True)
    if not state or not isinstance(state["messages"][-1], AIMessage):
        raise ValueError("The agent ended without a final model response.")
    response = state["messages"][-1]
    if response.tool_calls or not response.text.strip() or response.response_metadata.get("done_reason") == "length":
        raise ValueError("The agent did not produce a complete final answer.")

    models = [m for m in state["messages"] if isinstance(m, AIMessage)]
    tools = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    print(f"\nAgent run: {perf_counter() - started:.2f}s | Model calls: {len(models)} | Tool results: {len(tools)}")
    if first_text is not None:
        print(f"First model text: {first_text:.2f}s (may be an intermediate response)")
    for key, label in [("load_duration", "Loading"), ("prompt_eval_duration", "Prompt processing"),
                       ("eval_duration", "Generation")]:
        durations = [m.response_metadata.get(key) for m in models]
        if all(isinstance(value, (int, float)) for value in durations):
            print(f"  {label}, all model calls: {sum(durations) / 1e9:.2f}s")
    return state
