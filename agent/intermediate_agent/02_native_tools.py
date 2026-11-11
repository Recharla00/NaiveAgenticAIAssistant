"""Step 2: let LangChain execute native model tool calls and return results.

Run from the project root:
agent/intermediate_agent/.venv/Scripts/python.exe agent/intermediate_agent/02_native_tools.py
"""

import argparse
import json
import math

from agent_app.config import Settings
from agent_app.execution_time import DEFAULT_MAX_SECONDS


def main():
    parser = argparse.ArgumentParser(description="Step 2: native calculator tools.", allow_abbrev=False)
    parser.add_argument("--question", default="What is 125 times 8?")
    parser.add_argument("--model", default=Settings.model)
    parser.add_argument("--max-steps", type=int, default=50,
                        help="LangGraph execution-step limit (includes model and tool steps).")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--debug", action="store_true", help="Print complete message state and traceback on failure.")
    args = parser.parse_args()
    if not args.question.strip() or not args.model.strip():
        parser.error("Question and model cannot be empty.")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive.")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--max-seconds must be a positive, finite number.")

    try:
        from agent_app.agent import build_agent
        from agent_app.tool_streaming import stream_tool_run
        from agent_app.tools import get_tools
        settings = Settings(model=args.model,
                            timeout_seconds=min(Settings.timeout_seconds, args.max_seconds))
        agent = build_agent(settings, enable_tools=True)
        print(f"Model: {settings.model} | Keep alive: {settings.keep_alive}")
        print("Available tools:", ", ".join(tool.name for tool in get_tools()))
        print(f"Question: {args.question}\n\nAgent activity:", flush=True)
        state = stream_tool_run(agent, args.question, args.max_steps,
                                max_seconds=args.max_seconds)
    except Exception as error:
        if args.debug:
            raise
        print(f"\nAgent stopped ({type(error).__name__}): {error}")
        print("Check Ollama/model availability. For a step-limit error, simplify the question or raise --max-steps.")
        return 1
    if args.debug:
        print(json.dumps([message.model_dump(mode="json") for message in state["messages"]], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
