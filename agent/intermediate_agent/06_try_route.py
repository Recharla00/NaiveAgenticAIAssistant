"""Step 6.2: run one selected path without session routing yet."""

import argparse
import json
import math

from agent_app.config import Settings
from agent_app.execution_time import DEFAULT_MAX_SECONDS, check_deadline, question_time_limit


def main(argv=None):
    parser = argparse.ArgumentParser(description="Try a specific agent route.", allow_abbrev=False)
    parser.add_argument("--route", required=True, choices=("billing", "technical", "general"))
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", default=Settings.model)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-model-calls", type=int, default=6)
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--verbose", action="store_true", help="Show route and knowledge-search flow.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    if not args.question.strip() or not args.model.strip():
        parser.error("Question and model cannot be empty.")
    if args.max_steps < 1 or args.max_model_calls < 1 or args.max_tool_calls < 1:
        parser.error("Step and call limits must be positive.")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--max-seconds must be a positive, finite number.")

    try:
        from agent_app.agent import build_routed_agent
        from agent_app.approvals import interactive_approval
        from agent_app.flow_logging import configure_flow_logging
        from agent_app.route_paths import prepare_route_messages
        from agent_app.tool_streaming import stream_tool_run
        configure_flow_logging(verbose=args.verbose, debug=args.debug)
        print(f"Route: {args.route} | Question: {args.question}", flush=True)
        with question_time_limit(args.max_seconds):
            prepared = prepare_route_messages(args.route, args.question)
            check_deadline()
            if prepared.answer is not None:
                print(prepared.answer)
                return 0
            settings = Settings(model=args.model,
                                timeout_seconds=min(Settings.timeout_seconds, args.max_seconds))
            agent = build_routed_agent(settings, args.route,
                                       max_model_calls=args.max_model_calls,
                                       max_tool_calls=args.max_tool_calls,
                                       _tool_calls_used=1 if args.route == "technical" else 0,
                                       approval_handler=interactive_approval)
            state = stream_tool_run(agent, args.question, args.max_steps,
                                    initial_messages=prepared.messages, max_seconds=args.max_seconds)
    except Exception as error:
        if args.debug:
            raise
        print(f"Agent stopped ({type(error).__name__}): {error}")
        return 1
    if args.debug:
        print(json.dumps([message.model_dump(mode="json") for message in state["messages"]], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
