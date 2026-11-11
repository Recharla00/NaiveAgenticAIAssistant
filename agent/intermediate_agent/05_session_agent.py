"""Step 5: ask the local agent questions in a persistent session."""

import argparse
import math
from pathlib import Path

from agent_app.config import Settings
from agent_app.ingestion import DEFAULT_DATA
from agent_app.execution_time import DEFAULT_MAX_SECONDS


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one persistent agent turn.", allow_abbrev=False)
    parser.add_argument("--session-id", required=True, help="Reuse this ID for follow-up questions.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", default=Settings.model)
    parser.add_argument("--session-dir", type=Path, default=DEFAULT_DATA / "sessions")
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-age-days", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-model-calls", type=int, default=6)
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--enable-tools", action="store_true", help="Enable calculator tools.")
    parser.add_argument("--enable-retrieval", action="store_true", help="Enable knowledge search and calculator tools.")
    parser.add_argument("--enable-demo-write", action="store_true",
                        help="Enable the local save_demo_note tool, which asks for approval before writing.")
    parser.add_argument("--auto-route", action="store_true", help="Choose billing, technical, or general for each question.")
    parser.add_argument("--verbose", action="store_true", help="Show routing, search, model, tool, and session flow.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    if not args.session_id.strip() or not args.question.strip() or not args.model.strip():
        parser.error("Session ID, question, and model cannot be empty.")
    if (args.max_turns < 1 or args.max_age_days < 1 or args.max_steps < 1 or
            args.max_model_calls < 1 or args.max_tool_calls < 1):
        parser.error("Maximum turns, age in days, steps, and call limits must be positive.")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--max-seconds must be a positive, finite number.")
    if args.auto_route and (args.enable_tools or args.enable_retrieval or args.enable_demo_write):
        parser.error("--auto-route chooses tools for each path; do not combine it with manual tool flags.")

    try:
        from agent_app.agent import build_agent, run_routed_session_agent, run_session_agent
        from agent_app.approvals import interactive_approval
        from agent_app.flow_logging import configure_flow_logging
        from agent_app.memory import SessionStore

        configure_flow_logging(verbose=args.verbose, debug=args.debug)
        store = SessionStore(args.session_dir, max_turns=args.max_turns, max_age_days=args.max_age_days)
        settings = Settings(model=args.model,
                            timeout_seconds=min(Settings.timeout_seconds, args.max_seconds))
        if args.auto_route:
            decision, state = run_routed_session_agent(settings, args.question,
                                                       args.session_id, store=store, max_steps=args.max_steps,
                                                       max_model_calls=args.max_model_calls,
                                                       max_tool_calls=args.max_tool_calls,
                                                       max_seconds=args.max_seconds,
                                                       approval_handler=interactive_approval)
        else:
            agent = build_agent(settings, enable_tools=args.enable_tools,
                                enable_retrieval=args.enable_retrieval,
                                enable_demo_write=args.enable_demo_write,
                                approval_handler=interactive_approval,
                                max_model_calls=args.max_model_calls,
                                max_tool_calls=args.max_tool_calls)
            state = run_session_agent(agent, args.question, args.session_id, store=store,
                                      max_steps=args.max_steps, max_seconds=args.max_seconds)
    except Exception as error:
        if args.debug:
            raise
        print(f"Agent stopped ({type(error).__name__}): {error}")
        return 1

    print(f"Session: {args.session_id}")
    if args.auto_route:
        print(f"Route: {decision.route} ({decision.reason})")
    print(state["messages"][-1].text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
