"""Step 7: one agent loop with all tools available to the model.

python agent/basic_agent/07_run_agent.py --question "What is 125 times 8?"
python agent/basic_agent/07_run_agent.py --question "What is the refund policy?"
"""

import argparse
import json

from agent_helpers import choose_action, execute_tool, run_lesson


def run_agent(question, model="qwen3:1.7b", max_steps=6, debug=False):
    if not isinstance(question, str) or not question.strip() or max_steps < 1:
        raise ValueError("Provide a question and a positive step limit.")

    # All tools are available. The model chooses whether any tool is needed.
    history = []
    for step in range(1, max_steps + 1):
        print(f"\nStep {step}/{max_steps}: asking the model...", flush=True)
        action = choose_action(question, history, model, debug, rag=True)
        print("Action:", json.dumps(action), flush=True)
        if action["action"] == "final":
            return action["answer"]
        observation = execute_tool(action, rag=True)
        print("Tool result:", json.dumps(observation, indent=2), flush=True)
        history.append({"action": action, "observation": observation})
    raise ValueError("Step limit reached without a final answer. Try a simpler question or increase --max-steps.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Step 7: let the model choose arithmetic, policy search, or a direct answer.",
        allow_abbrev=False,
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum model calls, including the final answer.")
    parser.add_argument("--debug", action="store_true", help="Print full model requests and responses.")
    args = parser.parse_args(argv)
    if not args.question.strip():
        parser.error("The question cannot be empty.")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive.")
    print(f"Question: {args.question}", flush=True)
    answer = run_agent(args.question, args.model, args.max_steps, args.debug)
    print("\nFinal answer:", answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_lesson(main))
