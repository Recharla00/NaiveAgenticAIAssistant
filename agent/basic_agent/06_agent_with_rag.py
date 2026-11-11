"""Step 6: let the agent search the existing policy documents.

python agent/basic_agent/06_agent_with_rag.py --question "What is the refund policy?"
Add --debug for full model requests/responses. Step 5 remains standalone.
"""

import json
from agent_helpers import choose_action, execute_tool, lesson_args, run_lesson


def run_agent_with_rag(question, model="qwen3:1.7b", max_steps=6, debug=False):
    if not question.strip() or max_steps < 1:
        raise ValueError("Provide a question and a positive step limit.")
    history = []
    for step in range(1, max_steps + 1):
        print(f"\nStep {step}/{max_steps}: asking the model...", flush=True)
        # The same model connection now describes the search tool as well.
        action = choose_action(question, history, model, debug, rag=True)
        print("Action:", json.dumps(action), flush=True)
        if action["action"] == "final":
            return action["answer"]

        # A search result is an observation, just like a calculator result.
        observation = execute_tool(action, rag=True)
        print("Tool result:", json.dumps(observation, indent=2), flush=True)
        history.append({"action": action, "observation": observation})
    raise ValueError("Step limit reached without a final answer.")


def main():
    args = lesson_args("Step 6: agent with policy search.", "What is the refund policy?", loop=True)
    answer = run_agent_with_rag(args.question, args.model, args.max_steps, args.debug)
    print("\nFinal answer:", answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_lesson(main))
