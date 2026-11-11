"""Step 5: repeat model decisions and tool calls until a final answer.

python agent/basic_agent/05_agent_loop.py --question "Add 20 and 5, then multiply the result by 8."
Each model call counts toward --max-steps (default 6), including the final answer.
History is kept in memory for this question only. RAG comes in a later lesson.
"""

import json
from agent_helpers import choose_action, execute_tool, lesson_args, run_lesson


def run_agent(question, model="qwen3:1.7b", max_steps=6, debug=False):
    if not question.strip() or max_steps < 1:
        raise ValueError("Provide a question and a positive step limit.")
    history = []
    for step in range(1, max_steps + 1):
        print(f"\nStep {step}/{max_steps}: asking the model...", flush=True)
        action = choose_action(question, history, model, debug)
        print("Action:", json.dumps(action), flush=True)

        # The model can finish immediately for a question that needs no tool.
        if action["action"] == "final":
            return action["answer"]

        observation = execute_tool(action)
        print("Tool result:", json.dumps(observation), flush=True)
        history.append({"action": action, "observation": observation})

    # Never present the last tool output as if it were a completed answer.
    raise ValueError("Step limit reached without a final answer. Try a simpler question or increase --max-steps.")


def main():
    args = lesson_args("Step 5: run the standalone agent.", "Add 20 and 5, then multiply the result by 8.", loop=True)
    answer = run_agent(args.question, args.model, args.max_steps, args.debug)
    print("\nFinal answer:", answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_lesson(main))
