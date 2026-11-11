"""Step 4: execute one tool and send its result back to the model.

python agent/basic_agent/04_execute_tool.py --question "What is 125 times 8?"
This lesson stops after one tool round trip. Step 5 handles repeated calls.
"""

import json
from agent_helpers import choose_action, execute_tool, lesson_args, run_lesson


def main():
    args = lesson_args("Step 4: execute one tool.", "What is 125 times 8?")
    history = []
    print("Choosing an action...", flush=True)
    action = choose_action(args.question, history, args.model, args.debug)
    print("Action:", json.dumps(action), flush=True)
    if action["action"] == "tool":
        observation = execute_tool(action)
        print("Tool result:", json.dumps(observation), flush=True)
        history.append({"action": action, "observation": observation})
        action = choose_action(args.question, history, args.model, args.debug)
        print("Next action:", json.dumps(action), flush=True)
    if action["action"] == "final":
        print("Answer:", action["answer"])
    else:
        print("The model requested another tool. Use step 5 for multiple calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_lesson(main))
