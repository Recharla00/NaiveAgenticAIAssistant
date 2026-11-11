"""Step 3: let the model choose an action, without executing it.

python agent/basic_agent/03_choose_action.py --question "What is 125 times 8?"
Add --debug to inspect the full request and response.
"""

import json
from agent_helpers import choose_action, lesson_args, run_lesson


def main():
    args = lesson_args("Step 3: choose an action.", "What is 125 times 8?")
    print("Asking the model to choose an action...", flush=True)
    action = choose_action(args.question, [], args.model, args.debug)
    print("action:", json.dumps(action, indent=2))
    print("This lesson only displays the choice; it does not execute a tool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_lesson(main))
