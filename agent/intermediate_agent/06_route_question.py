"""Preview Step 6.1 routing without calling the model or tools."""

import argparse

from agent_app.routing import route_question


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify a question for conditional routing.", allow_abbrev=False)
    parser.add_argument("question", help="Question to classify.")
    args = parser.parse_args(argv)
    try:
        decision = route_question(args.question)
    except ValueError as error:
        parser.error(str(error))
    print(f"Route: {decision.route}")
    print(f"Reason: {decision.reason}")
    if decision.billing_matches:
        print("Billing matches:", ", ".join(decision.billing_matches))
    if decision.technical_matches:
        print("Technical matches:", ", ".join(decision.technical_matches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
