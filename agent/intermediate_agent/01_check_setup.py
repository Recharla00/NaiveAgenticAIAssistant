"""Step 1: verify LangChain -> ChatOllama -> local model -> response.

Run from the project root using this lesson's virtual environment:
agent/intermediate_agent/.venv/Scripts/python.exe agent/intermediate_agent/01_check_setup.py
"""

import argparse
from importlib.metadata import version
import json
import sys
from time import perf_counter

from agent_app.config import Settings


def main():
    started = perf_counter()
    parser = argparse.ArgumentParser(description="Step 1: verify LangChain and local Ollama.", allow_abbrev=False)
    parser.add_argument("--question", default="Why indians are great? Explain in 5 bullet points.")
    parser.add_argument("--model", default=Settings.model)
    parser.add_argument("--debug", action="store_true", help="Print the returned messages and metadata.")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("The question cannot be empty.")
    if not args.model.strip():
        parser.error("The model name cannot be empty.")

    try:
        from agent_app.agent import build_agent
        from agent_app.streaming import print_timings, stream_answer
    except ModuleNotFoundError as error:
        print(f"Missing dependency: {error.name}")
        print("Install requirements.txt using agent/intermediate_agent/.venv/Scripts/python.exe -m pip.")
        return 1

    settings = Settings(model=args.model)
    print(f"Python: {sys.version.split()[0]}")
    print(f"LangChain: {version('langchain')} | langchain-ollama: {version('langchain-ollama')}")
    print(f"Model: {settings.model} | Server: {settings.base_url} | Keep alive: {settings.keep_alive}")
    print(f"Question: {args.question}\nWaiting for Ollama...", flush=True)
    try:
        agent = build_agent(settings)
        print("\nAnswer:", flush=True)
        result, first_text_seconds, elapsed = stream_answer(agent, args.question)
        response = result["messages"][-1]
    except Exception as error:
        # A CLI boundary gives beginners a readable failure; --debug keeps the traceback.
        if args.debug:
            raise
        print(f"\nSetup check failed ({type(error).__name__}): {error}")
        print("Check that Ollama is running and 'ollama list' contains the selected model.")
        return 1

    print_timings(response, first_text_seconds, elapsed)
    print(f"  Total lesson time (including imports/setup): {perf_counter() - started:.2f}s")
    if response.response_metadata.get("done_reason") == "length":
        print("[Output limit reached; the answer may be incomplete.]")
    if args.debug:
        print("\nReturned messages:")
        print(json.dumps([message.model_dump(mode="json") for message in result["messages"]], indent=2))
    print("\nStep 1 passed: LangChain received a response from local Ollama.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
