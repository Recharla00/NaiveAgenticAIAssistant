"""Step 1: send a question to local Ollama and print the answer.

Run from the project root:
    python agent/basic_agent/01_call_llm.py
    python agent/basic_agent/01_call_llm.py --question "What is an AI agent?"

Uses Python's standard library. Ollama must be running with the selected model.
This first lesson is one LLM call; tools and the agent loop come later.
"""

import argparse
import json
import urllib.error
import urllib.request


def call_llm(question, model):
    # 1. Describe the request using a Python dictionary.
    payload = {
        "model": model,
        "prompt": question,
        "system": "Answer clearly and briefly, using beginner-friendly language.",
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 512},
    }


    # 2. Encode the dictionary as JSON and send it to Ollama on this computer.
    print("Request:", json.dumps(payload, indent=2))
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    print("Response:", json.dumps(result, indent=2))

    # 3. Read the answer from the response dictionary.
    answer = result.get("response", "").strip()
    if not answer:
        raise ValueError("Ollama returned an empty answer.")
    if result.get("done_reason") == "length":
        answer += "\n[Output limit reached; the answer may be incomplete.]"
    return answer


def main():
    parser = argparse.ArgumentParser(description="Step 1: ask a local LLM a question.")
    parser.add_argument("--question", default="What is an AI agent? Explain in two points.")
    parser.add_argument("--model", default="qwen3:1.7b")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("The question cannot be empty.")

    print(f"Question: {args.question}")
    print(f"Model: {args.model}\nWaiting for Ollama...", flush=True)
    try:
        answer = call_llm(args.question, args.model)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        print(f"Could not generate an answer: {error}")
        print("Check that Ollama is running and use 'ollama list' to check the model.")
        return 1
    print(f"\nAnswer:\n{answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
