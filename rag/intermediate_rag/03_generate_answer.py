"""Lesson 3: retrieve semantic matches -> build prompt -> generate an answer.

Run: python rag/intermediate_rag/03_generate_answer.py --question "Can I get my money back?"
Add --prompt-only to inspect the prompt without calling the generation model.
Semantic retrieval still requires the embedding model and a running Ollama server.
"""

import argparse
import sys

from semantic_helpers import CHUNKING_STRATEGIES, build_prompt, get_embedding_model, load_store, request_ollama, retrieve


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Answer with semantic retrieval and local Ollama.")
    parser.add_argument("--question", default="Can I get my money back?")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--embedding-model", help="Override RAG_EMBEDDING_MODEL; must match the rebuilt store.")
    parser.add_argument("--model", default="qwen3:1.7b", help="Text-generation model, not embedding model.")
    parser.add_argument("--prompt-only", action="store_true")
    parser.add_argument("--chunking", choices=list(CHUNKING_STRATEGIES), default="words")
    args = parser.parse_args()
    if not args.question.strip() or args.top_k < 1:
        parser.error("Use a non-empty question and --top-k of at least 1.")
    try:
        # Retrieval automatically uses the embedding model recorded in the store.
        store = load_store(expected_model=get_embedding_model(args.embedding_model), strategy=args.chunking)
        print("Chunking strategy:", args.chunking)
        print("Embedding model:", store["embedding_model"], flush=True)
        matches = retrieve(args.question, store, args.top_k)
        print(f"Retrieved {len(matches)} chunk(s) for question: {args.question}", flush=True)
        print("score | chunk-id | chunk-text preview")
        for match in matches:
            print(f"{match['score']:.3f} | {match['chunk']['id']} | {match['chunk']['text'][:100]}...")

        prompt = build_prompt(args.question, matches)
        print("--- Retrieved context and prompt ---\n" + prompt, flush=True)
        if args.prompt_only:
            print("\nStopped before answer generation.")
            return 0

        # The generation model reads text; it does not read the embedding vectors.
        print(f"\nGenerating with {args.model}...", flush=True)
        result = request_ollama("generate", {
            "model": args.model, "prompt": prompt, "stream": False, "think": False,
            "options": {"temperature": 0, "num_predict": 512},
        })
        answer = result.get("response", "").strip()
        if not answer:
            raise ValueError("The generation model returned no answer text.")
        print("\nAnswer:\n" + answer)
        if result.get("done_reason") == "length":
            print("\nOutput limit reached; this answer may be incomplete.")
            return 1
        print("\nCheck the answer and citations against the displayed reference text.")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Could not answer: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
