"""Lesson 2: embed a question -> compare vectors -> retrieve relevant text.

Run: python rag/intermediate_rag/02_retrieve.py --compare
Try: python rag/intermediate_rag/02_retrieve.py --question "Can I get my money back?"

The default query is 'When will my parcel arrive?'. Compare semantic retrieval
across all registered chunking strategies, including a word-count baseline.
"""

import argparse
import sys

from semantic_helpers import CHUNKING_STRATEGIES, get_embedding_model, load_store, retrieve, word_count_matches


def show_matches(title, matches):
    print(f"\n{title}")
    if not matches:
        print("No matching words found.")
    for match in matches:
        print(f"\nChunk {match['chunk']['id']} | similarity {match['score']:.3f}")
        print(f"Source: {match['chunk'].get('source')} | PDF page {match['chunk'].get('page')}")
        print(match["chunk"]["text"])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Retrieve chunks using semantic embeddings.")
    parser.add_argument("--question", default="When will my parcel arrive?")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--embedding-model", help="Override RAG_EMBEDDING_MODEL; must match the rebuilt store.")
    parser.add_argument("--compare", action="store_true", help="Compare all registered chunking strategies, with semantic and word-count results.")
    parser.add_argument("--chunking", choices=list(CHUNKING_STRATEGIES), default="words")
    args = parser.parse_args()
    print(f"arguments: {args}")

    if not args.question.strip() or args.top_k < 1:
        parser.error("Use a non-empty question and --top-k of at least 1.")
    try:
        model = get_embedding_model(args.embedding_model)
        strategies = tuple(CHUNKING_STRATEGIES) if args.compare else (args.chunking,)
        # Validate all selected stores before making any comparison requests.
        stores = [(strategy, load_store(expected_model=model, strategy=strategy)) for strategy in strategies]
        if len({store["dimensions"] for _, store in stores}) != 1:
            raise ValueError("Store dimensions differ. Rebuild all stores with the same embedding model.")
        print("Question:", args.question)
        print("Embedding model:", model, flush=True)
        for strategy, store in stores:
            print(f"\n=== {strategy}: {len(store['chunks'])} indexed chunks ===")
            print("Chunking settings:", store.get("chunking"))
            matches = retrieve(args.question, store, args.top_k)
            show_matches(f"Semantic matches ({strategy} chunking):", matches)
            print("Retrieved context characters:", sum(len(item["chunk"]["text"]) for item in matches))
            if args.compare:
                show_matches(f"Word-count matches ({strategy} chunks):", word_count_matches(args.question, store, args.top_k))
        if args.compare:
            print("\nCompare answer evidence and pages, not just scores. Chunk IDs are local to each store.")
        print("\nSimilarity is not answer confidence. This ranks chunks even for unrelated questions.")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Could not retrieve: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
