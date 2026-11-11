"""Lesson 1: PDF -> page text -> chunks -> semantic embeddings -> JSON store.

Run: python rag/intermediate_rag/01_create_embeddings.py
Setup: ollama pull embeddinggemma:latest (Ollama must be running).

Unlike word counts, a semantic embedding is a list of learned numbers whose
positions do not correspond to individual words. Similar meanings can have
similar vectors even when the texts use different words.
"""

import argparse
import json
import sys

from semantic_helpers import CHUNKING_STRATEGIES, chunking_settings, DOCUMENT, store_path, get_embedding_model, embed_texts, load_pdf_chunks, source_hash


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Build a semantic embedding store.")
    parser.add_argument("--embedding-model", "--model", dest="model", help="Override RAG_EMBEDDING_MODEL; embedding model, not a chat model.")
    parser.add_argument("--batch-size", type=int, default=12, help="Number of chunks per embedding request.")
    parser.add_argument("--chunking", choices=[*CHUNKING_STRATEGIES, "all", "both"], default="words")
    parser.add_argument("--chunk-size", type=int, default=600, help="Recursive maximum characters per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Recursive target overlap in characters.")
    args = parser.parse_args()
    print(f"arguments: {args}")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.chunk_size < 1 or not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error("Use --chunk-size > 0 and 0 <= --chunk-overlap < --chunk-size.")
    try:
        args.model = get_embedding_model(args.model)
        strategies = tuple(CHUNKING_STRATEGIES) if args.chunking in ("all", "both") else (args.chunking,)
        for strategy in strategies:
            # 1. Extract PDF text using the selected strategy, retaining page numbers.
            original_hash = source_hash()
            chunks, page_count = load_pdf_chunks(strategy=strategy, chunk_size=args.chunk_size,
                                                 chunk_overlap=args.chunk_overlap)
            print(f"[{strategy}] Loaded {page_count} PDF pages and {len(chunks)} chunks. Embedding with {args.model}...", flush=True)
            print("First chunk preview:", chunks[0]["text"][:50], "...", flush=True)
            print("Last chunk preview:", chunks[-1]["text"][:50], "...", flush=True)

            # 2. Send the chunks to an embedding model, not the text-generation model.
            # Small batches avoid one large request for the whole 50-page document.
            vectors = []
            for start in range(0, len(chunks), args.batch_size):
                batch = chunks[start:start + args.batch_size]
                vectors.extend(embed_texts([chunk["text"] for chunk in batch], args.model))
                print(f"Embedded {len(vectors)}/{len(chunks)} chunks.", flush=True)
            if any(len(vector) != len(vectors[0]) for vector in vectors):
                raise ValueError("Embedding dimensions changed between batches. Rebuild with one model.")
            if source_hash() != original_hash:
                raise ValueError("The PDF changed during embedding. Rerun after finishing your edits.")
            print(f"First vector: {len(vectors[0])} numbers; first vector preview = {[round(n, 2) for n in vectors[0][:10]]}")

            # 3. Store text alongside its embedding and remember which model made it.
            store = {
                "embedding_model": args.model,
                "dimensions": len(vectors[0]),
                "source": DOCUMENT.name,
                "source_sha256": original_hash,
                "source_type": "pdf",
                "page_count": page_count,
                "chunking": chunking_settings(strategy, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap),
                "chunks": [{**chunk, "embedding": vector} for chunk, vector in zip(chunks, vectors)],
            }
            path = store_path(strategy)
            path.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Saved: {path}")
        print("Next: python rag/intermediate_rag/02_retrieve.py --compare" if args.chunking in ("all", "both") else
              f"Next: python rag/intermediate_rag/02_retrieve.py --chunking {args.chunking}")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Could not build embeddings: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
