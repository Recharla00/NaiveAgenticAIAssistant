"""Step 3: ingest local PDFs/Markdown into a persistent vector database.

Run with the intermediate_agent virtual environment. --preview does not call
Ollama or write the database. Hybrid retrieval and agent integration are step 4.
"""

import argparse
from pathlib import Path
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description="Step 3: documents to persistent vectors.", allow_abbrev=False)
    from agent_app.ingestion import DEFAULT_DATA, DEFAULT_INPUT, DEFAULT_EMBEDDING_MODEL, prepare_documents, persist_index
    from agent_app.config import Settings
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="One PDF/Markdown file or a folder (recursive).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=600, help="Maximum characters per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Target overlap in characters.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or not args.embedding_model.strip():
        parser.error("Use a positive batch size and a nonempty embedding model.")
    started = perf_counter()
    try:
        chunks, corpus = prepare_documents(args.input, args.chunk_size, args.chunk_overlap)
        print(f"Sources: {len(corpus['sources'])} | Chunks: {len(chunks)}")
        print("First chunk metadata:", chunks[0].metadata)
        print("First chunk preview:\n" + chunks[0].page_content[:300], flush=True)
        print("Corpus metadata:", corpus, flush=True)
        if args.preview:
            print("Preview complete. No embeddings generated or database written.")
            return 0

        from ollama import Client
        from langchain_ollama import OllamaEmbeddings
        settings = Settings()
        # Include the actual model digest in the index identity, not just its mutable tag.
        available = Client(host=settings.base_url, timeout=settings.timeout_seconds).list().models
        tag = args.embedding_model if ":" in args.embedding_model else args.embedding_model + ":latest"
        installed = next((model for model in available if model.model == tag), None)
        if installed is None:
            raise ValueError(f"Embedding model not installed. Run: ollama pull {tag}")
        embeddings = OllamaEmbeddings(model=tag, base_url=settings.base_url,
                                      client_kwargs={"timeout": settings.timeout_seconds})
        manifest, added = persist_index(chunks, corpus, embeddings, tag, installed.digest,
                                       args.data_dir, args.batch_size)
        print(f"\nActive collection: {manifest['collection']}")
        print(f"Stored chunks: {manifest['count']} | New embeddings: {added}")
        print(f"Index metadata: {args.data_dir.resolve() / 'active_index.json'}")
        print(f"Completed in {perf_counter() - started:.2f}s. No chat model called.")
        return 0
    except Exception as error:
        if args.debug:
            raise
        print(f"Ingestion failed ({type(error).__name__}): {error}")
        print("Check document text, installed embedding model, dependencies, and Ollama availability.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
