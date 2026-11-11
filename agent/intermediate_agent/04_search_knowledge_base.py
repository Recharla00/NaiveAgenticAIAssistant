"""Step 4: search the active index with hybrid keyword/vector retrieval."""

import argparse
import json
from pathlib import Path

from agent_app.ingestion import DEFAULT_DATA
from agent_app.retrieval import HybridRetriever


def main():
    parser = argparse.ArgumentParser(description="Step 4: hybrid internal knowledge search.", allow_abbrev=False)
    parser.add_argument("query", nargs="?", default="What is the refund policy?")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    try:
        result = HybridRetriever(args.data_dir).search(args.query, args.top_k)
    except Exception as error:
        print(f"Search failed ({type(error).__name__}): {error}")
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
