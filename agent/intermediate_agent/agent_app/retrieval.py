"""Hybrid keyword and vector retrieval over the active Step 3 snapshot."""

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any

from .ingestion import DEFAULT_DATA

_TOKEN = re.compile(r"[\w']+", re.UNICODE)
_RRF_K = 60


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN.findall(text) if len(token) > 1}


class HybridRetriever:
    """Read-only hybrid retriever for a published ``active_index.json``."""

    def __init__(self, data_dir=DEFAULT_DATA, *, vector_store=None):
        self.data_dir = Path(data_dir).resolve()
        manifest_path = self.data_dir / "active_index.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No active knowledge-base index found at {manifest_path}. "
                "Run 03_ingest_documents.py first."
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not self.manifest.get("collection") or not self.manifest.get("embedding_model"):
            raise ValueError("active_index.json is missing collection or embedding_model metadata")
        if vector_store is None:
            from langchain_chroma import Chroma
            from langchain_ollama import OllamaEmbeddings
            from .config import Settings

            settings = Settings()
            embeddings = OllamaEmbeddings(
                model=self.manifest["embedding_model"],
                base_url=settings.base_url,
                client_kwargs={"timeout": settings.timeout_seconds},
            )
            vector_store = Chroma(
                collection_name=self.manifest["collection"],
                persist_directory=self.manifest["persist_directory"],
                embedding_function=embeddings,
            )
        self.store = vector_store

    def _all_chunks(self) -> list[dict[str, Any]]:
        raw = self.store.get(include=["documents", "metadatas"])
        return [
            {"id": chunk_id, "text": text or "", "metadata": metadata or {}}
            for chunk_id, text, metadata in zip(
                raw.get("ids", []), raw.get("documents", []), raw.get("metadatas", [])
            )
        ]

    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be an integer from 1 to 20")

        vector_docs = self.store.similarity_search(query, k=min(max(top_k * 2, top_k), 20))
        vector_ids = [doc.metadata.get("chunk_id") for doc in vector_docs]
        chunks = self._all_chunks()
        query_tokens = _tokens(query)
        lexical_scores = []
        for chunk in chunks:
            terms = _tokens(chunk["text"])
            overlap = len(query_tokens & terms)
            if overlap:
                # Reward matching several query terms and exact phrase mentions.
                phrase_bonus = 1 if query.casefold() in chunk["text"].casefold() else 0
                lexical_scores.append((overlap + phrase_bonus, chunk))
        lexical_scores.sort(key=lambda item: (-item[0], item[1]["id"]))
        lexical_chunks = [chunk for _, chunk in lexical_scores[: min(top_k * 2, 20)]]

        by_id = {chunk["id"]: chunk for chunk in chunks}
        fused: dict[str, float] = defaultdict(float)
        methods: dict[str, set[str]] = defaultdict(set)
        for rank, chunk_id in enumerate(vector_ids, 1):
            if chunk_id in by_id:
                fused[chunk_id] += 1 / (_RRF_K + rank)
                methods[chunk_id].add("vector")
        for rank, chunk in enumerate(lexical_chunks, 1):
            fused[chunk["id"]] += 1 / (_RRF_K + rank)
            methods[chunk["id"]].add("keyword")

        ranked = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))[:top_k]
        results = []
        for rank, chunk_id in enumerate(ranked, 1):
            chunk = by_id[chunk_id]
            metadata = chunk["metadata"]
            evidence = {
                "rank": rank,
                "score": round(fused[chunk_id], 8),
                "retrieval": "+".join(sorted(methods[chunk_id])),
                "text": chunk["text"],
                "source": metadata.get("source", "unknown"),
                "chunk_id": metadata.get("chunk_id", chunk_id),
            }
            if "page" in metadata:
                evidence["page"] = metadata["page"]
            if "start_index" in metadata:
                evidence["start_index"] = metadata["start_index"]
            results.append(evidence)
        return {"query": query, "results": results, "count": len(results)}


def search_internal_knowledge_base(query: str, top_k: int = 5) -> dict[str, Any]:
    """Search internal PDF/Markdown knowledge and return cited evidence chunks."""
    return HybridRetriever().search(query, top_k)


try:
    from langchain.tools import tool
    search_internal_knowledge_base = tool(search_internal_knowledge_base)
    search_internal_knowledge_base.metadata = {"data_access": "read_only"}
except ImportError:  # pragma: no cover - dependencies are required by the lesson
    pass
