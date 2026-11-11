"""Offline tests for hybrid ranking and the registered search tool."""

import json
import tempfile
import unittest
from pathlib import Path

from agent_app.retrieval import HybridRetriever
from agent_app.tools import get_tools

HERE = Path(__file__).resolve().parent


class FakeStore:
    def __init__(self):
        self.chunks = [
            ("a", "Refunds are available within thirty days.", {"source": "policies.md", "chunk_id": "a"}),
            ("b", "Delivery normally takes five business days.", {"source": "shipping.md", "chunk_id": "b"}),
            ("c", "Refund requests require the original receipt.", {"source": "policies.md", "chunk_id": "c"}),
        ]

    def get(self, include):
        return {"ids": [x[0] for x in self.chunks], "documents": [x[1] for x in self.chunks],
                "metadatas": [x[2] for x in self.chunks]}

    def similarity_search(self, query, k):
        # Deliberately put the exact keyword result second to verify fusion.
        return [type("Doc", (), {"page_content": self.chunks[2][1], "metadata": self.chunks[2][2]})(),
                type("Doc", (), {"page_content": self.chunks[0][1], "metadata": self.chunks[0][2]})()][:k]


class RetrievalTests(unittest.TestCase):
    def test_hybrid_results_include_source_evidence(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            root = Path(folder)
            (root / "active_index.json").write_text(json.dumps({"collection": "x", "embedding_model": "fake"}), encoding="utf-8")
            result = HybridRetriever(root, vector_store=FakeStore()).search("refund", top_k=2)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["results"][0]["source"], "policies.md")
        self.assertTrue(all(item["text"] and item["chunk_id"] for item in result["results"]))

    def test_invalid_query_and_limit(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            root = Path(folder)
            (root / "active_index.json").write_text(json.dumps({"collection": "x", "embedding_model": "fake"}), encoding="utf-8")
            retriever = HybridRetriever(root, vector_store=FakeStore())
            with self.assertRaises(ValueError): retriever.search("", 5)
            with self.assertRaises(ValueError): retriever.search("refund", 21)

    def test_search_tool_is_opt_in(self):
        self.assertEqual({tool.name for tool in get_tools()}, {"add_numbers", "multiply_numbers"})
        self.assertIn("search_internal_knowledge_base", {tool.name for tool in get_tools(include_search=True)})


if __name__ == "__main__":
    unittest.main()
