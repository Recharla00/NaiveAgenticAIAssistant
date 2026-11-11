"""Offline ingestion checks; persistent Chroma runs in a child to release Windows file locks."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from agent_app.ingestion import prepare_documents

HERE = Path(__file__).resolve().parent


def persistence_worker(folder):
    from langchain_core.embeddings import Embeddings
    from langchain_chroma import Chroma
    from agent_app.ingestion import persist_index

    class FakeEmbeddings(Embeddings):
        calls = 0
        fail = False

        def embed_documents(self, texts):
            if self.fail:
                raise RuntimeError("Simulated embedding outage")
            self.calls += len(texts)
            return [[float(len(text)), 1.0, 0.5] for text in texts]

        def embed_query(self, text):
            return [float(len(text)), 1.0, 0.5]

    root = Path(folder)
    source = root / "docs"
    source.mkdir()
    (source / "one.md").write_text("Refund within thirty days.", encoding="utf-8")
    (source / "two.md").write_text("Delivery takes five days.", encoding="utf-8")
    data = root / "data"
    embeddings = FakeEmbeddings()
    chunks, corpus = prepare_documents(source)
    original, added = persist_index(chunks, corpus, embeddings, "fake", "digest-1", data)
    assert added == 2
    again, added = persist_index(chunks, corpus, embeddings, "fake", "digest-1", data)
    assert added == 0 and embeddings.calls == 2 and again == original
    reopened = Chroma(collection_name=original["collection"], persist_directory=str(data / "chroma"),
                      embedding_function=embeddings)
    assert len(reopened.get()["ids"]) == 2
    # Delete only this test's explicitly created source file.
    (source / "two.md").unlink()
    (source / "one.md").write_text("Refund within forty days.", encoding="utf-8")
    chunks, corpus = prepare_documents(source)
    embeddings.fail = True
    try:
        persist_index(chunks, corpus, embeddings, "fake", "digest-1", data)
        raise AssertionError("Expected outage")
    except RuntimeError:
        pass
    assert json.loads((data / "active_index.json").read_text()) == original
    embeddings.fail = False
    updated, added = persist_index(chunks, corpus, embeddings, "fake", "digest-1", data)
    assert updated["collection"] != original["collection"] and updated["count"] == 1
    replaced, _ = persist_index(chunks, corpus, embeddings, "fake", "digest-2", data)
    assert replaced["collection"] != updated["collection"]


class IngestionTests(unittest.TestCase):
    def test_markdown_chunks_and_stable_ids(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            path = Path(folder) / "policy.md"
            path.write_text("Refund within thirty days. " * 15, encoding="utf-8")
            chunks, corpus = prepare_documents(path, 100, 20)
            repeated, _ = prepare_documents(path, 100, 20)
            self.assertTrue(all(len(chunk.page_content) <= 100 for chunk in chunks))
            self.assertTrue(all(chunk.metadata["source"] == "policy.md" for chunk in chunks))
            self.assertEqual([c.metadata["chunk_id"] for c in chunks], [c.metadata["chunk_id"] for c in repeated])
            self.assertEqual(len(corpus["sources"]), 1)
            with self.assertRaises(ValueError):
                prepare_documents(path, 100, 100)

    def test_pdf_page_metadata(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            path = Path(folder) / "policy.pdf"
            path.write_bytes(b"fixture")
            reader = Mock(is_encrypted=False, pages=[Mock(), Mock()])
            reader.pages[0].extract_text.return_value = "Refund within thirty days."
            reader.pages[1].extract_text.return_value = "Delivery takes five days."
            with patch("agent_app.ingestion.PdfReader", return_value=reader):
                chunks, _ = prepare_documents(path)
            self.assertEqual([c.metadata["page"] for c in chunks], [1, 2])

    def test_no_text_pdf_and_empty_folder(self):
        from pypdf import PdfWriter
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            with self.assertRaises(ValueError):
                prepare_documents(folder)
            path = Path(folder) / "blank.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            writer.write(path)
            with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, "No extractable text"):
                prepare_documents(path)

    def test_persistent_snapshots(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", folder],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        persistence_worker(sys.argv[2])
    else:
        unittest.main()
