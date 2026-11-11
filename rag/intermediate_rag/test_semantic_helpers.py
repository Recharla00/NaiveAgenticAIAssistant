"""Offline behavioral checks. These do not measure the real model's quality."""

import importlib.util
import json
import sys
import unittest
from unittest.mock import MagicMock, patch

import semantic_helpers as helpers


class SemanticTests(unittest.TestCase):
    def test_compare_discovers_future_strategy_and_single_mode_stays_selected(self):
        spec = importlib.util.spec_from_file_location("retrieve_lesson", helpers.FOLDER / "02_retrieve.py")
        lesson = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lesson)
        future = {"split": lambda text, settings: [text], "settings": lambda options: {}}
        store = {"dimensions": 2, "chunks": [], "chunking": {}}
        with patch.dict(helpers.CHUNKING_STRATEGIES, {"future": future}), \
             patch.object(lesson, "load_store", return_value=store) as load, \
             patch.object(lesson, "retrieve", return_value=[]) as retrieve, \
             patch.object(lesson, "word_count_matches", return_value=[]) as baseline, \
             patch.object(sys, "stdout"):
            with patch.object(sys, "argv", ["02_retrieve.py", "--compare"]):
                self.assertEqual(lesson.main(), 0)
            self.assertEqual([call.kwargs["strategy"] for call in load.call_args_list], ["words", "recursive", "future"])
            self.assertEqual(retrieve.call_count, 3)
            self.assertEqual(baseline.call_count, 3)
            load.reset_mock(); retrieve.reset_mock(); baseline.reset_mock()
            with patch.object(sys, "argv", ["02_retrieve.py", "--chunking", "future"]):
                self.assertEqual(lesson.main(), 0)
            self.assertEqual(load.call_args.kwargs["strategy"], "future")
            self.assertEqual(retrieve.call_count, 1)
            baseline.assert_not_called()
            self.assertEqual(helpers.store_path("future").name, "embedding_store_future.json")
            self.assertEqual(helpers.chunking_settings("future"), {"strategy": "future"})

    def test_recursive_prefers_paragraph_boundaries(self):
        first = "Alpha policy applies here."
        second = "Beta policy applies there."
        self.assertEqual(helpers.recursive_chunks(first + "\n\n" + second, 30, 0), [first, second])

    def test_recursive_character_fallback_preserves_text_and_overlap(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = helpers.recursive_chunks(text, 10, 2)
        self.assertTrue(all(0 < len(chunk) <= 10 for chunk in chunks))
        self.assertEqual(chunks[0] + "".join(chunk[2:] for chunk in chunks[1:]), text)
        for left, right in zip(chunks, chunks[1:]):
            self.assertEqual(left[-2:], right[:2])

    def test_recursive_rejects_invalid_settings(self):
        for size, overlap in [(0, 0), (10, -1), (10, 10)]:
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                helpers.recursive_chunks("text", size, overlap)

    def test_recursive_pdf_keeps_page_metadata_and_character_limit(self):
        chunks, count = helpers.load_pdf_chunks(strategy="recursive")
        self.assertEqual(count, 50)
        self.assertEqual({chunk["page"] for chunk in chunks}, set(range(1, 51)))
        self.assertTrue(all(0 < len(chunk["text"]) <= 600 for chunk in chunks))
        self.assertTrue(all(chunk["source"] == "sample_document.pdf" for chunk in chunks))
        first, second = MagicMock(), MagicMock()
        first.extract_text.return_value = "First page only."
        second.extract_text.return_value = "Second page only."
        with patch("pypdf.PdfReader", return_value=MagicMock(is_encrypted=False, pages=[first, second])):
            chunks, _ = helpers.load_pdf_chunks(strategy="recursive")
        self.assertEqual([(c["page"], c["text"]) for c in chunks], [(1, "First page only."), (2, "Second page only.")])

    def test_strategy_stores_are_separate_and_missing_store_is_actionable(self):
        self.assertNotEqual(helpers.store_path("words"), helpers.store_path("recursive"))
        with patch.object(helpers, "store_path") as path:
            path.return_value.exists.return_value = False
            with self.assertRaisesRegex(ValueError, "--chunking recursive"):
                helpers.load_store(strategy="recursive")

    def test_wrong_strategy_store_is_rejected(self):
        with patch.object(helpers, "store_path") as path:
            path.return_value.read_text.return_value = json.dumps({"chunking": {"strategy": "words"}})
            with self.assertRaisesRegex(ValueError, "Wrong chunking strategy"):
                helpers.load_store(strategy="recursive")

    def test_model_selection_precedence_and_empty_values(self):
        with patch.dict(helpers.os.environ, {}, clear=True):
            self.assertEqual(helpers.get_embedding_model(), "embeddinggemma:latest")
        with patch.dict(helpers.os.environ, {"RAG_EMBEDDING_MODEL": "all-minilm:22m"}):
            self.assertEqual(helpers.get_embedding_model(), "all-minilm:22m")
            self.assertEqual(helpers.get_embedding_model(" embeddinggemma "), "embeddinggemma:latest")
        with patch.dict(helpers.os.environ, {"RAG_EMBEDDING_MODEL": " "}):
            with self.assertRaisesRegex(ValueError, "cannot be empty"):
                helpers.get_embedding_model()

    def test_store_model_mismatch_is_rejected_even_with_same_dimensions(self):
        data = {"source_sha256": "same", "embedding_model": "all-minilm:22m",
                "dimensions": 2, "chunks": [{"embedding": [1, 0]}]}
        with patch.object(helpers, "STORE") as path, patch.object(helpers, "source_hash", return_value="same"):
            path.exists.return_value = True
            path.read_text.return_value = json.dumps(data)
            with self.assertRaisesRegex(ValueError, "Rerun 01_create_embeddings"):
                helpers.load_store("embeddinggemma:latest")
            self.assertEqual(helpers.load_store("all-minilm:22m"), data)
            data["embedding_model"] = "embeddinggemma"
            path.read_text.return_value = json.dumps(data)
            self.assertEqual(helpers.load_store("embeddinggemma:latest"), data)

    def test_real_pdf_has_50_nonempty_pages_and_bounded_chunks(self):
        chunks, page_count = helpers.load_pdf_chunks()
        self.assertEqual(page_count, 50)
        self.assertEqual({chunk["page"] for chunk in chunks}, set(range(1, 51)))
        self.assertTrue(all(0 < len(chunk["text"].split()) <= 100 for chunk in chunks))
        self.assertTrue(all(chunk["source"] == "sample_document.pdf" for chunk in chunks))
        self.assertEqual(len({chunk["id"] for chunk in chunks}), len(chunks))

    def test_pdf_chunk_overlap_does_not_cross_pages(self):
        first, second = MagicMock(), MagicMock()
        first.extract_text.return_value = " ".join(f"word{i}" for i in range(12))
        second.extract_text.return_value = "different page"
        reader = MagicMock(is_encrypted=False, pages=[first, second])
        with patch("pypdf.PdfReader", return_value=reader):
            chunks, count = helpers.load_pdf_chunks(max_words=5, overlap_words=2)
        self.assertEqual(count, 2)
        self.assertEqual(chunks[0]["text"], "word0 word1 word2 word3 word4")
        self.assertEqual(chunks[1]["text"], "word3 word4 word5 word6 word7")
        self.assertEqual(chunks[-1]["page"], 2)
        self.assertEqual(chunks[-1]["text"], "different page")

    def test_scanned_page_requires_ocr(self):
        page = MagicMock()
        page.extract_text.return_value = ""
        with patch("pypdf.PdfReader", return_value=MagicMock(is_encrypted=False, pages=[page])):
            with self.assertRaisesRegex(ValueError, "OCR"):
                helpers.load_pdf_chunks()

    def test_pdf_prompt_retains_source_and_page(self):
        matches = [{"chunk": {"id": 9, "source": "sample_document.pdf", "page": 5, "text": "Refunds take five days."}}]
        prompt = helpers.build_prompt("How long?", matches)
        self.assertIn("Chunk 9 | Source: sample_document.pdf | Page 5", prompt)

    def test_cosine_directions(self):
        self.assertAlmostEqual(helpers.cosine_similarity([1, 0], [5, 0]), 1)
        self.assertAlmostEqual(helpers.cosine_similarity([1, 0], [0, 1]), 0)
        self.assertAlmostEqual(helpers.cosine_similarity([1, 0], [-1, 0]), -1)

    def test_invalid_vectors_are_rejected(self):
        for vector in ([], [0, 0], [float("nan"), 1], [True, 1], [float("inf"), 0]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                helpers.cosine_similarity([1, 0], vector)
        with self.assertRaises(ValueError):
            helpers.cosine_similarity([1, 0], [1, 0, 0])

    def test_incomplete_embedding_response_is_rejected(self):
        with patch.object(helpers, "request_ollama", return_value={"embeddings": [[1, 0]]}):
            with self.assertRaises(ValueError):
                helpers.embed_texts(["one", "two"], "test-model")

    def test_retrieval_uses_stored_model_and_ranks_vectors(self):
        store = {"embedding_model": "stored-model", "dimensions": 2, "chunks": [
            {"id": 1, "text": "First", "embedding": [0, 1]},
            {"id": 2, "text": "Second", "embedding": [1, 0]},
        ]}
        with patch.object(helpers, "embed_texts", return_value=[[1, 0]]) as embed:
            matches = helpers.retrieve("a paraphrase", store, top_k=1)
        embed.assert_called_once_with(["a paraphrase"], "stored-model")
        self.assertEqual([item["chunk"]["id"] for item in matches], [2])
        self.assertIn("Second", helpers.build_prompt("a paraphrase", matches))
        self.assertNotIn("First", helpers.build_prompt("a paraphrase", matches))

    def test_stale_store_is_rejected(self):
        with patch.object(helpers, "STORE") as store, patch.object(helpers, "source_hash", return_value="new"):
            store.exists.return_value = True
            store.read_text.return_value = json.dumps({"source_sha256": "old"})
            with self.assertRaisesRegex(ValueError, "document changed"):
                helpers.load_store()

    def test_word_count_baseline_has_no_match_for_synonyms(self):
        store = {"chunks": [{"id": 1, "text": "Refund policy"}]}
        self.assertEqual(helpers.word_count_matches("reimbursement eligibility", store), [])


if __name__ == "__main__":
    unittest.main()
