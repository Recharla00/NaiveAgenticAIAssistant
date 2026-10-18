"""RAG baby step 9: check whether retrieval selects useful evidence.

Run from the project folder:
    python rag/09_evaluate_retrieval.py

For each question, we define acceptable chunk IDs BEFORE running retrieval.
We ask the existing step-7 retriever for one match and compare it with that list.
An empty expected_ids list means our document cannot answer the question, so
we want no match. Chunk 7 is acceptable for several questions because its table
also contains refund, delivery, and support information.

These expectations describe our current sample document. If you change its
chunks, rebuild the store with step 4 and review the expected IDs below.

This checks retrieval only. It does not call Ollama or grade generated answers.
The result describes these five examples, not overall RAG accuracy. Failures
are useful: they show what we should improve before trusting the results.

Practice: add another question and its acceptable chunk IDs to TEST_CASES.
Try an unrelated question containing common words such as 'the' and 'is'.
"""

import json
from pathlib import Path
import runpy
import sys


TEST_CASES = [
    {"question": "What is the refund policy?", "expected_ids": [2, 7]},
    {"question": "How long does delivery take?", "expected_ids": [3, 7]},
    {"question": "When is customer support available?", "expected_ids": [4, 7]},
    {"question": "galaxy telescope", "expected_ids": []},
    {"question": "What is the capital of India?", "expected_ids": []},
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    folder = Path(__file__).resolve().parent

    # 1. Load the same stored chunks used by our RAG flow.
    store_path = folder / "chunk_store.json"
    if not store_path.exists():
        print("Store missing. Run: python rag/04_store_vectors.py")
        return 1
    chunk_store = json.loads(store_path.read_text(encoding="utf-8"))
    if not chunk_store:
        print("The store is empty. Add document text and rerun step 4 first.")
        return 1

    # 2. Reuse the ACTUAL retrieval function from step 7.
    # Our lesson filenames start with digits, so we use runpy to load that file's
    # functions. Its __main__ block does not run, and no prompt file is changed.
    lesson = runpy.run_path(str(folder / "07_build_prompt.py"))
    retrieve_chunks = lesson["retrieve_chunks"]

    # 3. Compare the top result with the expected evidence for each question.
    passed_count = 0
    for case in TEST_CASES:
        question = case["question"]
        expected_ids = case["expected_ids"]
        results = retrieve_chunks(question, chunk_store, top_k=1)
        actual_id = results[0]["chunk"]["id"] if results else None

        if expected_ids:
            passed = actual_id in expected_ids
        else:
            passed = actual_id is None
        if passed:
            passed_count += 1

        print(f"\n{'PASS' if passed else 'FAIL'}: {question}")
        print("  Expected:", expected_ids if expected_ids else "no matching chunk")
        print("  Retrieved:", actual_id if actual_id is not None else "no matching chunk")
        if results:
            entry = results[0]["chunk"]
            print(f"  Score: {results[0]['score']:.3f}")
            print("  Evidence:", entry["text"].replace("\n", " "))
            query_words = lesson["make_vector"](question)
            shared_words = sorted(set(query_words) & set(entry["vector"]))
            print("  Shared words:", ", ".join(shared_words))

    # 4. Inspect failures before changing the retrieval method.
    print(f"\nResult: {passed_count}/{len(TEST_CASES)} retrieval checks passed.")
    print("Common words can cause irrelevant matches. A positive score is not proof")
    print("that a chunk answers the question. Inspect the failed examples above.")
    print("No model was called; answer quality still needs a separate check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
