"""RAG baby step 4: store each chunk together with its vector.

Run from the project folder:
    python rag/04_store_vectors.py

This creates rag/chunk_store.json, then reads it back to demonstrate loading.
Each entry has an ID, a source filename, the original text, and its word counts.
We keep the text because the LLM will eventually need readable context;
the vector is used to find relevant chunks.

JSON is a readable file format for saving lists and dictionaries. This small
file is our practice store, not a vector database or a semantic embedding index.
No extra packages or AI model are needed.

Practice: open chunk_store.json and find a chunk's text and matching vector.
After editing sample_document.txt, rerun this script to rebuild the store.
Each run replaces the generated JSON file with the current document's chunks.
"""

from collections import Counter
import json
from pathlib import Path
import re
import sys


def make_vector(text):
    """Use the same word-count method as step 3."""
    words = re.findall(r"\b\w+\b", text.lower())
    return dict(Counter(words))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    # Repeat step 1: load the document beside this script.
    document_path = Path(__file__).with_name("sample_document.txt")
    text = document_path.read_text(encoding="utf-8")

    # Repeat step 2: split the document into paragraphs.
    chunks = []
    for paragraph in text.split("\n\n"):
        if paragraph.strip():
            chunks.append(paragraph.strip())

    # Step 3 + new step 4: keep each vector beside the text it represents.
    chunk_store = []
    for number, chunk in enumerate(chunks, start=1):
        entry = {
            "id": number,
            "source": document_path.name,
            "text": chunk,
            "vector": make_vector(chunk),
        }
        chunk_store.append(entry)

    # Save the Python list as JSON text on disk.
    # indent makes it readable; ensure_ascii=False preserves symbols like ₹.
    store_path = Path(__file__).with_name("chunk_store.json")
    json_text = json.dumps(chunk_store, indent=2, ensure_ascii=False)
    store_path.write_text(json_text, encoding="utf-8")
    print(f"Saved {len(chunk_store)} chunks and their vectors to {store_path.name}.")

    # Read it back: JSON text becomes a Python list of dictionaries again.
    loaded_store = json.loads(store_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(loaded_store)} entries back from disk.")

    if loaded_store:
        print("\nFirst stored entry:")
        print(json.dumps(loaded_store[0], indent=2, ensure_ascii=False))
    else:
        print("The document is empty. Add some text and run this script again.")
