"""RAG baby step 3: represent each chunk as numbers.

Run from the project folder:
    python rag/03_vectorize_text.py

We use word counts to make the idea of a vector easy to see:
    "refund refund policy" -> {"refund": 2, "policy": 1}

Each word is a vector dimension, and its count is the value. A word missing
from a chunk has a count of zero. Storing only non-zero counts in a dictionary
is called a sparse vector.

This is a learning version, NOT a semantic embedding model. It counts words;
it does not understand that "refund" and "reimbursement" have similar meanings.
An embedding model is a later upgrade. No packages or AI model are needed here.

Practice: repeat a word in sample_document.txt, save, and rerun. Find the
changed count. Changing a word's capitalization should not change its count.
"""

from collections import Counter
from pathlib import Path
import re
import sys


def make_vector(text):
    """Convert a text string into a dictionary of word counts."""
    # 1. Treat 'Refund' and 'refund' as the same word.
    lowercase_text = text.lower()

    # 2. Find words and numbers, ignoring punctuation and whitespace.
    words = re.findall(r"\b\w+\b", lowercase_text)

    # 3. Count how many times each word appears.
    return dict(Counter(words))


if __name__ == "__main__":
    # Support symbols such as the rupee sign in Windows output.
    sys.stdout.reconfigure(encoding="utf-8")

    # Repeat step 1: load our document.
    document_path = Path(__file__).with_name("sample_document.txt")
    text = document_path.read_text(encoding="utf-8")

    # Repeat step 2: split at blank lines and skip empty pieces.
    chunks = []
    for paragraph in text.split("\n\n"):
        if paragraph.strip():
            chunks.append(paragraph.strip())

    # New step 3: create one vector for each chunk, in the same order.
    vectors = []
    for chunk in chunks:
        vectors.append(make_vector(chunk))

    print("Small example:")
    print("refund refund policy ->", make_vector("refund refund policy"))

    print(f"\nLoaded: {document_path.name}")
    print(f"Created {len(vectors)} word-count vectors for {len(chunks)} chunks.")

    # zip pairs each chunk with its corresponding vector.
    for number, (chunk, vector) in enumerate(zip(chunks, vectors), start=1):
        print(f"\n--- Chunk {number} ---")
        print(chunk)
        print("Word-count vector:", vector)
