"""RAG baby step 2: split a document into smaller pieces called chunks.

Run from the project folder:
    python rag/02_split_text.py

For this first example, each paragraph becomes one chunk. A blank line
separates paragraphs. A normal line break stays inside the same chunk.
Later, retrieval can select relevant chunks instead of the entire document.

Practice: add a blank line inside a paragraph in sample_document.txt,
save it, and run this script again. Observe how the chunks change.
This simple method does not limit chunk size; long paragraphs stay together.
"""

from pathlib import Path
import sys


def split_text(text):
    """Return a list of non-empty paragraphs."""
    paragraphs = text.split("\n\n")  # Two newline characters mean a blank line.
    chunks = []

    for paragraph in paragraphs:
        chunk = paragraph.strip()  # Remove whitespace at the beginning and end.
        if chunk:  # Skip empty pieces caused by extra blank lines.
            chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    # Let Windows output symbols such as the rupee sign in our document.
    sys.stdout.reconfigure(encoding="utf-8")

    # 1. Load the same document we used in step 1.
    document_path = Path(__file__).with_name("sample_document.txt")
    text = document_path.read_text(encoding="utf-8")

    # 2. Turn one text string into a list of smaller text strings.
    chunks = split_text(text)

    # 3. Inspect each chunk. enumerate gives us its number and its text.
    print(f"Loaded: {document_path.name}")
    print(f"Total chunks: {len(chunks)}")

    for number, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {number} ---")
        print(chunk)
