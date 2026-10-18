"""RAG baby step 1: load a plain-text (.txt) document.

Run from the project folder:
    python rag/01_load_text.py

Practice: edit sample_document.txt, save it, and run this script again.
No extra packages or AI model are needed for this step.
"""

from pathlib import Path


def load_text(file_path):
    """Read a UTF-8 text file and return its contents as a Python string."""
    return file_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    # 1. Find the document beside this Python file.
    # __file__ is this script's path; with_name changes just its filename.
    document_path = Path(__file__).with_name("sample_document.txt")

    # 2. Load the document into the variable called text.
    text = load_text(document_path)

    # 3. Print the text so we can check that it loaded correctly.
    print(f"Loaded: {document_path.name}\n")
    print(text)
