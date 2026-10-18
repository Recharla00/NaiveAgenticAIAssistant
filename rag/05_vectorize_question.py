"""RAG baby step 5: turn a question into a vector.

Run from the project folder:
    python rag/05_vectorize_question.py

Try your own question:
    python rag/05_vectorize_question.py --question "How long does delivery take?"

We use exactly the same word-count method as steps 3 and 4. This gives words
the same meaning as vector dimensions in both questions and document chunks.
For example, the 'refund' count in a question can be compared with the 'refund'
count in a chunk. A missing word has a count of zero.

This step only prepares the question. Comparing it with stored chunks comes
next. These word-count vectors do not understand synonyms or sentence meaning.
No extra packages or AI model are needed.

Practice: try "refund refund policy". Notice that the count for 'refund' is 2.
Then try "REFUND refund policy" and observe that the vector stays the same.
"""

import argparse
from collections import Counter
import re
import sys


def make_vector(text):
    """Use the same lowercasing, tokenization, and counting as our chunks."""
    words = re.findall(r"\b\w+\b", text.lower())
    return dict(Counter(words))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    # 1. Accept a question, or use a sample question when none is provided.
    parser = argparse.ArgumentParser(description="Turn a question into word counts.")
    parser.add_argument("--question", default="What is the refund policy?")
    args = parser.parse_args()
    question = args.question.strip()

    # 2. Convert the question into numbers, just like our document chunks.
    question_vector = make_vector(question)
    if not question_vector:
        parser.error("Please enter a question containing words or numbers.")

    # 3. Inspect the result. Common words stay in, just as they did in step 3.
    print("Question:", question)
    print("\nQuestion vector:", question_vector)
    print("\nEach word and its count:")
    for word, count in question_vector.items():
        print(f"  {word}: {count}")

    print("\nNext step: compare this vector with the stored chunk vectors.")
