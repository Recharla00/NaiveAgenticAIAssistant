"""RAG baby step 6: retrieve the chunks closest to a question.

Run from the project folder after step 4 has created chunk_store.json:
    python rag/06_retrieve_chunks.py
    python rag/06_retrieve_chunks.py --question "delivery time" --top-k 1

Flow: load stored chunks -> vectorize question -> score -> sort -> select.
Top-k means the maximum number of chunks to return. We skip zero-score chunks.

We compare word-count vectors using cosine similarity. For these non-negative
vectors, 0 means no shared words; 1 means the vectors point in the same direction.
Dividing by vector lengths keeps larger word counts from dominating by size alone.
The score measures word overlap, NOT the probability that a chunk answers us.

This simple method still counts common words like 'the' and 'is', which can
produce weak matches. It does not understand synonyms. Inspect the retrieved
text yourself. No LLM is called and no answer is generated in this step.

Practice: compare "refund policy", "delivery time", and "galaxy telescope".
Try --top-k 1 and --top-k 3. After changing the source document, rerun step 4
to rebuild the stored vectors before running this script again.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys


def make_vector(text):
    """Use the same word-count method as the stored chunks and step 5."""
    words = re.findall(r"\b\w+\b", text.lower())
    return dict(Counter(words))


def cosine_similarity(left, right):
    """Compare two word-count dictionaries, treating missing words as zero."""
    # Multiply counts for matching words, then add those products.
    dot_product = 0
    for word, count in left.items():
        dot_product += count * right.get(word, 0)

    # A vector's length is the square root of the sum of its squared counts.
    left_length = math.sqrt(sum(count ** 2 for count in left.values()))
    right_length = math.sqrt(sum(count ** 2 for count in right.values()))

    # Empty vectors have length zero, so avoid division by zero.
    if left_length == 0 or right_length == 0:
        return 0.0

    return dot_product / (left_length * right_length)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Find relevant stored chunks.")
    parser.add_argument("--question", default="What is the refund policy?")
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    # 1. Load the chunks and vectors saved in step 4.
    store_path = Path(__file__).with_name("chunk_store.json")
    if not store_path.exists():
        parser.error("Store missing. Run: python rag/04_store_vectors.py")
    chunk_store = json.loads(store_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunk_store)} stored chunk(s) from {store_path.name}.")

    # 2. Represent the question using the same method as the stored chunks.
    question = args.question.strip()
    question_vector = make_vector(question)
    if not question_vector:
        parser.error("Please enter a question containing words or numbers.")
    print("Question:", question)
    print("Question vector:", question_vector)

    # 3. Score every chunk and keep its text and source beside the score.
    scored_chunks = []
    print("\nSimilarity scores for all chunks:")
    for entry in chunk_store:
        #print(f"  entry:", entry)
        score = cosine_similarity(question_vector, entry["vector"])
        scored_chunks.append({"score": score, "chunk": entry})
        print(f"  Chunk {entry['id']}: {score:.3f}")

    # 4. Sort from highest score to lowest. lambda selects the value to sort by.
    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    # 5. Select up to top-k chunks that share at least one word with the question.
    matches = [item for item in scored_chunks if item["score"] > 0]
    retrieved_chunks = matches[:args.top_k]
    print(f"\nRetrieved {len(retrieved_chunks)} chunk(s):")
    for rank, result in enumerate(retrieved_chunks, start=1):
        entry = result["chunk"]
        shared_words = sorted(set(question_vector) & set(entry["vector"]))
        print(f"\nMatch {rank}: chunk {entry['id']} | score {result['score']:.3f}")
        print("Source:", entry["source"])
        print("Shared words:", ", ".join(shared_words))
        print(entry["text"])

    if not retrieved_chunks:
        print("No shared words found. Try another question or add relevant source text.")
    else:
        print("\nNext step: combine the retrieved text and question into a prompt.")
