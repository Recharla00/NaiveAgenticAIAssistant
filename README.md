# Learn RAG in Baby Steps

A beginner Python project for learning Retrieval-Augmented Generation (RAG), one small step at a time.

RAG finds relevant information in your documents, adds that information to a prompt, and asks a language model to answer using it. It does not train the model or permanently teach it your document.

```text
Prepare the knowledge:
Document -> Chunks -> Word-count vectors -> Saved chunk store

Answer a question:
Question -> Question vector -> Retrieve chunks -> Build prompt -> Generate answer

Check the results:
Test questions -> Compare retrieved chunks with expected evidence
```

This first version uses **word-count vectors**, not semantic embeddings. Each vector is a dictionary such as `{"refund": 2, "policy": 1}`. Retrieval compares these vectors using cosine similarity. This makes the mechanics visible before introducing embedding models or a vector database.

## Requirements

- Python 3. The examples have been run with Python 3.13.
- No additional Python packages or API key are needed.
- For answer generation, [Ollama](https://ollama.com/) must be running with the default model, `qwen3.5:4b`, available locally. The other numbered lessons do not need Ollama.

Run the commands below from the project root, the folder containing this README and `rag`.

On Windows PowerShell, enable UTF-8 for the current terminal session so the document's rupee symbol prints correctly, including in lesson 1:

```powershell
$env:PYTHONUTF8 = "1"
python --version
```

Check Ollama before generating an answer:

```powershell
ollama list
ollama ps
```

`ollama list` shows installed models; `ollama ps` shows currently loaded models. If the service is unavailable, open the Ollama app or run `ollama serve` in a separate terminal. The scripts connect to `http://127.0.0.1:11434/api/generate` on your own computer. See the [Ollama generation API reference](https://docs.ollama.com/api/generate).

## Follow the lessons

Each numbered file is a small lesson. Read its opening explanation, run it, and inspect its output before moving on.

| Step | Python file | What you learn |
| --- | --- | --- |
| 1 | [01_load_text.py](rag/01_load_text.py) | Read a UTF-8 text document into a Python string. |
| 2 | [02_split_text.py](rag/02_split_text.py) | Split the text at blank lines into paragraph chunks. |
| 3 | [03_vectorize_text.py](rag/03_vectorize_text.py) | Count words in each chunk to create vectors. |
| 4 | [04_store_vectors.py](rag/04_store_vectors.py) | Save chunk IDs, sources, text, and vectors as JSON. |
| 5 | [05_vectorize_question.py](rag/05_vectorize_question.py) | Represent a question using the same word-count method. |
| 6 | [06_retrieve_chunks.py](rag/06_retrieve_chunks.py) | Score stored chunks and select the closest matches. |
| 7 | [07_build_prompt.py](rag/07_build_prompt.py) | Combine instructions, retrieved context, and the question. |
| 8 | [08_generate_answer.py](rag/08_generate_answer.py) | Send the saved prompt to Ollama and print its answer. |
| 9 | [09_evaluate_retrieval.py](rag/09_evaluate_retrieval.py) | Compare retrieval results with expected evidence. |

```powershell
python rag/01_load_text.py
python rag/02_split_text.py
python rag/03_vectorize_text.py
python rag/04_store_vectors.py
python rag/05_vectorize_question.py
python rag/06_retrieve_chunks.py
python rag/07_build_prompt.py
python rag/08_generate_answer.py
python rag/09_evaluate_retrieval.py
```

Steps 1–3 and 5 print intermediate results for learning. Step 4 repeats document preparation and saves the store. Step 7 loads that store, repeats retrieval, and saves the prompt. Step 8 uses the saved prompt exactly as it is.

## Run a complete question through the flow

With Ollama running and the default model available:

```powershell
python rag/04_store_vectors.py
python rag/07_build_prompt.py --question "What is the refund policy?"
python rag/08_generate_answer.py
```

The sample document says refunds can be requested within 30 days, with an unused item and a receipt. Check that the generated answer agrees with the retrieved text and cites a supporting chunk. Wording may vary.

For another question, rebuild the prompt and generate again:

```powershell
python rag/07_build_prompt.py --question "How long does delivery take?" --top-k 1
python rag/08_generate_answer.py
```

`--top-k 1` selects at most one matching chunk; the default is two. Zero-score chunks are excluded. To select another installed model, use `--model MODEL_NAME` with step 8. The generation settings were tested with `qwen3.5:4b`; another model may need different settings.

## Understand the files and refresh order

| File | Purpose | When it changes |
| --- | --- | --- |
| [sample_document.txt](rag/sample_document.txt) | Fictional shop policies used as source text. | Edit this to practice with different information. |
| [chunk_store.json](rag/chunk_store.json) | Saved chunks and word-count vectors. | Step 4 replaces it using the current source document. |
| [rag_prompt.txt](rag/rag_prompt.txt) | Instructions, retrieved context, and the current question. | Step 7 replaces it for the question you provide. |

After editing the source document, run **step 4 -> step 7 -> step 8**. For just a new question, run **step 7 -> step 8**. Running step 8 alone answers the question already stored in the prompt file.

## Evaluate and practice

```powershell
python rag/09_evaluate_retrieval.py
```

The evaluation reuses step 7's retrieval function and checks the highest-scoring result against acceptable chunk IDs. It does not call Ollama or grade generated answers.

With the provided sample store, the initial result was **4 of 5 checks passed**. The question “What is the capital of India?” retrieved the refund policy because they share “is,” “of,” and “the.” This is a useful retrieval failure, not evidence that the document answers the question. The score describes only these examples, not overall RAG accuracy.

Try these exercises:

1. Change the refund period to 45 days, rebuild the store and prompt, and inspect the answer.
2. Add a blank line to split a paragraph and observe the new chunk boundaries.
3. Repeat a word and inspect how its count changes in step 3.
4. Compare retrieval for `refund policy`, `delivery time`, and `galaxy telescope`.
5. Add a question and acceptable chunk IDs to `TEST_CASES` in step 9. Review those IDs whenever you change chunk boundaries.

## Standalone overview

[simple_rag.py](rag/simple_rag.py) demonstrates the whole flow in one file:

```powershell
python rag/simple_rag.py --retrieve-only
python rag/simple_rag.py --step
python rag/simple_rag.py --question "What is the refund policy?"
```

This overview uses its own embedded sample text and removes some common words during retrieval. It does not read the numbered lessons' document, chunk store, or saved prompt, so its results can differ.

## Troubleshooting and current limits

- **Missing store:** run step 4 before steps 6, 7, or 9.
- **Missing or outdated prompt:** run step 7 before step 8.
- **Ollama connection error:** check that Ollama is running and inspect `ollama list` for the selected model.
- **Windows Unicode error:** set `$env:PYTHONUTF8 = "1"` in the current PowerShell session and rerun.
- **Irrelevant chunks:** common words can create matches. Similarity is not answer confidence; inspect the retrieved evidence.
- **Unexpected evaluation failures after edits:** check the source, rebuild the store, and review the expected chunk IDs.

The lessons currently load plain-text files, split on blank lines, and use exact word counts. They do not parse PDFs, limit long paragraph sizes, understand synonyms, or use a vector database. Model answers and citations still need checking against the source. Later improvements can include semantic embeddings, better chunking, and evaluation of answer quality.
