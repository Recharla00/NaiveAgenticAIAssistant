# Advanced RAG: policy PDFs, hybrid search, and parent sections

This local learning pipeline uses a **100-page fictional policy handbook**, Ollama EmbeddingGemma, PostgreSQL with pgvector in Docker, and the existing `qwen3:1.7b` answer model. There are 50 policy topics with two pages per topic: policy/eligibility and procedure/case handling. Each page has headings, bullet conditions, and a reference table. Ten procedure pages also include raster workflow illustrations with selectable text captions. The editable source is `policy_topics.json`; it is synthetic training material, not real business terms.

## Flow

```text
PDF -> extract page text, font headings, and table geometry
    -> page/section parents + small children
    -> recursive OR layout-aware chunking
    -> EmbeddingGemma document embeddings -> PostgreSQL/pgvector

Question -> EmbeddingGemma query embedding
    -> BM25 keyword ranking + pgvector cosine ranking
    -> Reciprocal Rank Fusion (RRF)
    -> return child chunks OR expand/deduplicate parent sections
    -> context budget -> prompt with source IDs/pages -> Qwen answer
```

## Setup (PowerShell, from project root)

Open Docker Desktop and Ollama first. A dedicated database uses **127.0.0.1:5433**; it does not reuse another project's PostgreSQL database.

```powershell
python -m venv rag/advanced_rag/.venv
.\rag\advanced_rag\.venv\Scripts\python.exe -m pip install -r rag/advanced_rag/requirements.txt
Copy-Item rag/advanced_rag/.env.example rag/advanced_rag/.env
docker compose -f rag/advanced_rag/compose.yaml up -d --wait
ollama pull embeddinggemma:latest
ollama list
.\rag\advanced_rag\.venv\Scripts\python.exe rag/advanced_rag/00_check_setup.py
```

Copy `.env.example` only on first setup; preserve an existing `.env`. The defaults work without a `.env`. The included password is for this localhost learning database only. If changing it, update both `POSTGRES_PASSWORD` and the password in `DATABASE_URL` before first database startup. Editing the environment later does not change credentials already stored in the PostgreSQL volume.

`EMBEDDING_MODEL`, `LLM_MODEL`, `OLLAMA_URL`, and `DATABASE_URL` can be set through `.env` or environment variables (environment takes precedence). The vector schema intentionally requires 768 dimensions. The current model's digest is recorded so a new model downloaded under the same `latest` tag invalidates existing indexes.

## 1. Create or inspect the handbook

The generated PDF is included at `output/pdf/policy_handbook.pdf`. To regenerate it after editing topics:

```powershell
.\rag\advanced_rag\.venv\Scripts\python.exe rag/advanced_rag/create_policy_pdf.py
```

This overwrites the sample PDF; do not run it after substituting your own document unless you want to restore the sample. Re-ingest after any document change.

## 2. Build both chunking indexes

```powershell
.\rag\advanced_rag\.venv\Scripts\python.exe rag/advanced_rag/01_ingest.py --chunking all
```

The default child limit is 700 **characters**, target overlap 100 characters. Customize with `--chunk-size 900 --overlap 120`; select just one with `--chunking recursive` or `--chunking layout`.

| Strategy | Behavior |
| --- | --- |
| `recursive` | LangChain recursive character splitter on page text: paragraphs, lines, spaces, then characters |
| `layout` | pdfplumber detects table geometry and heading fonts; table rows retain column labels, and sections are split separately |

Both retain page numbers and parent IDs. Parents are complete page sections in this deliberately one-section-per-page handbook. This is **not** a general multi-page hierarchy parser. For arbitrary PDFs, page-based parents may need replacing with multi-page section detection. Figure captions are indexed as text and identified as `figure_caption` blocks; image pixels are not embedded or interpreted. Images without useful text captions require an additional OCR/vision stage. Layout extraction is heuristic, not an OCR or vision model; scanned pages fail with an OCR message. Complex merged tables, multi-column reading order, and unusual fonts need additional validation. Oversized table rows can be fragmented by the child size limit. Character limits are not token limits; Ollama rejects overlong embedding input rather than silently truncating it.

Unchanged indexes are skipped when the PDF hash, model digest, pipeline version, and chunk settings match. Use `--force` to rebuild anyway. Identical document texts share embeddings within a build. Each run replaces only selected strategy indexes in a transaction. A failure leaves prior committed indexes intact. The metadata, page parents, child text, and vectors are stored in PostgreSQL; no JSON vector store is used. Dense retrieval uses **exact pgvector cosine search** for reproducible results on this small dataset. An ANN/HNSW index is intentionally not used here.

## 3. Compare retrieval and chunking

```powershell
.\rag\advanced_rag\.venv\Scripts\python.exe rag/advanced_rag/02_retrieve.py --compare --question "How long does standard delivery take?"
```

One `--compare` flag runs all four combinations:

| Chunking | Retrieval |
| --- | --- |
| Recursive | Hybrid |
| Recursive | Parent-child |
| Layout | Hybrid |
| Layout | Parent-child |

`hybrid` returns child chunks ranked by combining BM25 and dense results using RRF (constant 60). `parent_child` uses that same hybrid child ranking, expands to complete parent page sections, and deduplicates parents before taking top-k. These approaches are complementary: parent-child is a context expansion step on top of hybrid retrieval, not an alternative lexical scoring algorithm.

BM25 is implemented in Python using term frequency, document frequency, length normalization (`k1=1.5`, `b=0.75`), and positive Robertson IDF. The sparse side reads child text from PostgreSQL; **pgvector itself does not implement BM25**. This full-corpus scan is suitable for the learning dataset. A large deployment should use a dedicated sparse index or a suitable PostgreSQL search extension. No separate word-count cosine baseline is included here.

For one configuration:

```powershell
.\rag\advanced_rag\.venv\Scripts\python.exe rag/advanced_rag/02_retrieve.py --chunking layout --retrieval parent_child --top-k 3 --candidates 20 --question "Who pays for return shipping?"
```

`--candidates` controls each candidate ranking before fusion and must be at least `--top-k`. Comparisons use one query embedding across all combinations. RRF scores are not confidence probabilities. Results show content, source pages, source IDs, and parent matches. Reports are saved to ignored `results/last_retrieve.json`. IDs belong to their own chunking strategy; do not compare IDs across strategies. All selected indexes must exist and be current.

## 4. Generate a cited answer

```powershell
.\rag\advanced_rag\.venv\Scripts\python.exe rag/advanced_rag/03_generate_answer.py --chunking layout --retrieval parent_child --question "What minimum purchase qualifies for free delivery?"
```

Add `--prompt-only` to inspect the context without Qwen. Add `--compare` to generate an answer for all four combinations (four generation calls). The default context budget is 14,000 characters, configurable with `--context-chars`. Whole results that do not fit are skipped and exact duplicate text is removed. A second limit, `--context-tokens 3000`, budgets evidence conservatively using UTF-8 byte length rather than assuming four characters per token. This overestimates tokens for ordinary English and preserves space for the question, instructions, output, and 512 tokens of template overhead. It is not the model's exact tokenizer. `--context-window 4096` and `--max-output-tokens 600` control Ollama's context and answer limits. The answer flow releases EmbeddingGemma before loading Qwen and releases Qwen after completion (`keep_alive=0`), reducing peak laptop memory use. Model files remain installed and reload automatically. This makes answer comparisons slower but avoids keeping two models resident. Actual `prompt_eval_count` (input) and `eval_count` (output) from Ollama are printed and saved in `results/last_answer.json`. Reducing `--top-k` or `--context-tokens` reduces prompt cost; `--compare` generates four answers and costs more.

The report checks that citation IDs and page numbers belong to the evidence actually included. Citation validity alone does not prove that the cited source supports the claim. Truncated answers are reported as errors. Parent sections preserve table column labels for layout indexes. If a whole parent cannot fit, use `--retrieval hybrid` for smaller child chunks or raise the evidence budget.

The prompt treats source material as untrusted data, asks for citations and exact conditions, and asks the model to abstain when the answer is absent. This is **not a calibrated no-answer detector or an automated faithfulness guarantee**. Check evidence and citations, especially for unrelated questions. A reranker, confidence calibration, and answer-grounding evaluation are useful later extensions, not silently claimed features.

## 5. Evaluate

```powershell
.\rag\advanced_rag\.venv\Scripts\python.exe -m unittest discover -s rag/advanced_rag -p "test_*.py"
.\rag\advanced_rag\.venv\Scripts\python.exe rag/advanced_rag/04_evaluate.py --compare
```

Ten editable answerable questions in `eval_cases.json` identify expected policy pages. The comparison reports Hit@k, MRR, retrieved context size, and retrieval time (excluding query embedding). Results are written to `results/evaluation.json`. This is a small page-level smoke evaluation, not an industry benchmark or proof of answer quality. Parent and child results have different context sizes, which the report makes visible. Evaluate actual answer completeness before choosing a winner. The hand-authored expected pages apply only to the included handbook.

## Stop and resume

```powershell
docker compose -f rag/advanced_rag/compose.yaml stop
docker compose -f rag/advanced_rag/compose.yaml start
```

The named `advanced-rag_pgdata` volume preserves indexes. `docker compose down` removes the container but keeps the volume; **do not add `-v` unless you intend to erase the database**. Ollama runs on the host and is managed separately.

Troubleshooting: `00_check_setup.py` checks Docker/PostgreSQL connectivity, vector extension, PDF presence, and model availability. `--debug` prints a traceback for any numbered command. Run `docker compose -f rag/advanced_rag/compose.yaml logs db` for database startup errors. Start Ollama if its API is unreachable. Re-run ingestion on a stale-index error.

## References

- [pgvector](https://github.com/pgvector/pgvector)
- [Recursive splitter](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter)
- [EmbeddingGemma query/document prefixes](https://huggingface.co/google/embeddinggemma-300m)

This folder has no runtime imports from `basic_rag`, `intermediate_rag`, or `llm`.

## Suggested practice sequence

1. Run setup checks, then ingest both strategies once.
2. Run `02_retrieve.py --compare` and inspect the BM25 and dense contributions in the saved JSON.
3. Run `04_evaluate.py --compare` to compare all configurations on the same ten questions.
4. Generate one answer with `--chunking layout --retrieval hybrid`; compare the evidence and citations manually.
5. Try `--retrieval parent_child` when conditions are missing from a small child. Check the input token count against the child-only answer.
6. Add your own questions, including unanswerable questions, before tuning retrieval. The shipped page-level evaluation does not score refusals.

This is a complete local learning flow, not a production deployment: it has no user access control, document permissions, database backups, calibrated rejection threshold, image vision/OCR, or general multi-document ingestion. A cross-encoder reranker can be tested later if evaluation shows ranking failures; it is not needed to run this version. The sample policies and tables are synthetic.

If Ollama reports `std::bad_alloc`, it could not allocate enough memory. The default answer flow already unloads the embedding model and uses a 4,096-token context. Close unnecessary applications or lower `--context-window` and `--context-tokens` together. Do not increase the context window blindly on a memory-limited laptop.
