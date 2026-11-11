# Semantic RAG with a 50-page PDF

This standalone example reads **sample_document.pdf**, a synthetic handbook with 50 distinct shop-policy topics. Each page contains a rule, procedure, worked example, and practice question. The policies are fictional learning data, not the terms of a real business.

Everything stays inside `intermediate_rag`. The existing `basic_rag` and `llm` lessons are unchanged.

```text
50-page PDF -> Extract text per page -> Small overlapping chunks
-> Embedding model -> JSON vector store -> Semantic retrieval
-> Prompt with PDF page references -> Generated answer
```

## Setup on Windows

Run from the project root. A separate environment keeps PDF dependencies isolated:

```powershell
python -m venv rag/intermediate_rag/.venv
.\rag\intermediate_rag\.venv\Scripts\python.exe -m pip install -r rag/intermediate_rag/requirements.txt
```

Use that environment's Python for the lessons. No activation or execution-policy change is required. The runtime dependencies are `pypdf` for PDF extraction and `langchain-text-splitters` for recursive splitting. Ollama calls and vector comparison use Python's standard library.

Ollama must be running. Open its app or run `ollama serve` in a separate terminal if it is stopped. Do not start a second server if one is already running.

```powershell
ollama pull embeddinggemma:latest
ollama list
```

`embeddinggemma:latest` creates embeddings. The existing `qwen3:1.7b` model writes answers. They serve different purposes.

## Choose the embedding model dynamically

The shared `EMBEDDING_MODEL` variable in `semantic_helpers.py` defaults to `embeddinggemma:latest`. You can edit it directly for VS Code debugging, or override it without editing code:

```powershell
$env:RAG_EMBEDDING_MODEL = "embeddinggemma:latest"
ollama pull $env:RAG_EMBEDDING_MODEL
.\rag\intermediate_rag\.venv\Scripts\python.exe rag/intermediate_rag/01_create_embeddings.py
.\rag\intermediate_rag\.venv\Scripts\python.exe rag/intermediate_rag/02_retrieve.py --question "How long does standard delivery take?"
.\rag\intermediate_rag\.venv\Scripts\python.exe rag/intermediate_rag/03_generate_answer.py --question "How long does standard delivery take?"
```

Priority: `--embedding-model` command-line option > `RAG_EMBEDDING_MODEL` environment variable > shared default. All three scripts support `--embedding-model`. The environment variable applies to programs launched from that PowerShell session; a VS Code debugger may use a different environment.

To switch back, set `$env:RAG_EMBEDDING_MODEL = "all-minilm:22m"` and rerun all three steps. Always rebuild with lesson 1 after changing models, even when their vector dimensions match. Lessons 2 and 3 reject a store made with a different selected model. One store per chunking strategy is kept; a successful rebuild replaces that strategy's store. Use `--chunking all` to rebuild both when changing models. Updating the same `latest` tag also requires rebuilding; tag updates are not detected automatically.

Lesson 1 still accepts `--model` as an alias. In lesson 3, `--model` continues to select the **answer-generation** model, independently of `--embedding-model`.

EmbeddingGemma requires Ollama 0.11.10 or newer according to its [Ollama model page](https://ollama.com/library/embeddinggemma). If it fails to load, check `ollama --version`. Other embedding models may require their own task-specific input formatting; selecting a model name does not configure that automatically.

Compare relevant pages returned for the same questions, not just raw cosine scores across models. A larger score is not necessarily a more accurate result.

## Lesson 1: extract and embed the PDF

```powershell
.\rag\intermediate_rag\.venv\Scripts\python.exe rag/intermediate_rag/01_create_embeddings.py
```

[01_create_embeddings.py](01_create_embeddings.py) extracts text from every page, creates chunks of at most **100 words** with **20 words of overlap**, and sends them to Ollama in batches of 12. A chunk never crosses a page boundary. Progress output shows the page count, chunk count, and completed batches.

The generated `embedding_store.json` contains the embedding model, dimensions, source fingerprint, page count, chunking settings, and each chunk's text, vector, source filename, and PDF page number. Word chunking uses `embedding_store.json`; recursive chunking uses `embedding_store_recursive.json`. Each successful build replaces only its strategy's store; Git ignores both.

Use `--batch-size 8` to send fewer chunks per request. Use `--model MODEL_NAME` to select a compatible embedding model, then rebuild the whole store. Documents and questions must use the same embedding model. Updating a model under the same tag also requires rebuilding the store.

The input is a text-based PDF. Scanned/image-only pages need OCR first; this loader reports pages with no extractable text rather than silently skipping them. It does not perform OCR. Multi-column documents and complex tables may require a more specialized layout extractor.

Word limits are a simple learning heuristic, not exact token counting. The embedding API uses `truncate=False`, so unusually long tokens or a model input limit cause a visible error instead of silently losing content.

## Compare word and recursive character chunking

Word chunking remains the default so existing commands still work. All three lessons accept `--chunking words` or `--chunking recursive`. Lesson 1 also accepts `--chunking all`.

Install the updated dependencies, then build both stores using the same selected embedding model:

```powershell
.\rag\intermediate_rag\.venv\Scripts\python.exe -m pip install -r rag/intermediate_rag/requirements.txt
$env:RAG_EMBEDDING_MODEL = "embeddinggemma:latest"
.\rag\intermediate_rag\.venv\Scripts\python.exe rag/intermediate_rag/01_create_embeddings.py --chunking all
.\rag\intermediate_rag\.venv\Scripts\python.exe rag/intermediate_rag/02_retrieve.py --compare --question "How long does standard delivery take?" --top-k 2
```

| Setting | Word chunks | Recursive chunks |
| --- | --- | --- |
| Maximum size | 100 words | 600 characters |
| Overlap | 20 words | Target of 120 characters |
| Boundaries | Word count | Paragraph, line, space, individual character |
| Store | `embedding_store.json` | `embedding_store_recursive.json` |

The recursive strategy uses LangChain's `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", " ", ""]`. It keeps extracted line breaks instead of flattening them before splitting. PDF line breaks may be visual wraps rather than paragraph boundaries. This splitter is not a sentence parser or semantic chunker; long sentences can still split. Overlap is a target and can be smaller or absent when boundary units cannot be reused. Both strategies keep page numbers and never cross a PDF page.

To change recursive sizes and rebuild only that store:

```powershell
.\rag\intermediate_rag\.venv\Scripts\python.exe rag/intermediate_rag/01_create_embeddings.py --chunking recursive --chunk-size 800 --chunk-overlap 100
.\rag\intermediate_rag\.venv\Scripts\python.exe rag/intermediate_rag/02_retrieve.py --chunking recursive
.\rag\intermediate_rag\.venv\Scripts\python.exe rag/intermediate_rag/03_generate_answer.py --chunking recursive --question "How long does standard delivery take?"
```

`--chunk-size` and `--chunk-overlap` apply only to recursive chunks and count **characters, not tokens**. The existing Ollama `truncate=False` check still catches inputs that exceed the model's token limit.

Use just `--compare` in lesson 2 to show every registered chunking strategy, each with semantic retrieval and a word-count baseline. Without it, `--chunking` selects one strategy and only semantic results are shown. When `--compare` is set, it takes precedence over `--chunking`.

Comparison requires stores for all registered strategies; missing or stale stores produce an error before any embedding request. Build them with lesson 1's `--chunking all` (`both` remains a compatibility alias). Add future strategies to `CHUNKING_STRATEGIES` in `semantic_helpers.py` with a splitter and settings provider. Store filenames, CLI choices, building all stores, and comparison then include the strategy automatically, with no new comparison flag.

Comparison validates both stores against the current PDF and selected embedding model. It shows settings, indexed chunk counts, retrieved text, scores, PDF pages, and context character totals. Chunk IDs belong to their own store and do not correspond between strategies. The 100-word and 600-character defaults are starting points, not equivalent sizes or a controlled benchmark. Judge whether each result includes the complete answer and its conditions; higher cosine scores alone do not establish a winner. Rebuild both after changing the model or PDF. Updates under the same model tag still require manual rebuilding.

See the [official recursive splitter documentation](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter).

## Lesson 2: retrieve by meaning

```powershell
.\rag\intermediate_rag\.venv\Scripts\python.exe rag/intermediate_rag/02_retrieve.py --compare
.\rag\intermediate_rag\.venv\Scripts\python.exe rag/intermediate_rag/02_retrieve.py --question "How long does standard delivery take?" --top-k 2
```

[02_retrieve.py](02_retrieve.py) embeds the question using the model recorded in the store, ranks chunks by cosine similarity, and prints their PDF page numbers. `--compare` compares all registered chunking strategies and includes their word-count baselines.

Word counts rely on shared terms. Semantic embeddings can associate paraphrases with relevant content, but can still confuse related topics. The larger handbook includes overlapping subjects, so inspect the actual evidence instead of assuming a high score means the answer is present. Results from the previous four-paragraph text sample do not describe this new corpus.

Dense embeddings can have cosine scores from -1 to 1. Scores are not confidence probabilities, and word-count scores are not directly interchangeable with semantic scores. There is no automatic no-answer threshold here: nearest-neighbor retrieval returns candidates even for unrelated questions.

## Lesson 3: generate an answer with page references

```powershell
.\rag\intermediate_rag\.venv\Scripts\python.exe rag/intermediate_rag/03_generate_answer.py --question "How long does standard delivery take?" --prompt-only
.\rag\intermediate_rag\.venv\Scripts\python.exe rag/intermediate_rag/03_generate_answer.py --question "How long does standard delivery take?"
```

[03_generate_answer.py](03_generate_answer.py) prints the retrieved context and prompt, then calls Qwen. Context labels contain the PDF filename, chunk ID, and page number. The model is instructed to cite the supporting chunks and pages. Verify those citations against the PDF; the instruction does not guarantee that every generated answer is grounded or correctly cited.

`--prompt-only` skips answer generation but still calls the embedding model. Lesson 3's `--model` selects the generation model, while lesson 1's `--model` selects the embedding model. The generation settings use `think=False` for the local Qwen model; other models may need adjustments.

## Explore the handbook

| PDF pages | Topics |
| --- | --- |
| 1-10 | Refunds, returns, exchanges, damaged goods, gift returns |
| 11-20 | Delivery methods, tracking, missing parcels, customs |
| 21-27 | Payments, invoices, discounts, gift cards, store credit |
| 28-35 | Order changes, preorders, stock, bulk orders, prices |
| 36-41 | Product care, warranty claims, repairs, spare parts |
| 42-50 | Support, account access, privacy, reviews, holidays |

Try these questions and inspect the cited pages:

- What is the time limit for requesting a refund? (page 1)
- When should standard delivery arrive after dispatch? (page 11)
- Can I add a discount code after payment? (page 25)
- How long is the manufacturing-defect warranty? (page 38)
- When is customer support available? (page 42)

## Replace or regenerate the document

[ sample_document.pdf ](sample_document.pdf) is the active source. To practice with your own text-based PDF, replace that file and rerun lesson 1. The loader accepts other page counts too; 50 pages is the size of the included sample. A source-hash check rejects an outdated store after the PDF changes.

To regenerate the fictional 50-page sample from its editable content:

```powershell
.\rag\intermediate_rag\.venv\Scripts\python.exe -m pip install -r rag/intermediate_rag/requirements-build.txt
.\rag\intermediate_rag\.venv\Scripts\python.exe rag/intermediate_rag/create_sample_pdf.py
.\rag\intermediate_rag\.venv\Scripts\python.exe rag/intermediate_rag/01_create_embeddings.py
```

[create_sample_pdf.py](create_sample_pdf.py) contains the 50 topic records and the ReportLab layout. Its build dependency is optional when you only want to read an existing PDF. Regeneration replaces the PDF, so do not run it after substituting your own document unless you intend to restore the sample.

## Check the code

```powershell
.\rag\intermediate_rag\.venv\Scripts\python.exe -m unittest discover -s rag/intermediate_rag -p "test_*.py"
```

The offline checks cover PDF page extraction, overlapping chunk boundaries, page references, scanned-page handling, vector validation, stale stores, and retrieval mechanics. They do not measure real-model answer quality. The included-document check expects the supplied 50-page sample; update it if you intentionally substitute another document.

The code uses [pypdf](https://pypdf.readthedocs.io/en/stable/user/extract-text.html) for extraction and the [Ollama embeddings API](https://docs.ollama.com/api/embed) for vectors.
