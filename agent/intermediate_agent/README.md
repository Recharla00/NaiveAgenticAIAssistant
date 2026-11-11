# Intermediate agent with LangChain

## Step 8.3: approval before changing data

The agent now checks every registered tool call before it runs. Calculator and
knowledge-search tools are marked read-only. Any other tool needs approval;
without an approval handler, the agent denies it. The opt-in `save_demo_note`
tool appends to `agent/intermediate_agent/data/approved_demo_notes.txt` only
after you see the proposed arguments and type `y` or `yes`. Press Enter or type
`n` to deny it. A denied call does not change the file or the saved session.

```powershell
.\.venv\Scripts\python.exe agent/intermediate_agent/05_session_agent.py --session-id approval-demo --question "Save demo note: Buy milk" --enable-demo-write --verbose
$env:PYTHONPATH = "agent/intermediate_agent"
.\.venv\Scripts\python.exe -m unittest discover -s agent/intermediate_agent -p test_approvals.py -v
```

The demo write tool is available only with `--enable-demo-write` in manual
session mode. Each proposed write asks again; approval is never remembered.

## Step 8.2: question time budget

`--max-seconds` sets a 180-second default budget for one question. The agent
checks the deadline before and after model calls, tool calls, and the required
technical document search. If the budget is used up, it gives a clear timeout
reply and does not save that failed question. The CLI also caps Ollama's HTTP
timeout to the chosen budget.

This is a cooperative deadline: a Python tool already inside one blocking call
can finish after the deadline, then the agent stops before continuing or saving.
The tests simulate slow model, tool, and search calls without Ollama.

```powershell
.\.venv\Scripts\python.exe agent/intermediate_agent/05_session_agent.py --auto-route --session-id demo --question "Add 2 and 3" --max-seconds 30 --verbose
$env:PYTHONPATH = "agent/intermediate_agent"
.\.venv\Scripts\python.exe -m unittest discover -s agent/intermediate_agent -p test_execution_time.py -v
```

## Step 8.1: model and tool call limits

Each question has a default limit of 6 model calls and 8 tool calls. Set
`--max-model-calls` and `--max-tool-calls` to change them. A technical question's
required document search uses one tool call from that budget. LangChain blocks
calls beyond the limit; the agent gives a clear reply and keeps the last saved
session unchanged. `--max-steps` remains a separate graph step safeguard and
defaults to 50 so it does not interrupt normal correction attempts.

```powershell
.\.venv\Scripts\python.exe agent/intermediate_agent/05_session_agent.py --auto-route --session-id demo --question "Add 2 and 3" --max-model-calls 2 --max-tool-calls 1 --verbose
$env:PYTHONPATH = "agent/intermediate_agent"
.\.venv\Scripts\python.exe -m unittest discover -s agent/intermediate_agent -p test_execution_limits.py -v
```

## Step 7.3: clear final failures and safe sessions

If the model times out, Ollama cannot be reached, or the agent reaches its step
limit, the command prints a short explanation. Exhausted tool retries and a
failed knowledge search also return clear answers. These failed questions are
not saved, so the last successful conversation remains available under the
same session ID. `--verbose` logs the failure and that the session was left
unchanged. An unexpected programming error still raises or prints an error for
debugging.

```powershell
$env:PYTHONPATH = "agent/intermediate_agent"
.\.venv\Scripts\python.exe -m unittest discover -s agent/intermediate_agent -p test_tool_errors.py -v
```

## Step 7.2: correct failed tool calls

After a tool error, the model sees the error and can fix its arguments or use
another available tool. Three failed tool results in one question stop further
model calls and return a clear answer. This permits an initial attempt and up
to two corrections. The limit starts over for the next question, even when the
same session ID is reused. `--verbose` shows recovery attempts and the limit.

```powershell
$env:PYTHONPATH = "agent/intermediate_agent"
.\.venv\Scripts\python.exe -m unittest discover -s agent/intermediate_agent -p test_tool_errors.py -v
```

## Step 7.1: readable tool errors

If a tool raises an exception, the agent receives an error tool result instead
of stopping immediately. Invalid calculator inputs already produce a LangChain
validation error result. A timed-out tool reports that it timed out. If the
technical route cannot search the knowledge base, the user gets a clear reply.
Use `--verbose` to see failures in the terminal.

```powershell
$env:PYTHONPATH = "agent/intermediate_agent"
.\.venv\Scripts\python.exe -m unittest discover -s agent/intermediate_agent -p test_tool_errors.py -v
```

## Follow the agent's flow in the terminal

Add `--verbose` to `05_session_agent.py` to see when history is loaded, which
route was chosen, whether document search ran, how many results remained
relevant, tool requests/results, model duration, and when the session was
saved. `--debug` includes those logs plus short message previews and shows a
full traceback on failure. Logs go to the terminal; they are not persisted.

```powershell
.\.venv\Scripts\python.exe agent/intermediate_agent/05_session_agent.py --auto-route --session-id demo --question "Add 2 and 3" --verbose
.\.venv\Scripts\python.exe agent/intermediate_agent/05_session_agent.py --auto-route --session-id demo --question "How do I configure the API?" --debug
```

The manual `06_try_route.py` command accepts the same flags. These readable
logs help inspect the lessons; persistent structured tracing is planned for
Step 9.

## Step 6.3: automatic routing in a saved session

Add `--auto-route` to the Step 5 session command. Each new question is labeled
using the Step 6.1 rules, then the matching Step 6.2 path runs. The command
prints the route and reason. All messages, including a technical search result,
are saved under the same session ID for later questions.

```powershell
.\.venv\Scripts\python.exe agent/intermediate_agent/05_session_agent.py --auto-route --session-id route-demo --question "I was charged twice."
.\.venv\Scripts\python.exe agent/intermediate_agent/05_session_agent.py --auto-route --session-id route-demo --question "How do I configure the API?"
.\.venv\Scripts\python.exe agent/intermediate_agent/05_session_agent.py --auto-route --session-id route-demo --question "What did we discuss earlier?"
```

Use the same `--session-dir` on every run if you choose a nondefault directory.
The auto route chooses its own tools, so it cannot be combined with
`--enable-tools` or `--enable-retrieval`. Routing uses the current question's
words; an unclear follow-up uses the general path but still receives the saved
conversation. A small local model may misread earlier messages even when the
stored history is correct. The route decision and stored order are verifiable;
answer quality is a separate evaluation concern.

Offline verification includes a billing-to-technical-to-general session, source
evidence retention, session isolation, and saving the no-documentation reply:

```powershell
$env:PYTHONPATH = "agent/intermediate_agent"
.\.venv\Scripts\python.exe -m unittest agent.intermediate_agent.test_routed_session
```

## Step 6.2: run each conditional path

`build_routed_agent(settings, route)` configures one agent for a selected path:

- `billing`: billing support instructions, policy search, and calculators;
- `technical`: documentation search and a technical evidence prompt;
- `general`: normal answers and calculators, without document search.

The technical path runs `search_internal_knowledge_base` before asking the
model to answer. If none of the retrieved chunks share meaningful words with
the question, it reports that no relevant internal documentation was found.
This conservative check avoids treating an unrelated vector result as an API
guide. It can also miss documents that use different words for the same idea.

Try each path directly; the billing and technical examples need the Step 3 index and local
Ollama server:

```powershell
.\.venv\Scripts\python.exe agent/intermediate_agent/06_try_route.py --route billing --question "What do I do about an incorrect charge?"
.\.venv\Scripts\python.exe agent/intermediate_agent/06_try_route.py --route technical --question "How do I configure the API?"
.\.venv\Scripts\python.exe agent/intermediate_agent/06_try_route.py --route general --question "Add 2 and 3."
```

Run the offline path checks:

```powershell
$env:PYTHONPATH = "agent/intermediate_agent"
.\.venv\Scripts\python.exe -m unittest agent.intermediate_agent.test_route_paths
```

## Step 6.1: preview conditional routing

The local `route_question()` function labels a question as `billing`,
`technical`, or `general`. It matches whole topic words and returns a reason
and the matching terms. A question with both billing and technical terms uses
the general route because the topic is unclear. This is a simple rule-based
first version; it does not call Ollama, search documents, or change agent
behavior yet.

```powershell
.\.venv\Scripts\python.exe agent/intermediate_agent/06_route_question.py "Why was I charged twice?"
.\.venv\Scripts\python.exe agent/intermediate_agent/06_route_question.py "How do I configure the API?"
.\.venv\Scripts\python.exe agent/intermediate_agent/06_route_question.py "Hello"
```

Verify the routing rules offline:

```powershell
$env:PYTHONPATH = "agent/intermediate_agent"
.\.venv\Scripts\python.exe -m unittest agent.intermediate_agent.test_routing
```

## Step 5.7–5.9: bounded history and a session CLI

Run these commands from the repository root after starting Ollama. Use the same
session ID to ask a follow-up in a later process:

```powershell
.\.venv\Scripts\python.exe agent/intermediate_agent/05_session_agent.py --session-id demo --question "Remember my order number is 42."
.\.venv\Scripts\python.exe agent/intermediate_agent/05_session_agent.py --session-id demo --question "What was my order number?"
```

Use `--enable-tools` for calculator tools or `--enable-retrieval` for both
knowledge-base search and calculator tools. Retrieval requires a Step 3 index.
`--session-dir` selects another storage directory. The default history keeps
the most recent 20 user turns and expires after 30 days of inactivity;
`--max-turns` and `--max-age-days` adjust those limits. A turn keeps its tool
calls and results together. Damaged or expired session files are renamed with
`.corrupt.` or `.expired.` in the same directory, a warning names the retained
file, and the session starts fresh. Existing files remain recoverable.

The CLI prints a completed answer and saves its history. It uses `invoke()`, so
text appears after the turn completes. The offline test suite verifies a
follow-up across separate Python processes, session isolation, recovery from
damaged/expired files, and tool history retention:

```powershell
$env:PYTHONPATH = "agent/intermediate_agent"
.\.venv\Scripts\python.exe -m unittest discover -s agent/intermediate_agent -p "test_*.py"
```

## Step 5.4–5.6: restore history and run the agent

`run_session_agent()` loads a session by ID, sends its earlier messages plus the
new question to the LangChain agent, then saves the completed message history.
LangChain message serialization preserves native calculator/search tool calls,
tool results, and retrieved source evidence for the next run.

```python
from agent_app.agent import build_agent, run_session_agent
from agent_app.config import Settings

agent = build_agent(Settings(), enable_retrieval=True)
run_session_agent(agent, "What is the refund policy?", "demo-session")

# Run later, even from a new Python process, using the same session ID.
agent = build_agent(Settings(), enable_retrieval=True)
run_session_agent(agent, "How many days was that?", "demo-session")
```

For retrieval, ingest documents first and keep local Ollama running. A failed or
incomplete agent turn leaves the previously saved session unchanged. The runner
currently uses `invoke()`; CLI session options and streaming integration are
later substeps.

## Step 5.2–5.3: local session storage and message saving

`SessionStore` saves each conversation as a JSON file in the ignored
`agent/intermediate_agent/data/sessions/` directory. A session ID identifies
the conversation; its filename is a hash so the ID cannot change the storage
path. `add_message()` saves one `user` or `assistant` message at a time.

```python
from agent_app.memory import SessionStore

store = SessionStore()
store.add_message("demo-session", "user", "My order number is 42")
store.add_message("demo-session", "assistant", "I can help with order 42")

# This also works in a later Python process:
history = SessionStore().load("demo-session")
print([(message.role, message.content) for message in history.messages])
```

Run the focused offline verification from the repository root:

```powershell
$env:PYTHONPATH = "agent/intermediate_agent"
.\.venv\Scripts\python.exe -m unittest agent.intermediate_agent.test_memory
```

This storage API is ready for the later agent integration substep. For now,
the numbered agent scripts do not automatically read or write these sessions.
Concurrent writers to the same session should be serialized by the caller.

## Step 3: document ingestion and vector storage

This lesson prepares a persistent **local Chroma database**. It uses the existing
Ollama `embeddinggemma:latest` embedding model, separately from the chat model.
No chat model or agent is called during ingestion. No cloud database account is needed.

Install the additional dependencies in the same dedicated environment:

```powershell
agent/intermediate_agent/.venv/Scripts/python.exe -m pip install -r agent/intermediate_agent/requirements.txt
ollama list
```

If missing, install the embedding model with `ollama pull embeddinggemma:latest`.
The default input is `agent/intermediate_agent/knowledge_base/`, which includes
a short fictional Markdown policy. Add your own UTF-8 Markdown or text-based PDFs
there, or choose a file/folder using `--input`.

```powershell
# Inspect extraction and chunking without calling Ollama or writing vectors.
agent/intermediate_agent/.venv/Scripts/python.exe agent/intermediate_agent/03_ingest_documents.py --preview

# Generate embeddings and persist the database.
agent/intermediate_agent/.venv/Scripts/python.exe agent/intermediate_agent/03_ingest_documents.py

# Preview the existing PDF from the intermediate RAG lessons.
agent/intermediate_agent/.venv/Scripts/python.exe agent/intermediate_agent/03_ingest_documents.py --input rag/intermediate_rag/sample_document.pdf --preview
```

Flow: PDF/Markdown -> text -> overlapping chunks -> Ollama embeddings -> Chroma.
Chunks retain source paths relative to the input folder, one-based PDF pages,
text offsets within each page/document, and stable chunk IDs. The default chunk
size is 600 **characters**, with a target overlap of 100. Adjust with
`--chunk-size`, `--chunk-overlap`, and `--batch-size` (default 8).

Generated files live under `agent/intermediate_agent/data/` (ignored by Git):

- `chroma/`: the persistent vector database, document text, and metadata.
- `active_index.json`: active collection, embedding tag/digest, source hashes,
  chunk settings, and chunk IDs for the retrieval lesson.

Running the same input/configuration again reuses stored vectors. Changed inputs
or embedding models produce a new collection; only a fully stored collection is
made active. A failed run leaves the previous active index intact. Old collections
remain on disk; this lesson does not implement database cleanup. Use one ingestion
process per data folder at a time. `--data-dir` lets you keep separate databases.
Each input selection becomes the complete active corpus, not an append operation.

PDF extraction handles existing text, not scanned images/OCR or complex table
layout. Empty pages are reported and skipped; PDFs with no extractable text fail
clearly. Markdown is treated as text, without following links or rendering it.
Keep ingestion folders limited to the documents you intend to index.

Step 4 below combines vector and keyword search and exposes
`search_internal_knowledge_base` to the agent.

## Step 4: hybrid internal knowledge search

Step 4 searches the active Step 3 snapshot with both retrieval methods:

- vector candidates from the persistent Chroma collection;
- keyword candidates from the stored chunk text, including exact phrase matches;
- reciprocal-rank fusion to combine both candidate lists deterministically.

Every result includes the chunk text, stable `chunk_id`, relative `source`,
optional PDF `page`, text offset, retrieval method, rank, and fused score. The
search is available to the LangChain agent as the opt-in
`search_internal_knowledge_base` tool; calculator tools remain registered too.

```powershell
# Search an existing active index and print JSON evidence.
agent/intermediate_agent/.venv/Scripts/python.exe agent/intermediate_agent/04_search_knowledge_base.py "refund within thirty days"

# Enable retrieval in an agent from Python:
from agent_app.agent import build_agent
agent = build_agent(Settings(), enable_retrieval=True)
```

Run ingestion first if `active_index.json` does not exist. Search does not modify
the index and fails clearly when no active snapshot is available.

References: [Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma),
[Ollama embeddings](https://docs.langchain.com/oss/python/integrations/embeddings/ollama),
[PDF extraction](https://pypdf.readthedocs.io/en/stable/user/extract-text.html).

## Step 2: native tool calling

```powershell
agent/intermediate_agent/.venv/Scripts/python.exe agent/intermediate_agent/02_native_tools.py
agent/intermediate_agent/.venv/Scripts/python.exe agent/intermediate_agent/02_native_tools.py --question "Add 20 and 5, then multiply the result by 8."
agent/intermediate_agent/.venv/Scripts/python.exe agent/intermediate_agent/02_native_tools.py --question "Hello!"
```

`agent_app/tools/calculator.py` defines `add_numbers` and `multiply_numbers` with
LangChain's `@tool` decorator. Their descriptions and validated number schemas
are sent to the model. `build_agent(settings, enable_tools=True)` registers them
with `create_agent`. LangChain executes native tool calls, creates tool-result
messages, and calls the model again. Python does not parse a custom action JSON
or manually dispatch these tools.

The default question should produce this sequence:

```text
Model requests multiply_numbers(a=125, b=8) -> tool returns 1000
Model returns a final answer
```

For the chained calculation exercise, the intended sequence is:

```text
Model requests add_numbers(a=20, b=5) -> tool returns 25
Model requests multiply_numbers(a=25, b=8) -> tool returns 200
Model returns a final answer
```

The lesson streams model text and prints completed tool requests/results with
matching call IDs. It reports wall time, model-call count, tool-result count,
and available Ollama timings summed across model calls. Tool-call JSON itself
is displayed once complete. `--debug` prints the complete message state.
`--max-steps` defaults to 12 LangGraph steps, including model and tool steps;
it is not a tool-call count. Exhausting it stops the run with an error.

The small model can still choose incorrectly or skip an operation. Inspect the
actual tool trace, not just the final number. Step 1 remains a no-tool setup
check. Both lessons retain the smaller model, disabled thinking, and 30-minute
keep-alive. No additional dependencies are needed for step 2.

In live testing, the small model sometimes submitted both calls together for
the chained calculation, precomputing the intermediate result. The prompt's
one-call-at-a-time instruction is not enforced by this lesson. This demonstrates
native execution, not guaranteed sequential planning. Prefer the single-operation
example when first learning the message flow; robust planning controls can be
added with execution-control and recovery lessons.

Run offline schema, native execution, and streaming checks:

```powershell
agent/intermediate_agent/.venv/Scripts/python.exe -m unittest discover -s agent/intermediate_agent -p "test_*.py"
```

Reference: [LangChain tools](https://docs.langchain.com/oss/python/langchain/tools).

## Step 1: project setup and a working local model

The first lesson runs a LangChain agent with no tools. It confirms the framework,
Ollama adapter, and model connection before adding native tool calling in step 2.
No API key is needed for local Ollama.

```text
intermediate_agent/
  01_check_setup.py        Command-line lesson
  requirements.txt        Direct dependencies
  requirements-lock.txt   Exact installed dependency versions
  agent_app/
    config.py             Model and connection settings
    model.py              ChatOllama adapter
    prompts.py            Model instructions
    agent.py              LangChain create_agent setup
    tools/__init__.py     Tool registry (enabled in step 2)
    tools/calculator.py  Native calculator tools
```

Use Python 3.11 or newer (tested with 3.13). Run these commands from the project
root in PowerShell. A dedicated environment keeps these dependencies separate
from the existing RAG lessons:

```powershell
python -m venv agent/intermediate_agent/.venv
agent/intermediate_agent/.venv/Scripts/python.exe -m pip install -r agent/intermediate_agent/requirements-lock.txt
ollama list
agent/intermediate_agent/.venv/Scripts/python.exe agent/intermediate_agent/01_check_setup.py
```

Ollama must be running and `qwen3:1.7b` must be installed. If missing, download it
with `ollama pull qwen3:1.7b`. If the server is unavailable, start the Ollama app
or run `ollama serve` in another terminal. Use `--model NAME` for another installed
model; model capabilities and behavior may differ.

The verified environment uses LangChain 1.4.7 and langchain-ollama 1.1.0.
`requirements-lock.txt` records all installed versions for repeatable setup;
`requirements.txt` lists the direct dependencies with allowed version ranges.

```powershell
agent/intermediate_agent/.venv/Scripts/python.exe agent/intermediate_agent/01_check_setup.py --question "Explain tool calling briefly." --debug
```

`--debug` prints returned messages and metadata, and shows a traceback on failure.
It does not enable remote tracing. This lesson configures no LangSmith tracing.

## Response speed and memory

The default is the smaller `qwen3:1.7b` model, with thinking disabled. It advertises
tool support, but complex tasks may be less reliable than with a larger model.
The lesson streams text as it arrives. Streaming reduces the visible wait; it
does not itself increase generation speed.

`keep_alive="30m"` retains the selected model in RAM for 30 minutes after a call,
avoiding repeated loading while you work. It deliberately keeps some RAM occupied.
To release RAM immediately, unload a model:

```powershell
ollama stop qwen3:1.7b
ollama stop qwen3:1.7b
```

Unloading is sufficient to release model RAM. Removing a downloaded model frees
disk space; it is not required for this change. The basic lessons still default
to the older model. Use `--model qwen3:1.7b` here to compare if desired.

Each successful run prints first-text latency, agent-call time, and available
Ollama loading, prompt-processing, generation, and tokens/second metrics.
First-text and agent-call times exclude Python imports and agent construction;
the total-lesson timer includes those (but not interpreter startup). These are
single-model-call metrics for step 1, not aggregated multi-tool timings.

Observed on this CPU-only laptop using the same two-sentence question:

| Model | First text | Agent call | Loading | Generation speed |
| --- | --- | --- | --- | --- |
| `qwen3:1.7b` | 37.39 s | 49.63 s | 30.95 s | 3.3 tokens/s |
| `qwen3:1.7b` | 6.92 s | 10.18 s | 4.66 s | 8.5 tokens/s |

These are single runs, not a controlled benchmark. Answers had different lengths
(41 versus 28 generated tokens), and download activity overlapped the old-model
run. Warm calls and longer prompts will have different timings.

To check streaming without calling Ollama:

```powershell
agent/intermediate_agent/.venv/Scripts/python.exe -m unittest discover -s agent/intermediate_agent -p test_streaming.py
```

References: [smaller model](https://ollama.com/library/qwen3:1.7b),
[LangChain streaming](https://docs.langchain.com/oss/python/langchain/streaming),
[Ollama memory controls](https://docs.ollama.com/faq).

The flow is:

```text
Question -> LangChain agent -> ChatOllama -> local Ollama -> AI message -> printed answer
```

The basic agent manually sent HTTP requests and ran a Python loop. Here,
`create_agent` builds the framework-managed execution, and `ChatOllama` handles
the provider connection. With tools disabled in step 1 this is only one model
response, not yet a tool-using workflow. Each run starts fresh; persistent memory,
RAG, routing, recovery, approvals, tracing, and evaluations come in later steps.

## Planned lessons

1. Project structure, LangChain setup, and local model connection (this lesson).
2. Native tool calling with calculator tools.
3. Document ingestion, embeddings, and vector storage.
4. Hybrid retrieval through `search_internal_knowledge_base`.
5. Persistent conversation memory by session ID.
6. Conditional routing.
7. Graceful error recovery.
8. Execution limits and approvals.
9. Structured logs and tracing.
10. Evaluation and end-to-end verification.

References: [LangChain quickstart](https://docs.langchain.com/oss/python/langchain/quickstart),
[ChatOllama integration](https://docs.langchain.com/oss/python/integrations/chat/ollama).
