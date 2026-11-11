# Basic agent: standalone and RAG lessons

## Step 7: let the model choose the tool

Run from the project root:

```powershell
python agent/basic_agent/07_run_agent.py --question "What is 125 times 8?"
python agent/basic_agent/07_run_agent.py --question "What is the refund policy?"
python agent/basic_agent/07_run_agent.py --question "Hello!"
```

One loop exposes addition, multiplication, and `search_policy` to the model.
The model chooses a tool or answers directly. Policy files are read only when
the model requests a search. There is no `--mode` option; `--question` is required.
Options: `--model NAME`, `--max-steps 6`, and `--debug` for full model logs.
History lasts for one question; there is no saved conversation memory.
Lessons 5 and 6 remain available as individual learning steps.

## Step 6: agent with RAG

```powershell
python agent/basic_agent/06_agent_with_rag.py --question "What is the refund policy?"
python agent/basic_agent/06_agent_with_rag.py --question "How long does delivery take?" --debug
```

This extends the same model connection and tool dispatcher with
`search_policy(query)`. The agent chooses when to retrieve, receives chunk IDs,
source names, text, and similarity scores, then answers with chunk references.
Arithmetic tools remain available. Step 5 still runs standalone without policy files.
Step 7 makes all tools available automatically in one command.

The search tool reads `rag/basic_rag/chunk_store.json` and reuses retrieval from
`rag/basic_rag/07_build_prompt.py`. If the store is missing or you edit
`rag/basic_rag/sample_document.txt`, rebuild it first:

```powershell
python rag/basic_rag/04_store_vectors.py
```

Search removes common query words and returns up to two positive-scoring chunks.
It uses word overlap, so synonyms can be missed and unrelated chunks can match.
Scores are not confidence. The prompt asks the model to treat documents as data,
cite supporting chunks, and admit missing information; these instructions do not
guarantee factual answers or correct citations. Inspect the printed tool results.

Run all offline checks:

```powershell
python -m unittest discover -s agent/basic_agent -p "test_*.py"
```

Run these lessons from the project root. They use Python's standard library.
Steps 1 and 3–7 need local Ollama running with `qwen3:1.7b`.
Use `--model NAME` to select another installed model; its behavior may differ.

| Step | File | What happens |
| --- | --- | --- |
| 1 | `01_call_llm.py` | Send a question and print the answer. |
| 2 | `02_create_tools.py` | Call Python tools manually. |
| 3 | `03_choose_action.py` | Ask the model to choose a tool or final answer; execute nothing. |
| 4 | `04_execute_tool.py` | Execute one tool, send its result back, and show the next action. |
| 5 | `05_agent_loop.py` | Repeat decisions and tool calls until a final answer or step limit. |
| 6 | `06_agent_with_rag.py` | Add policy retrieval to the available tools. |
| 7 | `07_run_agent.py` | Let the model choose arithmetic, policy search, or a direct answer. |

```powershell
python agent/basic_agent/03_choose_action.py --question "What is 125 times 8?"
python agent/basic_agent/04_execute_tool.py --question "What is 125 times 8?"
python agent/basic_agent/05_agent_loop.py --question "Add 20 and 5, then multiply the result by 8."
python agent/basic_agent/05_agent_loop.py --question "What is an AI agent?"
```

The multi-step example should call addition (25), multiplication (200), then
return a final answer. Model output can vary. Each model call counts toward
`--max-steps`, including the final answer; the default is 6.

The model requests actions using JSON. Python validates them and dispatches
only registered tools. This is a simple custom action protocol using Ollama's
[structured outputs](https://docs.ollama.com/capabilities/structured-outputs),
not its native tool-calling API. Tool definitions live in `standalone_tools.py`;
connection and validation code live in `agent_helpers.py`.

Add `--debug` to steps 3–7 to print the full request and response. Normal output
already shows selected actions and tool results. History lasts only for the
current question. Malformed actions and connection errors stop the lesson with
an error. Only addition and multiplication are supported tools at this stage.
Step 6 adds RAG document search; step 7 exposes all tools without mode selection.

Run the offline validation and loop checks without Ollama:

```powershell
python -m unittest discover -s agent/basic_agent -p test_agent.py
```
