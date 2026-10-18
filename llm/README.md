# LLM evaluations in baby steps

An **eval** is a repeatable check of a model's response. We define the input and expected result before asking the model, then compare its actual answer with that expectation.

```text
Test case -> Prompt -> Local Ollama model -> Actual answer -> Python grader -> Report
```

This folder evaluates generated answers. The existing `rag/09_evaluate_retrieval.py` evaluates retrieved evidence. These are different parts of the system: good retrieval alone does not guarantee a good answer.

## Start with one case

Use the same Python and Ollama setup as the RAG lessons. No extra Python packages or API key are required. The default model is `qwen3.5:4b`.

Run from the project root:

```powershell
python llm/01_evaluate_llm.py --dry-run
python llm/01_evaluate_llm.py --case arithmetic
```

The dry run prints the dataset without contacting Ollama. The second command sends one prompt, checks its answer, and saves a report. Ollama must be running for model calls; open its app or use `ollama serve` in a separate terminal if it is stopped.

Run all five cases:

```powershell
python llm/01_evaluate_llm.py
```

Use `--model MODEL_NAME` to select another installed model. Requests run one at a time. CPU generation, especially the first request that loads the model, can take a while. Settings include `think=False`, which was chosen for the local Qwen model; other models may need adjustments.

## Understand the dataset

Edit [eval_cases.json](eval_cases.json) to add examples.

| Case | What it checks | Expected answer |
| --- | --- | --- |
| `arithmetic` | Basic correctness | `5` |
| `sentiment` | Follow a classification instruction | `positive` |
| `context_answer` | Read a fact from supplied context | `30` |
| `missing_information` | Avoid inventing an absent fact | `UNKNOWN` |
| `json_extraction` | Produce the requested JSON fields and values | Name Asha, city Pune |

Each case has a unique `id`, a `prompt`, a `check`, and an `expected` result. Only the prompt goes to the model. Expected answers are used by the Python grader, not added to the request.

## Read the grading rules

- **`exact`:** compare the whole answer, ignoring letter case and repeated whitespace. `5` passes; `50`, `5.`, and `The answer is 5` fail. This intentionally checks the requested format as well as content.
- **`json`:** require valid JSON and compare keys, values, and types. Key order is ignored. Extra fields, Markdown fences, duplicate fields, and non-standard JSON constants fail.

These are deterministic checks, not an LLM judge or a measure of general intelligence. They suit short, constrained answers. Open-ended summaries and explanations need a different rubric and human review. A model can give a useful answer that fails a strict format check, so read the actual response when investigating a failure.

## Inspect the report

Each live run creates a timestamped JSON file in `llm/results/`. These local reports are ignored by Git. Each report records the selected model and settings, prompts, expected and actual answers, grading reasons, and elapsed request times.

- **Pass:** the answer met its declared rule.
- **Fail:** an answer was generated but did not meet the rule.
- **Error:** the request failed, the output was empty, or generation hit its output limit.

The pass rate is passes divided by all selected cases, including request errors in the denominator. Errors are also counted separately; do not interpret an unavailable server as poor model reasoning. The script exits with code 1 if any case fails or errors, and 0 if all pass.

Timings include local request overhead and model loading when needed. They are not isolated model-speed benchmarks. A five-case pass rate is only a practice result, and even temperature 0 does not promise identical answers across runs or environments.

## Practice

1. Run only `arithmetic` and inspect its saved answer.
2. Run all cases and read each failure rather than only the score.
3. Add another factual or formatting case with a unique ID.
4. Run the same dataset again and compare reports.
5. Later, add a rubric for open-ended answers and compare it with human judgments.

Check the graders without Ollama:

```powershell
python -m unittest discover -s llm -p "test_*.py"
```

Ollama API reference: [Generate a response](https://docs.ollama.com/api/generate).
