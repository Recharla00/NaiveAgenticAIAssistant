"""LLM evaluation baby step 1: prompt -> answer -> check -> report.

Run from the project root with Ollama running:
    python llm/01_evaluate_llm.py

Start small, or inspect the cases without calling a model:
    python llm/01_evaluate_llm.py --case arithmetic
    python llm/01_evaluate_llm.py --dry-run

This evaluates actual model answers using simple Python checks. It does not
use another LLM as a judge. See README.md in this folder for grading limits.
Only the prompt is sent to the model; expected answers stay with the grader.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


def normalize(text):
    """Ignore capitalization and repeated whitespace, but preserve punctuation."""
    return " ".join(text.casefold().split())


def grade_answer(answer, case):
    """Return (passed, explanation) using the case's declared grading rule."""
    if case["check"] == "exact":
        passed = normalize(answer) == normalize(case["expected"])
        return passed, "Compare the whole answer after normalizing case and whitespace."

    if case["check"] == "json":
        try:
            # Reject duplicate fields and non-standard values such as NaN.
            def unique_fields(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f"Duplicate JSON field: {key}")
                    result[key] = value
                return result

            def reject_constant(value):
                raise ValueError(f"Invalid JSON constant: {value}")

            actual = json.loads(answer, object_pairs_hook=unique_fields,
                                parse_constant=reject_constant)
        except ValueError as error:
            return False, f"Expected valid JSON only: {error}"
        # Canonical JSON ignores key order and preserves distinctions like true vs 1.
        passed = json.dumps(actual, sort_keys=True) == json.dumps(case["expected"], sort_keys=True)
        return passed, "Compare JSON keys, values, and types; extra fields also fail."

    raise ValueError(f"Unknown grading rule: {case['check']}")


def generate_answer(prompt, model):
    """Call the same local Ollama service used by the RAG lessons."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 256},
    }
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    if result.get("error"):
        raise ValueError(str(result["error"]))
    if result.get("done_reason") == "length":
        raise ValueError("Generation reached its output limit; answer is incomplete.")
    answer = result.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Ollama returned no answer text.")
    return answer.strip()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate local LLM answers with simple checks.")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--case", help="Run only the case with this ID.")
    parser.add_argument("--dry-run", action="store_true", help="Print cases without calling Ollama.")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent

    # 1. Load and validate our small dataset BEFORE sending requests.
    cases = json.loads((folder / "eval_cases.json").read_text(encoding="utf-8"))
    seen_ids = set()
    for case in cases:
        if not case.get("id") or case["id"] in seen_ids:
            parser.error("Each evaluation case needs a unique, non-empty ID.")
        seen_ids.add(case["id"])
        if not isinstance(case.get("prompt"), str) or not case["prompt"].strip():
            parser.error(f"Case {case['id']} needs a non-empty prompt.")
        if case.get("check") not in ("exact", "json") or "expected" not in case:
            parser.error(f"Case {case['id']} needs a supported check and an expected answer.")
        if case["check"] == "exact" and not isinstance(case["expected"], str):
            parser.error(f"Case {case['id']}: exact checks require a string expected answer.")
    if args.case:
        cases = [case for case in cases if case["id"] == args.case]
    if not cases:
        parser.error("No cases selected. Check --case against the IDs in eval_cases.json.")
    if args.dry_run:
        print(json.dumps(cases, indent=2, ensure_ascii=False))
        print("\nDry run: no model calls, scores, or report file.")
        return 0

    # 2. Generate an answer for each prompt, then grade the answer independently.
    started_at = datetime.now(timezone.utc).isoformat()
    results = []
    for case in cases:
        print(f"\nRunning {case['id']} with {args.model}...", flush=True)
        start = time.perf_counter()
        record = {**case, "answer": None, "status": "error", "reason": ""}
        try:
            record["answer"] = generate_answer(case["prompt"], args.model)
            passed, record["reason"] = grade_answer(record["answer"], case)
            record["status"] = "pass" if passed else "fail"
        except urllib.error.HTTPError as error:
            record["reason"] = f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}"
        except (urllib.error.URLError, OSError, ValueError) as error:
            record["reason"] = str(error)
        record["latency_seconds"] = round(time.perf_counter() - start, 3)
        results.append(record)
        print(f"  {record['status'].upper()} | Answer: {record['answer']!r}")
        print(f"  Expected: {case['expected']!r}")
        print(f"  {record['reason']}", flush=True)

    # 3. Count passes, grading failures, and request errors separately.
    summary = {
        "total": len(results),
        "passed": sum(item["status"] == "pass" for item in results),
        "failed": sum(item["status"] == "fail" for item in results),
        "errors": sum(item["status"] == "error" for item in results),
    }
    summary["pass_rate"] = summary["passed"] / summary["total"]

    # 4. Save a report with prompts, expectations, actual answers, and timings.
    report = {"started_at_utc": started_at, "model": args.model,
              "settings": {"temperature": 0, "num_predict": 256, "think": False},
              "summary": summary, "results": results}
    output_folder = folder / "results"
    output_folder.mkdir(exist_ok=True)
    filename = datetime.now(timezone.utc).strftime("eval_%Y%m%d_%H%M%S_%f.json")
    report_path = output_folder / filename
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPassed: {summary['passed']} | Failed: {summary['failed']} | Errors: {summary['errors']}")
    print(f"Pass rate across all selected cases: {summary['pass_rate']:.0%}")
    print(f"Report: {report_path}")
    if summary["errors"]:
        print("For connection/model errors, check Ollama is running and use `ollama list`.")
    print("This measures these cases and rules only; inspect the answers yourself.")
    return 1 if summary["failed"] or summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
