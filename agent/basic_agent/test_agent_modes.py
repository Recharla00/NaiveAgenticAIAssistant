"""Check automatic tool selection, evidence flow, and CLI validation offline."""

import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("mode_agent", Path(__file__).with_name("07_run_agent.py"))
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


class ModeTests(unittest.TestCase):
    def test_arithmetic_does_not_require_search(self):
        actions = [
            {"action": "tool", "tool": "multiply_numbers", "arguments": {"a": 125, "b": 8}},
            {"action": "final", "answer": "1000"},
        ]
        with patch.object(agent, "choose_action", side_effect=actions) as choose, \
             patch("policy_tools.search_policy", side_effect=AssertionError("Unexpected search")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(agent.run_agent("125 times 8"), "1000")
        self.assertTrue(all(call.kwargs["rag"] is True for call in choose.call_args_list))

    def test_direct_answer_runs_no_tool(self):
        action = {"action": "final", "answer": "Hello!"}
        with patch.object(agent, "choose_action", return_value=action), \
             patch.object(agent, "execute_tool") as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(agent.run_agent("Hello"), "Hello!")
            execute.assert_not_called()

    def test_rag_passes_evidence_back_to_model(self):
        evidence = {"matches": [{"id": 2, "text": "Refund within 30 days."}]}

        def decide(question, history, model, debug, rag):
            self.assertTrue(rag)
            if not history:
                return {"action": "tool", "tool": "search_policy", "arguments": {"query": "refund"}}
            self.assertEqual(history[0]["observation"]["result"], evidence)
            return {"action": "final", "answer": "Within 30 days [Chunk 2]."}

        with patch.object(agent, "choose_action", side_effect=decide), \
             patch("policy_tools.search_policy", return_value=evidence), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(agent.run_agent("refund"), "Within 30 days [Chunk 2].")

    def test_stops_at_limit(self):
        action = {"action": "tool", "tool": "add_numbers", "arguments": {"a": 1, "b": 2}}
        with patch.object(agent, "choose_action", return_value=action) as choose, \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Step limit"):
            agent.run_agent("repeat", max_steps=2)
        self.assertEqual(choose.call_count, 2)

    def test_invalid_cli_arguments_do_not_call_model(self):
        for argv in [["--mode", "other", "--question", "hi"], ["--question", " "],
                     ["--question", "hi", "--max-steps", "0"]]:
            with self.subTest(argv=argv), patch.object(agent, "choose_action") as choose, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                agent.main(argv)
            self.assertEqual(error.exception.code, 2)
            choose.assert_not_called()


if __name__ == "__main__":
    unittest.main()
