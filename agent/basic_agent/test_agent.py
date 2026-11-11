"""Offline checks: python -m unittest discover -s agent/basic_agent -p test_agent.py"""

import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from agent_helpers import execute_tool, validate_action

spec = importlib.util.spec_from_file_location("agent_loop", Path(__file__).with_name("05_agent_loop.py"))
agent_loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_loop)


def tool(name="add_numbers", a=20, b=5):
    return {"action": "tool", "tool": name, "arguments": {"a": a, "b": b}}


class AgentTests(unittest.TestCase):
    def test_execution_and_invalid_requests(self):
        self.assertEqual(execute_tool(tool())["result"], 25)
        self.assertEqual(execute_tool(tool("multiply_numbers", -2, 0.5))["result"], -1)
        for action in [tool("unknown"), tool(a=True), tool(a="20"), tool(a=float("nan")),
                       tool(a=float("inf")), tool(a=10**400), [],
                       {"action": "final", "answer": ""},
                       {"action": "tool", "tool": "add_numbers", "arguments": {"a": 1}}]:
            with self.subTest(action=action), self.assertRaises(ValueError):
                validate_action(action)

    def test_two_tools_then_answer_receives_history(self):
        actions = [tool(), tool("multiply_numbers", 25, 8), {"action": "final", "answer": "200"}]
        histories = []

        def decide(question, history, model, debug):
            histories.append(list(history))
            return actions[len(histories) - 1]

        with patch.object(agent_loop, "choose_action", side_effect=decide), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(agent_loop.run_agent("Add then multiply"), "200")
        self.assertEqual([len(h) for h in histories], [0, 1, 2])
        self.assertEqual(histories[-1][0]["observation"]["result"], 25)
        self.assertEqual(histories[-1][1]["observation"]["result"], 200)

    def test_direct_answer_skips_tools(self):
        with patch.object(agent_loop, "choose_action", return_value={"action": "final", "answer": "Hello"}), \
             patch.object(agent_loop, "execute_tool") as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(agent_loop.run_agent("Hello"), "Hello")
            execute.assert_not_called()

    def test_repeated_tool_stops_at_limit(self):
        with patch.object(agent_loop, "choose_action", return_value=tool()) as choose, \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Step limit"):
            agent_loop.run_agent("Keep adding", max_steps=2)
        self.assertEqual(choose.call_count, 2)


if __name__ == "__main__":
    unittest.main()
