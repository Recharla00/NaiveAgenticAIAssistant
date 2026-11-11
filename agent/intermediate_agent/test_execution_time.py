"""Offline checks for Step 8.2 question time budgets."""

from pathlib import Path
from time import sleep
import tempfile
import unittest
from unittest.mock import patch

from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agent_app.agent import build_agent, run_routed_session_agent, run_session_agent
from agent_app.config import Settings
from agent_app.execution_time import (RunDeadlineExceeded, check_deadline,
                                      question_time_limit, validate_max_seconds)
from agent_app.memory import SessionStore


HERE = Path(__file__).resolve().parent


class FakeToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class SlowModel(FakeToolModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        sleep(0.04)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class ExecutionTimeTests(unittest.TestCase):
    def test_time_limit_must_be_positive_and_finite(self):
        for value in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_max_seconds(value)

    def test_nested_limit_cannot_extend_existing_deadline(self):
        with question_time_limit(0.01):
            sleep(0.02)
            with question_time_limit(10):
                with self.assertRaises(RunDeadlineExceeded):
                    check_deadline()

    def test_slow_tool_times_out_and_preserves_saved_history(self):
        calls = []

        @tool
        def slow_tool(value: int) -> int:
            """Return a number after slow work."""
            calls.append(value)
            sleep(0.2)
            return value

        slow_tool.metadata = {"data_access": "read_only"}

        first_model = FakeToolModel(responses=[AIMessage(content="Remembered.")])
        with patch("agent_app.agent.build_model", return_value=first_model):
            first_agent = build_agent(Settings())
        slow_model = FakeToolModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "slow_tool", "args": {"value": 2}, "id": "slow-1",
            }]),
            AIMessage(content="Should not be reached."),
        ])
        with patch("agent_app.agent.build_model", return_value=slow_model), \
                patch("agent_app.agent.get_tools", return_value=[slow_tool]):
            slow_agent = build_agent(Settings(), enable_tools=True)
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder)
            run_session_agent(first_agent, "Remember this", "time-demo", store=store)
            before = store.load("time-demo").to_dict()
            with self.assertLogs("agent_app.flow", level="INFO") as captured:
                state = run_session_agent(slow_agent, "Run the slow tool", "time-demo",
                                          store=store, max_seconds=0.1)
            self.assertIn("time limit", state["messages"][-1].text)
            self.assertEqual(calls, [2])
            self.assertEqual(slow_model.i, 1)
            self.assertEqual(store.load("time-demo").to_dict(), before)
            self.assertIn("Question time limit reached", "\n".join(captured.output))

    def test_slow_model_times_out_without_saving_question(self):
        model = SlowModel(responses=[AIMessage(content="Too late")])
        with patch("agent_app.agent.build_model", return_value=model):
            agent = build_agent(Settings())
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder)
            state = run_session_agent(agent, "Hello", "slow-model", store=store,
                                      max_seconds=0.01)
            self.assertIn("time limit", state["messages"][-1].text)
            self.assertEqual(store.load("slow-model").messages, [])

    def test_slow_technical_search_times_out_before_model(self):
        evidence = {"query": "How do I configure the API?", "results": [
            {"source": "api_guide.md", "text": "Configure the API in Settings."},
        ]}

        def slow_search(_args):
            sleep(0.04)
            return evidence

        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder)
            with patch("agent_app.route_paths.search_internal_knowledge_base") as search, \
                    patch("agent_app.agent.build_routed_agent") as build:
                search.invoke.side_effect = slow_search
                decision, state = run_routed_session_agent(
                    Settings(), "How do I configure the API?", "search-time",
                    store=store, max_seconds=0.01)
            self.assertEqual(decision.route, "technical")
            self.assertIn("time limit", state["messages"][-1].text)
            self.assertEqual(store.load("search-time").messages, [])
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
