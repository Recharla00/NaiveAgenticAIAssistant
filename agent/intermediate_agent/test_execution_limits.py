"""Offline checks for Step 8.1 model and tool call limits."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agent_app.agent import build_agent, run_routed_session_agent, run_session_agent
from agent_app.config import Settings
from agent_app.memory import SessionStore


HERE = Path(__file__).resolve().parent


class FakeToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def tool_request(call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{
        "name": "add_numbers", "args": {"a": 2, "b": 3}, "id": call_id,
    }])


class ExecutionLimitTests(unittest.TestCase):
    def test_model_call_limit_stops_before_third_call_and_preserves_session(self):
        model = FakeToolModel(responses=[tool_request("one"), tool_request("two"),
                                         AIMessage(content="Should not be reached")])
        with patch("agent_app.agent.build_model", return_value=model):
            agent = build_agent(Settings(), enable_tools=True, max_model_calls=2,
                                max_tool_calls=5)
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder)
            with self.assertLogs("agent_app.flow", level="INFO") as captured:
                state = run_session_agent(agent, "Keep calculating", "model-limit", store=store,
                                          max_steps=20)
            self.assertIn("model-call limit", state["messages"][-1].text)
            self.assertEqual(model.i, 2)
            self.assertEqual(store.load("model-limit").messages, [])
            self.assertIn("Session unchanged", "\n".join(captured.output))

    def test_tool_call_limit_blocks_second_tool_before_it_runs(self):
        calls = []

        @tool
        def add_numbers(a: int, b: int) -> int:
            """Add two integers."""
            calls.append((a, b))
            return a + b

        add_numbers.metadata = {"data_access": "read_only"}

        model = FakeToolModel(responses=[tool_request("one"), tool_request("two"),
                                         AIMessage(content="Should not be reached")])
        with patch("agent_app.agent.build_model", return_value=model), \
                patch("agent_app.agent.get_tools", return_value=[add_numbers]):
            agent = build_agent(Settings(), enable_tools=True, max_model_calls=5,
                                max_tool_calls=1)
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder)
            state = run_session_agent(agent, "Keep calculating", "tool-limit", store=store,
                                      max_steps=20)
            self.assertIn("tool-call limit", state["messages"][-1].text)
            self.assertEqual(calls, [(2, 3)])
            self.assertEqual(store.load("tool-limit").messages, [])

    def test_routed_path_uses_its_call_limits(self):
        model = FakeToolModel(responses=[tool_request("one"),
                                         AIMessage(content="Should not be reached")])
        with patch("agent_app.agent.build_model", return_value=model):
            with tempfile.TemporaryDirectory(dir=HERE) as folder:
                decision, state = run_routed_session_agent(
                    Settings(), "Add 2 and 3", "routed-limit", store=SessionStore(folder),
                    max_model_calls=1, max_tool_calls=5, max_steps=20)
        self.assertEqual(decision.route, "general")
        self.assertIn("model-call limit", state["messages"][-1].text)

    def test_technical_presearch_counts_toward_tool_limit(self):
        model = FakeToolModel(responses=[AIMessage(content="", tool_calls=[{
            "name": "search_internal_knowledge_base",
            "args": {"query": "API configuration"}, "id": "again",
        }])])
        evidence = {"query": "How do I configure the API?", "results": [
            {"source": "api_guide.md", "text": "Configure the API in Settings."},
        ]}
        with patch("agent_app.agent.build_model", return_value=model), \
                patch("agent_app.route_paths.search_internal_knowledge_base") as search:
            search.invoke.return_value = evidence
            with tempfile.TemporaryDirectory(dir=HERE) as folder:
                decision, state = run_routed_session_agent(
                    Settings(), "How do I configure the API?", "technical-limit",
                    store=SessionStore(folder), max_tool_calls=1, max_steps=20)
                self.assertEqual(SessionStore(folder).load("technical-limit").messages, [])
        self.assertEqual(decision.route, "technical")
        self.assertIn("tool-call limit", state["messages"][-1].text)
        search.invoke.assert_called_once()

    def test_call_limits_must_be_positive_integers(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_agent(Settings(), max_model_calls=value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_agent(Settings(), max_tool_calls=value)


if __name__ == "__main__":
    unittest.main()
