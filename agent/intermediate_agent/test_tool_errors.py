"""Offline checks for Step 7 tool failure and recovery behavior."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agent_app.agent import build_agent, run_routed_session_agent, run_session_agent
from agent_app.config import Settings
from agent_app.memory import SessionStore
from agent_app.tool_errors import RECOVERY_EXHAUSTED_ANSWER


HERE = Path(__file__).resolve().parent


class FakeToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class ToolErrorTests(unittest.TestCase):
    def test_model_can_correct_failed_tool_arguments(self):
        model = FakeToolModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "add_numbers", "args": {"a": "two", "b": 3}, "id": "bad-1",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "add_numbers", "args": {"a": 2, "b": 3}, "id": "fixed-2",
            }]),
            AIMessage(content="The answer is 5."),
        ])
        with patch("agent_app.agent.build_model", return_value=model):
            agent = build_agent(Settings(), enable_tools=True)
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            state = run_session_agent(agent, "Add 2 and 3", "corrected", store=SessionStore(folder))
        results = [message for message in state["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual([result.status for result in results], ["error", "success"],
                         state["messages"][-1].text)
        self.assertEqual(results[1].text, "5.0")
        self.assertEqual(state["messages"][-1].text, "The answer is 5.")

    def test_repeated_failures_stop_and_new_turn_gets_fresh_limit(self):
        calls = [AIMessage(content="", tool_calls=[{
            "name": "add_numbers", "args": {"a": "two", "b": 3}, "id": f"bad-{n}",
        }]) for n in range(1, 5)]
        model = FakeToolModel(responses=calls)
        with patch("agent_app.agent.build_model", return_value=model):
            agent = build_agent(Settings(), enable_tools=True)
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder)
            with self.assertLogs("agent_app.flow", level="INFO") as captured:
                state = run_session_agent(agent, "Add two and three", "limited", store=store)
            results = [message for message in state["messages"] if isinstance(message, ToolMessage)]
            self.assertEqual(len(results), 3, state["messages"][-1].text)
            self.assertTrue(all(result.status == "error" for result in results))
            self.assertEqual(state["messages"][-1].text, RECOVERY_EXHAUSTED_ANSWER)
            self.assertIn("Tool recovery limit reached", "\n".join(captured.output))
            self.assertEqual(store.load("limited").messages, [])

            next_model = FakeToolModel(responses=[
                AIMessage(content="", tool_calls=[{
                    "name": "add_numbers", "args": {"a": 2, "b": 3}, "id": "fresh-1",
                }]),
                AIMessage(content="5"),
            ])
            with patch("agent_app.agent.build_model", return_value=next_model):
                next_agent = build_agent(Settings(), enable_tools=True)
            next_state = run_session_agent(next_agent, "Add 2 and 3", "limited", store=store)
            self.assertEqual(next_state["messages"][-1].text, "5")
            self.assertEqual([message.status for message in next_state["messages"]
                              if isinstance(message, ToolMessage)][-1], "success")

    def test_bad_calculator_arguments_reach_model_as_error_result(self):
        model = FakeToolModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "add_numbers", "args": {"a": "two", "b": 3}, "id": "bad-1",
            }]),
            AIMessage(content="I need numeric values for both numbers."),
        ])
        with patch("agent_app.agent.build_model", return_value=model):
            agent = build_agent(Settings(), enable_tools=True)
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            state = run_session_agent(agent, "Add two and three", "bad-input", store=SessionStore(folder))
        result = next(message for message in state["messages"] if isinstance(message, ToolMessage))
        self.assertEqual(result.status, "error")
        self.assertEqual(result.tool_call_id, "bad-1")
        self.assertIn("Please fix the error and try again", result.text)
        self.assertEqual(state["messages"][-1].text, "I need numeric values for both numbers.")

    def test_timeout_reaches_model_as_error_result(self):
        @tool
        def slow_tool(value: int) -> int:
            """Return a value after some work."""
            raise TimeoutError("service took too long")

        slow_tool.metadata = {"data_access": "read_only"}

        model = FakeToolModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "slow_tool", "args": {"value": 1}, "id": "slow-1",
            }]),
            AIMessage(content="The tool timed out. Please try again."),
        ])
        with patch("agent_app.agent.build_model", return_value=model), \
                patch("agent_app.agent.get_tools", return_value=[slow_tool]):
            agent = build_agent(Settings(), enable_tools=True)
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            state = run_session_agent(agent, "Try slow tool", "timeout", store=SessionStore(folder))
        result = next(message for message in state["messages"] if isinstance(message, ToolMessage))
        self.assertEqual(result.status, "error")
        self.assertIn("timed out", result.text)

    def test_routing_search_failure_returns_clear_answer_without_saving(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder)
            with patch("agent_app.route_paths.search_internal_knowledge_base") as search:
                search.invoke.side_effect = FileNotFoundError("No active index")
                decision, state = run_routed_session_agent(
                    Settings(), "How do I configure the API?", "search-failure", store=store)
            self.assertEqual(decision.route, "technical")
            self.assertIn("could not search", state["messages"][-1].text)
            self.assertEqual(next(m for m in state["messages"] if isinstance(m, ToolMessage)).status,
                             "error")
            self.assertEqual(store.load("search-failure").messages, [])

    def test_model_timeout_preserves_last_saved_turn(self):
        first_model = FakeToolModel(responses=[AIMessage(content="My order is 123.")])
        with patch("agent_app.agent.build_model", return_value=first_model):
            first_agent = build_agent(Settings())
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder)
            run_session_agent(first_agent, "Remember order 123", "timeout-session", store=store)
            before = store.load("timeout-session").to_dict()
            failed_agent = Mock()
            failed_agent.invoke.side_effect = TimeoutError("model stalled")
            with self.assertLogs("agent_app.flow", level="INFO") as captured:
                state = run_session_agent(failed_agent, "What is my order?",
                                          "timeout-session", store=store)
            self.assertIn("took too long", state["messages"][-1].text)
            self.assertEqual(store.load("timeout-session").to_dict(), before)
            self.assertIn("Session unchanged", "\n".join(captured.output))

    def test_routed_step_limit_returns_answer_without_saving(self):
        failed_agent = Mock()
        failed_agent.invoke.side_effect = GraphRecursionError("limit reached")
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder)
            with patch("agent_app.agent.build_routed_agent", return_value=failed_agent):
                decision, state = run_routed_session_agent(
                    Settings(), "Hello", "step-limit", store=store)
            self.assertEqual(decision.route, "general")
            self.assertIn("step limit", state["messages"][-1].text)
            self.assertEqual(store.load("step-limit").messages, [])


if __name__ == "__main__":
    unittest.main()
