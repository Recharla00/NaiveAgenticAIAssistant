"""Checks that optional flow logs expose the important execution stages."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agent_app.agent import build_routed_agent, run_routed_session_agent, run_session_agent
from agent_app.config import Settings
from agent_app.memory import SessionStore

HERE = Path(__file__).resolve().parent


class FakeToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class FlowLoggingTests(unittest.TestCase):
    def test_routed_search_logs_route_evidence_and_save(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            with patch("agent_app.route_paths.search_internal_knowledge_base") as search:
                search.invoke.return_value = {"query": "How do I configure the API?", "results": [
                    {"source": "shop_policies.md", "text": "Refunds require a receipt."},
                ], "count": 1}
                with self.assertLogs("agent_app.flow", level="INFO") as captured:
                    run_routed_session_agent(Settings(), "How do I configure the API?", "logs",
                                             store=SessionStore(folder))
        output = "\n".join(captured.output)
        self.assertIn("Session loaded", output)
        self.assertIn("Route selected: technical", output)
        self.assertIn("Knowledge search started", output)
        self.assertIn("candidates=1 relevant=0", output)
        self.assertIn("Model skipped", output)
        self.assertIn("Session saved", output)

    def test_model_tool_request_and_result_are_logged(self):
        model = FakeToolModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "add_numbers", "args": {"a": 2, "b": 3}, "id": "call-1",
            }]),
            AIMessage(content="The answer is 5."),
        ])
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            with patch("agent_app.agent.build_model", return_value=model):
                agent = build_routed_agent(Settings(), "general")
            with self.assertLogs("agent_app.flow", level="INFO") as captured:
                run_session_agent(agent, "Add 2 and 3", "logs", store=SessionStore(folder))
        output = "\n".join(captured.output)
        self.assertIn("Agent started", output)
        self.assertIn("Model requested tool: add_numbers call_id=call-1", output)
        self.assertIn("Tool returned: add_numbers call_id=call-1", output)
        self.assertIn("Agent finished", output)
        self.assertIn("Session saved", output)

        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            # Recreate a saved calculator exchange and inspect its restored trace.
            with patch("agent_app.agent.build_model", return_value=FakeToolModel(responses=[
                AIMessage(content="", tool_calls=[{
                    "name": "add_numbers", "args": {"a": 2, "b": 3}, "id": "call-1",
                }]), AIMessage(content="5")
            ])):
                first_agent = build_routed_agent(Settings(), "general")
            run_session_agent(first_agent, "Add 2 and 3", "debug-history", store=SessionStore(folder))
            with patch("agent_app.agent.build_model", return_value=FakeToolModel(responses=[
                AIMessage(content="The answer was 5."),
            ])):
                next_agent = build_routed_agent(Settings(), "general")
            with self.assertLogs("agent_app.flow", level="DEBUG") as debug_log:
                run_session_agent(next_agent, "What was the answer?", "debug-history",
                                  store=SessionStore(folder))
            detail = "\n".join(debug_log.output)
            self.assertIn("History 2: assistant requested add_numbers[call-1]", detail)
            self.assertIn("History 3: tool add_numbers[call-1]", detail)


if __name__ == "__main__":
    unittest.main()
