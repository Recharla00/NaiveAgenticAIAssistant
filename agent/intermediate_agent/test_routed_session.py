"""Offline end-to-end checks for Step 6.3 routing with saved sessions."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from agent_app.agent import run_routed_session_agent
from agent_app.config import Settings
from agent_app.memory import SessionStore

HERE = Path(__file__).resolve().parent


class FakeToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class RoutedSessionTests(unittest.TestCase):
    def test_topic_switches_keep_history_and_technical_citation(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder)
            with patch("agent_app.agent.build_model", return_value=FakeToolModel(responses=[
                AIMessage(content="Contact billing support with your order number and receipt."),
            ])):
                billing, first = run_routed_session_agent(Settings(), "I was charged twice.", "demo", store=store)
            self.assertEqual(billing.route, "billing")
            self.assertIn("billing support", first["messages"][-1].text)

            evidence = {"query": "How do I configure the API?", "results": [
                {"source": "guide.md", "text": "Configure the API in Settings."},
            ], "count": 1}
            with patch("agent_app.route_paths.search_internal_knowledge_base") as search, \
                 patch("agent_app.agent.build_model", return_value=FakeToolModel(responses=[
                     AIMessage(content="Configure it in Settings [guide.md]."),
                 ])):
                search.invoke.return_value = evidence
                technical, second = run_routed_session_agent(
                    Settings(), "How do I configure the API?", "demo", store=SessionStore(folder))
            self.assertEqual(technical.route, "technical")
            search.invoke.assert_called_once()
            self.assertIn("[guide.md]", second["messages"][-1].text)
            self.assertEqual(len([message for message in second["messages"]
                                  if isinstance(message, ToolMessage)]), 1)

            with patch("agent_app.agent.build_model", return_value=FakeToolModel(responses=[
                AIMessage(content="The guide said to use Settings [guide.md]."),
            ])):
                general, third = run_routed_session_agent(
                    Settings(), "What did that guide say?", "demo", store=SessionStore(folder))
            self.assertEqual(general.route, "general")
            self.assertEqual([message.content for message in third["messages"] if message.type == "human"],
                             ["I was charged twice.", "How do I configure the API?", "What did that guide say?"])
            self.assertIn("guide.md", SessionStore(folder).load("demo").messages[-1].content)
            self.assertEqual(SessionStore(folder).load("another-session").messages, [])

    def test_no_documentation_answer_is_saved_without_model_call(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder)
            with patch("agent_app.route_paths.search_internal_knowledge_base") as search, \
                 patch("agent_app.agent.build_model") as build_model:
                search.invoke.return_value = {"query": "How do I configure the API?", "results": [
                    {"source": "shop_policies.md", "text": "Refunds require a receipt."},
                ], "count": 1}
                decision, state = run_routed_session_agent(
                    Settings(), "How do I configure the API?", "missing-docs", store=store)
            self.assertEqual(decision.route, "technical")
            self.assertIn("could not find relevant", state["messages"][-1].text)
            build_model.assert_not_called()
            restored = SessionStore(folder).load("missing-docs")
            self.assertEqual([message.role for message in restored.messages],
                             ["user", "assistant", "tool", "assistant"])
            self.assertEqual(restored.langchain_messages()[1].tool_calls[0]["id"],
                             restored.langchain_messages()[2].tool_call_id)


if __name__ == "__main__":
    unittest.main()
