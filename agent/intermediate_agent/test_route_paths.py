"""Offline checks for Step 6.2 route-specific agent behavior."""

import json
import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from agent_app.agent import build_routed_agent
from agent_app.config import Settings
from agent_app.prompts import (BILLING_SYSTEM_PROMPT, GENERAL_SYSTEM_PROMPT,
                               TECHNICAL_SYSTEM_PROMPT, TOOL_RECOVERY_INSTRUCTION)
from agent_app.route_paths import prepare_route_messages


class FakeToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class RoutePathTests(unittest.TestCase):
    def test_each_route_gets_its_own_prompt_and_tools(self):
        expected = {
            "billing": (BILLING_SYSTEM_PROMPT,
                        {"add_numbers", "multiply_numbers", "search_internal_knowledge_base"}),
            "technical": (TECHNICAL_SYSTEM_PROMPT, {"search_internal_knowledge_base"}),
            "general": (GENERAL_SYSTEM_PROMPT, {"add_numbers", "multiply_numbers"}),
        }
        for route, (prompt, names) in expected.items():
            with self.subTest(route=route), patch("agent_app.agent.build_model"), \
                 patch("agent_app.agent.create_agent") as create:
                build_routed_agent(Settings(), route)
                self.assertEqual(create.call_args.kwargs["system_prompt"],
                                 prompt + TOOL_RECOVERY_INSTRUCTION)
                self.assertEqual({tool.name for tool in create.call_args.kwargs["tools"]}, names)
        with self.assertRaises(ValueError):
            build_routed_agent(Settings(), "unknown")

    def test_technical_path_executes_document_search(self):
        model = FakeToolModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "search_internal_knowledge_base", "args": {"query": "API setup"}, "id": "search-1",
            }]),
            AIMessage(content="Follow the setup guide [guide.md]."),
        ])
        with patch("agent_app.agent.build_model", return_value=model):
            agent = build_routed_agent(Settings(), "technical")
        with patch("agent_app.retrieval.HybridRetriever") as retriever:
            retriever.return_value.search.return_value = {
                "query": "API setup", "results": [{"source": "guide.md", "text": "Follow the setup guide."}],
            }
            state = agent.invoke({"messages": [{"role": "user", "content": "How do I set up the API?"}]})
        results = [message for message in state["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tool_call_id, "search-1")
        self.assertIn("guide.md", results[0].content)

    def test_technical_path_searches_even_when_model_would_skip_tool(self):
        evidence = {"query": "How do I configure the API?", "results": [
            {"source": "guide.md", "text": "Configure the API in Settings."},
        ]}
        with patch("agent_app.route_paths.search_internal_knowledge_base") as search:
            search.invoke.return_value = evidence
            prepared = prepare_route_messages("technical", "How do I configure the API?")
        search.invoke.assert_called_once_with({"query": "How do I configure the API?"})
        self.assertIsNone(prepared.answer)
        messages = prepared.messages
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1].tool_calls[0]["id"], messages[2].tool_call_id)
        self.assertIn("guide.md", messages[2].content)

        model = FakeToolModel(responses=[AIMessage(content="Use Settings [guide.md].")])
        with patch("agent_app.agent.build_model", return_value=model):
            agent = build_routed_agent(Settings(), "technical")
        state = agent.invoke({"messages": messages})
        self.assertEqual(state["messages"][-1].text, "Use Settings [guide.md].")
        self.assertTrue(any(isinstance(message, ToolMessage) for message in state["messages"]))

    def test_other_paths_do_not_search_before_model(self):
        with patch("agent_app.route_paths.search_internal_knowledge_base") as search:
            for route in ("billing", "general"):
                self.assertEqual(len(prepare_route_messages(route, "Hello").messages), 1)
        search.invoke.assert_not_called()

    def test_irrelevant_documents_stop_technical_answer(self):
        with patch("agent_app.route_paths.search_internal_knowledge_base") as search:
            search.invoke.return_value = {"query": "How do I configure the API?", "results": [
                {"source": "shop_policies.md", "text": "Refunds require a receipt."},
            ], "count": 1}
            prepared = prepare_route_messages("technical", "How do I configure the API?")
        self.assertIn("could not find relevant", prepared.answer)
        self.assertEqual(json.loads(prepared.messages[-1].content)["results"], [])

    def test_general_path_can_calculate_without_search(self):
        model = FakeToolModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "add_numbers", "args": {"a": 2, "b": 3}, "id": "calc-1",
            }]),
            AIMessage(content="5"),
        ])
        with patch("agent_app.agent.build_model", return_value=model):
            agent = build_routed_agent(Settings(), "general")
        state = agent.invoke({"messages": [{"role": "user", "content": "Add 2 and 3"}]})
        results = [message for message in state["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual(len(results), 1)
        self.assertEqual(float(results[0].content), 5)


if __name__ == "__main__":
    unittest.main()
