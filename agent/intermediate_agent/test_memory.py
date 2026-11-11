"""Offline checks for Step 5 conversation data and local persistence."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_app.agent import build_agent, run_session_agent
from agent_app.config import Settings
from agent_app.memory import MessageRecord, SessionState, SessionStore

HERE = Path(__file__).resolve().parent


class MemoryModelTests(unittest.TestCase):
    def test_message_round_trip_preserves_tool_metadata(self):
        message = MessageRecord(
            role="tool",
            content="25",
            name="add_numbers",
            tool_call_id="call-1",
            metadata={"duration_ms": 3},
        )
        restored = MessageRecord.from_dict(message.to_dict())
        self.assertEqual(restored.role, "tool")
        self.assertEqual(restored.content, "25")
        self.assertEqual(restored.tool_call_id, "call-1")
        self.assertEqual(restored.metadata, {"duration_ms": 3})

    def test_session_add_and_round_trip(self):
        session = SessionState("session-1")
        session.add(MessageRecord("user", "What is the refund policy?"))
        session.add(MessageRecord("assistant", "I will search the knowledge base."))
        restored = SessionState.from_dict(session.to_dict())
        self.assertEqual(restored.session_id, "session-1")
        self.assertEqual([item.role for item in restored.messages], ["user", "assistant"])
        self.assertEqual(restored.messages[0].content, "What is the refund policy?")

    def test_invalid_values_fail_early(self):
        with self.assertRaises(ValueError): MessageRecord("unknown", "text")
        with self.assertRaises(ValueError): SessionState(" ")
        with self.assertRaises(ValueError): MessageRecord.from_dict({"role": "user"})
        with self.assertRaises(ValueError): SessionState.from_dict({"session_id": "x", "schema_version": 99})


class SessionStoreTests(unittest.TestCase):
    def test_messages_survive_a_new_store_instance_and_sessions_are_isolated(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder)
            user = store.add_message("conversation-1", "user", "My order is 42")
            assistant = store.add_message("conversation-1", "assistant", "I can help with order 42")
            restored = SessionStore(folder).load("conversation-1")
            self.assertEqual([item.content for item in restored.messages],
                             ["My order is 42", "I can help with order 42"])
            self.assertEqual([item.message_id for item in restored.messages],
                             [user.message_id, assistant.message_id])
            self.assertEqual(SessionStore(folder).load("conversation-2").messages, [])
            self.assertEqual(len(list(Path(folder).glob("*.json"))), 1)

    def test_session_id_cannot_escape_store_directory(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder)
            path = store.save(SessionState("../another/session"))
            self.assertEqual(path.parent, Path(folder).resolve())
            self.assertEqual(store.load("../another/session").session_id, "../another/session")

    def test_invalid_role_does_not_modify_saved_session(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder)
            store.add_message("conversation-1", "user", "Hello")
            with self.assertRaises(ValueError):
                store.add_message("conversation-1", "tool", "unexpected")
            self.assertEqual(len(store.load("conversation-1").messages), 1)

    def test_history_limit_preserves_complete_tool_turn(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder, max_turns=1)
            session = SessionState("session-1")
            session.add(MessageRecord("user", "Old question"))
            session.add(MessageRecord("assistant", "Old answer"))
            for message in [
                HumanMessage(content="What is the policy?"),
                AIMessage(content="", tool_calls=[{"name": "search_internal_knowledge_base",
                                                   "args": {"query": "policy"}, "id": "call-1"}]),
                ToolMessage(content='{"source": "policy.md"}', tool_call_id="call-1"),
                AIMessage(content="The policy is in policy.md."),
            ]:
                session.add(MessageRecord.from_langchain(message))
            store.save(session)
            restored = store.load("session-1")
            self.assertEqual([item.role for item in restored.messages],
                             ["user", "assistant", "tool", "assistant"])
            self.assertEqual(restored.langchain_messages()[2].tool_call_id, "call-1")

    def test_corrupt_session_is_archived_and_can_be_restarted(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder)
            path = store.save(SessionState("session-1"))
            path.write_text("not JSON", encoding="utf-8")
            with self.assertWarnsRegex(RuntimeWarning, "corrupt"):
                fresh = store.load("session-1")
            self.assertEqual(fresh.messages, [])
            archived = list(Path(folder).glob("*.corrupt.*.json"))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_text(encoding="utf-8"), "not JSON")
            store.add_message("session-1", "user", "Start again")
            self.assertEqual(store.load("session-1").messages[0].content, "Start again")

    def test_expired_session_is_archived(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder, max_age_days=30)
            session = SessionState("session-1", updated_at="2000-01-01T00:00:00Z")
            path = store.save(session)
            with self.assertWarnsRegex(RuntimeWarning, "expired"):
                fresh = store.load("session-1")
            self.assertEqual(fresh.messages, [])
            self.assertFalse(path.exists())
            archived = list(Path(folder).glob("*.expired.*.json"))
            self.assertEqual(len(archived), 1)
            self.assertEqual(json.loads(archived[0].read_text(encoding="utf-8"))["session_id"], "session-1")


class FakeToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class RecordingAgent:
    def __init__(self, graph):
        self.graph = graph
        self.received = None

    def invoke(self, payload, config):
        self.received = payload["messages"]
        return self.graph.invoke(payload, config=config)


class SessionAgentTests(unittest.TestCase):
    def test_restored_history_is_given_to_agent(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder)
            store.add_message("conversation-1", "user", "My order is 42")
            store.add_message("conversation-1", "assistant", "I can help with order 42")
            model = FakeToolModel(responses=[AIMessage(content="Your order was 42.")])
            with patch("agent_app.agent.build_model", return_value=model):
                agent = RecordingAgent(build_agent(Settings()))
            run_session_agent(agent, "What was my order number?", "conversation-1", store=SessionStore(folder))
            self.assertEqual([message.content for message in agent.received],
                             ["My order is 42", "I can help with order 42", "What was my order number?"])
            self.assertEqual([message.role for message in store.load("conversation-1").messages],
                             ["user", "assistant", "user", "assistant"])
            self.assertEqual(store.load("conversation-2").messages, [])

    def test_search_tool_call_and_evidence_survive_restart(self):
        evidence = {"query": "refund", "results": [{"text": "Refund in 30 days", "source": "policy.md"}]}
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder)
            model = FakeToolModel(responses=[
                AIMessage(content="", tool_calls=[{"name": "search_internal_knowledge_base",
                                                   "args": {"query": "refund"}, "id": "search-1"}]),
                AIMessage(content="Refunds are available within 30 days [policy.md]."),
            ])
            with patch("agent_app.agent.build_model", return_value=model):
                graph = build_agent(Settings(), enable_retrieval=True)
            with patch("agent_app.retrieval.HybridRetriever") as retriever:
                retriever.return_value.search.return_value = evidence
                first = run_session_agent(graph, "What is the refund policy?", "session-rag", store=store)
            self.assertTrue(any(isinstance(message, ToolMessage) for message in first["messages"]))

            resumed_model = FakeToolModel(responses=[AIMessage(content="It is 30 days [policy.md].")])
            with patch("agent_app.agent.build_model", return_value=resumed_model):
                resumed = RecordingAgent(build_agent(Settings(), enable_retrieval=True))
            run_session_agent(resumed, "How many days was that?", "session-rag", store=SessionStore(folder))
            request = next(message for message in resumed.received
                           if isinstance(message, AIMessage) and message.tool_calls)
            result = next(message for message in resumed.received if isinstance(message, ToolMessage))
            self.assertEqual(request.tool_calls[0]["id"], "search-1")
            self.assertEqual(request.tool_calls[0]["args"], {"query": "refund"})
            self.assertEqual(result.tool_call_id, "search-1")
            self.assertIn("policy.md", result.content)
            self.assertIn("Refund in 30 days", result.content)

    def test_incomplete_turn_does_not_replace_saved_history(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            store = SessionStore(folder)
            store.add_message("session-1", "user", "Before")
            model = FakeToolModel(responses=[AIMessage(content="")])
            with patch("agent_app.agent.build_model", return_value=model):
                agent = build_agent(Settings())
            with self.assertRaisesRegex(ValueError, "complete final answer"):
                run_session_agent(agent, "After", "session-1", store=store)
            self.assertEqual([message.content for message in store.load("session-1").messages], ["Before"])

    def test_follow_up_after_process_restart_and_cli_help(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", folder],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            model = FakeToolModel(responses=[AIMessage(content="Your order number is 42.")])
            with patch("agent_app.agent.build_model", return_value=model):
                agent = RecordingAgent(build_agent(Settings()))
            run_session_agent(agent, "What was my order number?", "restart-session",
                              store=SessionStore(folder))
            self.assertEqual([message.content for message in agent.received],
                             ["My order number is 42.", "I can help with order 42.",
                              "What was my order number?"])
        help_result = subprocess.run([sys.executable, str(HERE / "05_session_agent.py"), "--help"],
                                     capture_output=True, text=True, timeout=30)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--session-id", help_result.stdout)


def persistence_worker(folder):
    model = FakeToolModel(responses=[AIMessage(content="I can help with order 42.")])
    with patch("agent_app.agent.build_model", return_value=model):
        agent = build_agent(Settings())
    run_session_agent(agent, "My order number is 42.", "restart-session", store=SessionStore(folder))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        persistence_worker(sys.argv[2])
    else:
        unittest.main()
