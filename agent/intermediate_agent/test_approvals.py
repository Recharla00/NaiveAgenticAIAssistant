"""Offline checks for Step 8.3 per-call approval before data changes."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from agent_app.agent import build_agent, run_session_agent
from agent_app.approvals import ApprovalRequest, interactive_approval
from agent_app.config import Settings
from agent_app.memory import SessionStore
from agent_app.tools import get_tools
from agent_app.tool_errors import APPROVAL_DENIED_ANSWER


HERE = Path(__file__).resolve().parent


class FakeToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def note_model():
    return FakeToolModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "save_demo_note", "args": {"note": "Buy milk"}, "id": "write-1",
        }]),
        AIMessage(content="The demo note was saved."),
    ])


class ApprovalTests(unittest.TestCase):
    def test_denial_keeps_file_and_session_unchanged(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            notes = Path(folder) / "notes.txt"
            with patch("agent_app.tools.demo_write.DEMO_NOTES_FILE", notes), \
                    patch("agent_app.agent.build_model", return_value=note_model()):
                agent = build_agent(Settings(), enable_demo_write=True,
                                    approval_handler=lambda request: False)
                store = SessionStore(Path(folder) / "sessions")
                state = run_session_agent(agent, "Save demo note: Buy milk", "denied", store=store)
                self.assertFalse(notes.exists())
                self.assertEqual(store.load("denied").messages, [])
        self.assertEqual(state["messages"][-1].text, APPROVAL_DENIED_ANSWER)
        result = next(m for m in state["messages"] if isinstance(m, ToolMessage))
        self.assertEqual(result.status, "error")
        self.assertTrue(result.artifact["approval_denied"])

    def test_approval_runs_tool_once_and_saves_successful_turn(self):
        requests = []

        def approve(request):
            requests.append(request)
            return True

        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            notes = Path(folder) / "notes.txt"
            with patch("agent_app.tools.demo_write.DEMO_NOTES_FILE", notes), \
                    patch("agent_app.agent.build_model", return_value=note_model()):
                agent = build_agent(Settings(), enable_demo_write=True,
                                    approval_handler=approve)
                store = SessionStore(Path(folder) / "sessions")
                state = run_session_agent(agent, "Save demo note: Buy milk", "approved", store=store)
                self.assertEqual(notes.read_text(encoding="utf-8"), "Buy milk\n")
                self.assertEqual(store.load("approved").messages[-1].content,
                                 "The demo note was saved.")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0], ApprovalRequest("save_demo_note", {"note": "Buy milk"},
                                                      "write-1"))
        self.assertEqual(state["messages"][-1].text, "The demo note was saved.")

    def test_read_only_tool_needs_no_approval(self):
        model = FakeToolModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "add_numbers", "args": {"a": 2, "b": 3}, "id": "read-1",
            }]), AIMessage(content="5"),
        ])

        def unexpected_approval(_request):
            raise AssertionError("Read-only tools should not prompt")

        with patch("agent_app.agent.build_model", return_value=model):
            agent = build_agent(Settings(), enable_tools=True,
                                approval_handler=unexpected_approval)
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            state = run_session_agent(agent, "Add 2 and 3", "read-only",
                                      store=SessionStore(folder))
        self.assertEqual(state["messages"][-1].text, "5")

    def test_unclassified_tool_is_denied_by_default(self):
        changes = []

        @tool
        def add_numbers(a: int, b: int) -> int:
            """Pretend to add numbers while changing data."""
            changes.append((a, b))
            return a + b

        model = FakeToolModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "add_numbers", "args": {"a": 2, "b": 3}, "id": "unknown-1",
            }]), AIMessage(content="Should not be reached"),
        ])
        with patch("agent_app.agent.get_tools", return_value=[add_numbers]), \
                patch("agent_app.agent.build_model", return_value=model):
            agent = build_agent(Settings(), enable_tools=True)
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            state = run_session_agent(agent, "Add 2 and 3", "unknown",
                                      store=SessionStore(folder))
        self.assertEqual(changes, [])
        self.assertEqual(state["messages"][-1].text, APPROVAL_DENIED_ANSWER)

    def test_interactive_prompt_shows_proposed_change(self):
        request = ApprovalRequest("save_demo_note", {"note": "Buy milk"}, "write-1")
        output = io.StringIO()
        with patch("builtins.input", return_value="yes"), redirect_stderr(output):
            self.assertTrue(interactive_approval(request))
        self.assertIn("save_demo_note", output.getvalue())
        self.assertIn("Buy milk", output.getvalue())
        with patch("builtins.input", return_value="no"):
            self.assertFalse(interactive_approval(request))


if __name__ == "__main__":
    unittest.main()
