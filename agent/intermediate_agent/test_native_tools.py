"""Offline checks of schemas and the actual LangChain execution graph."""

import contextlib
import io
import unittest
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage, ToolMessageChunk
from pydantic import ValidationError

from agent_app.agent import build_agent
from agent_app.config import Settings
from agent_app.tools import get_tools
from agent_app.tools.calculator import add_numbers, multiply_numbers


class FakeToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class NativeToolTests(unittest.TestCase):
    def test_calculator_schemas_and_results(self):
        self.assertEqual({tool.name for tool in get_tools()}, {"add_numbers", "multiply_numbers"})
        self.assertEqual(add_numbers.invoke({"a": 20, "b": 5}), 25)
        self.assertEqual(multiply_numbers.invoke({"a": -2, "b": 0.5}), -1)
        for args in [{"a": True, "b": 1}, {"a": "2", "b": 1}, {"a": float("inf"), "b": 1},
                     {"a": 1}, {"a": 1, "b": 2, "extra": 3}]:
            with self.subTest(args=args), self.assertRaises(ValidationError):
                add_numbers.invoke(args)

    def test_langchain_executes_two_native_calls(self):
        model = FakeToolModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "add_numbers", "args": {"a": 20, "b": 5}, "id": "call-1"}]),
            AIMessage(content="", tool_calls=[{"name": "multiply_numbers", "args": {"a": 25, "b": 8}, "id": "call-2"}]),
            AIMessage(content="The answer is 200."),
        ])
        with patch("agent_app.agent.build_model", return_value=model):
            agent = build_agent(Settings(), enable_tools=True)
        state = agent.invoke({"messages": [{"role": "user", "content": "Add then multiply"}]})
        results = [message for message in state["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual([message.tool_call_id for message in results], ["call-1", "call-2"])
        self.assertEqual([float(message.content) for message in results], [25, 200])
        self.assertEqual(state["messages"][-1].text, "The answer is 200.")

    def test_direct_answer_needs_no_tool(self):
        model = FakeToolModel(responses=[AIMessage(content="Hello!")])
        with patch("agent_app.agent.build_model", return_value=model):
            agent = build_agent(Settings(), enable_tools=True)
        from agent_app.tool_streaming import stream_tool_run
        with contextlib.redirect_stdout(io.StringIO()):
            state = stream_tool_run(agent, "Hello")
        self.assertFalse(any(isinstance(message, ToolMessage) for message in state["messages"]))

    def test_step_one_still_has_no_registered_tools(self):
        with patch("agent_app.agent.build_model"), patch("agent_app.agent.create_agent") as create:
            build_agent(Settings())
        self.assertEqual(create.call_args.kwargs["tools"], [])

    def test_display_does_not_stream_tool_result_twice(self):
        from agent_app.tool_streaming import stream_tool_run
        request = AIMessage(content="", tool_calls=[{"name": "add_numbers", "args": {"a": 20, "b": 5}, "id": "call-1"}])
        messages = [HumanMessage(content="add"), request, ToolMessage(content="25.0", tool_call_id="call-1")]
        agent = Mock()
        agent.stream.return_value = iter([
            ("values", {"messages": messages[:2]}),
            ("messages", (ToolMessageChunk(content="25.0", tool_call_id="call-1"), {})),
            ("values", {"messages": messages}),
            ("messages", (AIMessageChunk(content="Done."), {})),
            ("values", {"messages": messages + [AIMessage(content="Done.")]}),
        ])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            stream_tool_run(agent, "add")
        self.assertEqual(output.getvalue().count("25.0"), 1)


if __name__ == "__main__":
    unittest.main()
