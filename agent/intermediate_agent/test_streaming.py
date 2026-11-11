"""Offline checks for streamed output and timing, without Ollama."""

import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent_app.streaming import print_timings, stream_answer


class StreamingTests(unittest.TestCase):
    def test_streams_chunks_once_and_retains_final_state(self):
        state = {"messages": [SimpleNamespace(text="Hello world")]}
        agent = Mock()
        agent.stream.return_value = iter([
            ("values", {"messages": []}),
            ("messages", (SimpleNamespace(text=""), {})),
            ("messages", (SimpleNamespace(text="Hello "), {})),
            ("messages", (SimpleNamespace(text="world"), {})),
            ("values", state),
        ])
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch("agent_app.streaming.perf_counter", side_effect=[10, 12, 15]):
            result, first, elapsed = stream_answer(agent, "hello")
        self.assertIs(result, state)
        self.assertEqual(output.getvalue(), "Hello world\n")
        self.assertEqual((first, elapsed), (2, 5))

    def test_empty_response_fails(self):
        agent = Mock()
        agent.stream.return_value = iter([])
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            stream_answer(agent, "hello")

    def test_timing_converts_nanoseconds_and_omits_missing_metrics(self):
        output = io.StringIO()
        response = SimpleNamespace(response_metadata={"load_duration": 1000000000,
                                                      "eval_duration": 2000000000, "eval_count": 10})
        with contextlib.redirect_stdout(output):
            print_timings(response, 1.5, 4)
        self.assertIn("Model loading: 1.00s", output.getvalue())
        self.assertIn("5.0 tokens/s", output.getvalue())
        self.assertNotIn("Prompt processing", output.getvalue())


if __name__ == "__main__":
    unittest.main()
