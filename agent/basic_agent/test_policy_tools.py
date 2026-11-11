"""Offline policy-tool checks using a temporary store."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent_helpers import execute_tool, validate_action
from policy_tools import search_policy


class PolicyTests(unittest.TestCase):
    def test_search_returns_evidence_and_no_match(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / "store.json"
            path.write_text(json.dumps([{
                "id": 2, "source": "policy.txt", "text": "Refund within 30 days.",
                "vector": {"refund": 1, "within": 1, "30": 1, "days": 1},
            }]), encoding="utf-8")
            with patch("policy_tools.STORE_PATH", path):
                result = search_policy("What is the refund?")
                self.assertEqual(result["matches"][0]["id"], 2)
                self.assertEqual(result["matches"][0]["source"], "policy.txt")
                self.assertEqual(search_policy("galaxy telescope")["matches"], [])
                self.assertEqual(search_policy("what is the")["matches"], [])
            with patch("policy_tools.STORE_PATH", Path(folder) / "missing.json"):
                with self.assertRaisesRegex(ValueError, "store missing"):
                    search_policy("refund")

    def test_search_only_allowed_in_rag(self):
        action = {"action": "tool", "tool": "search_policy", "arguments": {"query": "refund"}}
        with self.assertRaises(ValueError):
            execute_tool(action)
        with patch("policy_tools.search_policy", return_value={"matches": []}) as search:
            self.assertEqual(execute_tool(action, rag=True)["result"], {"matches": []})
            search.assert_called_once_with(query="refund")
        for arguments in [{"query": ""}, {"query": 3}, {"query": "refund", "path": "other"}]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_action({**action, "arguments": arguments}, rag=True)


if __name__ == "__main__":
    unittest.main()
