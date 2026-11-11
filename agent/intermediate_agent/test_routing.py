"""Offline checks for Step 6.1 route selection."""

import unittest

from agent_app.routing import route_question


class RoutingTests(unittest.TestCase):
    def test_billing_questions(self):
        for question in ("Why was I charged twice?", "Can I get a refund?", "Where is my invoice?"):
            with self.subTest(question=question):
                self.assertEqual(route_question(question).route, "billing")

    def test_technical_questions(self):
        for question in ("How do I configure the API?", "The server has an error.", "Where are the docs?"):
            with self.subTest(question=question):
                self.assertEqual(route_question(question).route, "technical")

    def test_general_and_mixed_questions(self):
        self.assertEqual(route_question("Hello, how are you?").route, "general")
        mixed = route_question("My payment API has an error")
        self.assertEqual(mixed.route, "general")
        self.assertIn("payment", mixed.billing_matches)
        self.assertIn("api", mixed.technical_matches)

    def test_matches_whole_words_and_validates_input(self):
        self.assertEqual(route_question("I enjoy debugging puzzles").route, "general")
        for value in ("", "   ", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                route_question(value)


if __name__ == "__main__":
    unittest.main()
