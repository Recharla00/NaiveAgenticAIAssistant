"""Verify that the evaluation rules do not accidentally accept wrong answers."""

from pathlib import Path
import runpy
import unittest


grade_answer = runpy.run_path(str(Path(__file__).with_name("01_evaluate_llm.py")))["grade_answer"]


class GradingTests(unittest.TestCase):
    def test_exact_ignores_case_and_whitespace(self):
        case = {"check": "exact", "expected": "NOT KNOWN"}
        self.assertTrue(grade_answer("  not\nknown  ", case)[0])

    def test_exact_rejects_wrong_numbers_and_extra_text(self):
        case = {"check": "exact", "expected": "5"}
        for answer in ("50", "not 5", "The answer is 5", "5.", ""):
            with self.subTest(answer=answer):
                self.assertFalse(grade_answer(answer, case)[0])

    def test_json_accepts_reordered_fields(self):
        case = {"check": "json", "expected": {"name": "Asha", "city": "Pune"}}
        self.assertTrue(grade_answer('{"city":"Pune", "name":"Asha"}', case)[0])

    def test_json_rejects_wrong_extra_missing_or_wrapped_content(self):
        case = {"check": "json", "expected": {"name": "Asha", "city": "Pune"}}
        for answer in ('{"name":"Asha","city":"Delhi"}', '{"name":"Asha"}',
                       '{"name":"Asha","city":"Pune","age":20}',
                       '```json\n{"name":"Asha","city":"Pune"}\n```'):
            with self.subTest(answer=answer):
                self.assertFalse(grade_answer(answer, case)[0])

    def test_json_rejects_boolean_as_number(self):
        self.assertFalse(grade_answer('{"count":true}', {"check": "json", "expected": {"count": 1}})[0])

    def test_json_rejects_duplicates_and_nonstandard_constants(self):
        case = {"check": "json", "expected": {"count": 1}}
        for answer in ('{"count":0,"count":1}', '{"count":NaN}'):
            self.assertFalse(grade_answer(answer, case)[0])


if __name__ == "__main__":
    unittest.main()
