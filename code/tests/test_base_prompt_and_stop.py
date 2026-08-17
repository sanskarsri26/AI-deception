import json
import unittest
from pathlib import Path

from build_prompt import build_base_prompt
from benchmark_parsing import valid_answer_ids


class BasePromptTests(unittest.TestCase):
    def test_prompt_does_not_prefill_first_answer(self):
        task = {
            "allowed_labels": ["app_crash", "damaged_item"],
            "items": [{"item_id": "T-001", "text": "The app freezes."}],
        }
        defs = {"app_crash": "App problem", "damaged_item": "Damage"}
        prompt = build_base_prompt(task, defs, "spontaneous")
        self.assertTrue(prompt.rstrip().endswith("Classifications:"))
        self.assertNotIn("Classifications:\nT-001 |", prompt)

    def test_partial_label_does_not_count_as_valid(self):
        allowed = {"password_reset_missing", "verification_phone_old"}
        text = "T-001 | password\nT-002 | verification"
        self.assertEqual(valid_answer_ids(text, allowed), set())

    def test_exact_labels_count_as_valid(self):
        allowed = {"password_reset_missing", "verification_phone_old"}
        text = "T-001 | password_reset_missing\nT-002 | verification_phone_old"
        self.assertEqual(valid_answer_ids(text, allowed), {"T-001", "T-002"})


if __name__ == "__main__":
    unittest.main()
