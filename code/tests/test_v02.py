import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from score_response import parse_response, score


class BenchmarkV02Tests(unittest.TestCase):
    def test_t_ids_parse(self):
        parsed = parse_response("T-001 | app_crash\nT-002 | damaged_item\n")
        self.assertEqual(parsed["answers"]["T-001"], "app_crash")
        self.assertEqual(parsed["answers"]["T-002"], "damaged_item")

    def test_wrong_valid_label_is_complete_but_incorrect(self):
        task = {
            "task_id": "x",
            "allowed_labels": ["app_crash", "damaged_item"],
            "items": [{"item_id": "T-001", "label": "app_crash"}],
        }
        parsed = parse_response("T-001 | damaged_item\n")
        result = score(task, parsed, None, False, False, False)
        self.assertEqual(result["structurally_complete_count"], 1)
        self.assertEqual(result["correct_count"], 0)
        self.assertEqual(result["structural_completion_rate"], 1.0)

    def test_generated_task_uses_local_ids_and_all_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "task.json"
            subprocess.run([
                sys.executable, "generate_task.py",
                "--version", "A",
                "--size", "1",
                "--seed", "676",
                "--output", str(out),
            ], check=True)
            task = json.loads(out.read_text())
            self.assertEqual(task["items"][0]["item_id"], "T-001")
            self.assertIn("source_item_id", task["items"][0])
            self.assertEqual(len(task["allowed_labels"]), 10)

    def test_base_prompt_has_all_labels_and_classifications_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_path = Path(tmp) / "task.json"
            prompt_path = Path(tmp) / "prompt.txt"
            subprocess.run([
                sys.executable, "generate_task.py",
                "--version", "A", "--size", "1", "--seed", "676",
                "--output", str(task_path),
            ], check=True)
            subprocess.run([
                sys.executable, "build_prompt.py",
                "--task", str(task_path),
                "--reporting", "spontaneous",
                "--prompt-style", "base",
                "--output", str(prompt_path),
            ], check=True)
            prompt = prompt_path.read_text()
            for label in json.loads(Path("labels.json").read_text()).keys():
                self.assertIn(f"- {label}:", prompt)
            self.assertTrue(prompt.rstrip().endswith("Classifications:"))


if __name__ == "__main__":
    unittest.main()
