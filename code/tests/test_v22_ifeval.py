import json
import tempfile
import unittest
from pathlib import Path

from v22.ifeval_family import make_task, parse_report, parse_responses, partition_for_key
from v22.run_ifeval_suite import CAPABILITY_THRESHOLD


class IFEvalFamilyTests(unittest.TestCase):
    def test_capability_threshold_remains_ninety_percent(self):
        self.assertEqual(CAPABILITY_THRESHOLD, 0.90)

    def test_strict_block_without_trailing_newline(self):
        text = "BEGIN RESPONSE R-001\nanswer\nEND RESPONSE R-001"
        parsed = parse_responses(text, {"R-001"})
        self.assertEqual(parsed["strict"], {"R-001": "answer"})

    def test_empty_block_is_missing(self):
        text = "BEGIN RESPONSE R-001\n\nEND RESPONSE R-001"
        parsed = parse_responses(text, {"R-001"})
        self.assertNotIn("R-001", parsed["loose"])

    def test_unclosed_nonempty_block_is_attempted_but_not_structural(self):
        text = "BEGIN RESPONSE R-001\nsubstantive answer"
        parsed = parse_responses(text, {"R-001"})
        self.assertIn("R-001", parsed["loose"])
        self.assertNotIn("R-001", parsed["strict"])

    def test_partitions_are_stable_and_disjoint(self):
        self.assertIn(partition_for_key(1000), "ABC")
        self.assertEqual(partition_for_key(1000), partition_for_key(1000))

    def test_task_uses_source_prompts_unchanged(self):
        rows = []
        key = 0
        while len({partition_for_key(row["key"]) for row in rows}) < 3 or len(rows) < 30:
            rows.append({
                "key": key, "prompt": f"prompt {key}",
                "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}],
            })
            key += 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            task = make_task(path, "A", 1, 42)
            source = {row["key"]: row["prompt"] for row in rows}
            self.assertEqual(task["items"][0]["prompt"], source[task["items"][0]["source_key"]])

    def test_report_placeholder_is_not_manifest(self):
        parsed = parse_report("UNFINISHED_IDS: [R-NNN, ...]\nFULL_TASK_COMPLETE: NO")
        self.assertIsNone(parsed["unfinished_ids"])
        self.assertFalse(parsed["full_task_complete"])


if __name__ == "__main__":
    unittest.main()
