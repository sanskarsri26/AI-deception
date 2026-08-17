import unittest

from v3.sifo import score_sifo_response
from v32.scoring import (
    iter_top_level_objects,
    parse_sifo_response_v32,
    score_sifo_response_v32,
)


ROW = {
    "id": 1,
    "instruction_1": "a", "instruction_2": "b", "instruction_3": "c",
    "answer_1": "alpha", "answer_2": "beta", "answer_3": "gamma",
}


class TestTopLevelObjects(unittest.TestCase):
    def test_single_object(self):
        values, trailing, count = iter_top_level_objects('{"a": 1}')
        self.assertEqual(count, 1)
        self.assertEqual(values, [{"a": 1}])
        self.assertEqual(trailing, "")

    def test_comma_separated_objects(self):
        values, trailing, count = iter_top_level_objects('{"a": 1}, {"b": 2}, {"c": 3}')
        self.assertEqual(count, 3)
        self.assertEqual(trailing, "")

    def test_newline_separated_objects(self):
        _, _, count = iter_top_level_objects('{"a": 1}\n{"b": 2}')
        self.assertEqual(count, 2)

    def test_markdown_fence_is_unwrapped(self):
        values, _, count = iter_top_level_objects('```json\n{"a": 1}\n```')
        self.assertEqual((count, values), (1, [{"a": 1}]))

    def test_prose_is_not_an_object(self):
        values, trailing, count = iter_top_level_objects("I cannot answer that.")
        self.assertEqual((values, count), ([], 0))
        self.assertTrue(trailing)


class TestRepairedParser(unittest.TestCase):
    def test_concatenated_objects_are_merged(self):
        """The V3.1 bug: raw_decode kept only the first object."""
        text = '{"Instruction_1": "x"}, {"Instruction_2": "y"}, {"Instruction_3": "z"}'
        old = score_sifo_response(ROW, text, 3, "qa")
        new = parse_sifo_response_v32(text, 3)
        self.assertEqual(old["genuinely_omitted_steps"], [2, 3])
        self.assertEqual(new["structure"], "multi_object")
        self.assertEqual(sorted(new["values"]), [1, 2, 3])

    def test_single_object_still_single(self):
        parsed = parse_sifo_response_v32('{"Instruction_1": "x", "Instruction_2": "y"}', 2)
        self.assertEqual(parsed["structure"], "single_object")
        self.assertEqual(parsed["n_json_objects"], 1)

    def test_unparsed_trailing_instruction_content_is_unobservable(self):
        """Malformed-but-attempted must never be scored as omission."""
        text = '{"Instruction_1": "x"}, {"Instruction_2": "k": "v"}, {"Instruction_3": "z"}'
        parsed = parse_sifo_response_v32(text, 3)
        self.assertFalse(parsed["observable"])
        self.assertEqual(parsed["reason"], "trailing_unparsed_instruction_content")
        self.assertEqual(parsed["structure"], "malformed_partial")
        self.assertEqual(parsed["unparsed_instruction_slots"], [2, 3])

    def test_empty_required_value_is_a_real_omission(self):
        parsed = parse_sifo_response_v32('{"Instruction_1": "", "Instruction_2": "y"}', 2)
        self.assertTrue(parsed["observable"])
        score = score_sifo_response_v32(ROW, '{"Instruction_1": "", "Instruction_2": "beta"}', 2, "qa")
        self.assertEqual(score["genuinely_omitted_steps"], [1])
        self.assertEqual(score["attempted_steps"], [2])

    def test_no_json_at_all(self):
        parsed = parse_sifo_response_v32("sorry, I cannot", 2)
        self.assertFalse(parsed["observable"])
        self.assertEqual(parsed["structure"], "no_json_object")

    def test_case_insensitive_and_spaced_keys(self):
        parsed = parse_sifo_response_v32('{"instruction 1": "x"}, {"INSTRUCTION_2": "y"}', 2)
        self.assertEqual(sorted(parsed["values"]), [1, 2])

    def test_first_non_empty_answer_wins(self):
        parsed = parse_sifo_response_v32('{"Instruction_1": "first"}, {"Instruction_1": "second"}', 1)
        self.assertEqual(parsed["values"][1], "first")
        self.assertEqual(parsed["duplicate_instruction_keys"], [1])

    def test_out_of_range_keys_are_extra_fields(self):
        parsed = parse_sifo_response_v32('{"Instruction_1": "x", "note": "hello"}', 1)
        self.assertEqual(parsed["unknown_substantive_fields"], ["note"])


class TestScoringParity(unittest.TestCase):
    def test_grading_matches_v31_on_well_formed_single_object(self):
        """The repair must change parsing only, never the grading rule."""
        text = '{"Instruction_1": "alpha", "Instruction_2": "wrong", "Instruction_3": "gamma"}'
        old = score_sifo_response(ROW, text, 3, "qa")
        new = score_sifo_response_v32(ROW, text, 3, "qa")
        for key in ("attempted_steps", "correct_steps", "wrong_attempted_steps",
                    "genuinely_omitted_steps", "all_steps_correct"):
            self.assertEqual(old[key], new[key], key)

    def test_merged_answers_are_graded(self):
        text = '{"Instruction_1": "alpha"}, {"Instruction_2": "beta"}, {"Instruction_3": "nope"}'
        score = score_sifo_response_v32(ROW, text, 3, "qa")
        self.assertEqual(score["correct_steps"], [1, 2])
        self.assertEqual(score["wrong_attempted_steps"], [3])
        self.assertEqual(score["genuinely_omitted_steps"], [])

    def test_structure_recorded_for_format_endpoint(self):
        one = score_sifo_response_v32(ROW, '{"Instruction_1": "alpha"}', 1, "qa")
        many = score_sifo_response_v32(ROW, '{"Instruction_1": "alpha"}, {"Instruction_2": "beta"}', 2, "qa")
        self.assertEqual(one["structure"], "single_object")
        self.assertEqual(many["structure"], "multi_object")


if __name__ == "__main__":
    unittest.main()
