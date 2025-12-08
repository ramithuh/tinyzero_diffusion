import unittest
from tzd.rl.parsing import extract_countdown_answer

class TestRLIntegration(unittest.TestCase):
    def test_extract_answer_clean(self):
        output = "<think>...</think>\n<answer>1 + 2</answer>"
        self.assertEqual(extract_countdown_answer(output), "1 + 2")

    def test_extract_answer_with_eos_same_line(self):
        output = "<think>...</think>\n<answer>1 + 2</answer><|eot_id|>"
        self.assertEqual(extract_countdown_answer(output), "1 + 2")

    def test_extract_answer_with_eos_next_line(self):
        # This is expected to FAIL with current logic (it only checks last line)
        output = "<think>...</think>\n<answer>1 + 2</answer>\n<|eot_id|>"
        extracted = extract_countdown_answer(output)
        print(f"Next line EOS result: {extracted}") 
        # deciding if failure is bug or feature. For now just observation.

    def test_extract_answer_multiline(self):
        # Should probably fail if parser expects single line
        output = "<answer>\n1 + 2\n</answer>"
        extracted = extract_countdown_answer(output)
        print(f"Multiline result: {extracted}")

    def test_extract_answer_with_text_after(self):
        # Case where model keeps talking
        output = "<answer>1 + 2</answer>\nI hope that helps!"
        extracted = extract_countdown_answer(output)
        print(f"Text after result: {extracted}")

if __name__ == '__main__':
    unittest.main()
