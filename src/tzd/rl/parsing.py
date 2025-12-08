"""Answer parsing utilities for Countdown task."""
import re


def extract_countdown_answer(solution_str: str) -> str | None:
    """
    Extract the equation from model output.

    Expected format:
        <reasoning>
        ... thinking process ...
        </reasoning>
        <answer>
        3 + 5 * 2
        </answer>

    Args:
        solution_str: Model's generated completion

    Returns:
        Extracted equation string or None if not found
    """
    # If full text is provided, try to split by assistant marker to avoid matching examples in prompt
    if "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant")[-1]
    elif "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:")[-1]

    # Robust parsing: Find the last <answer> block in the text
    # DOTALL allows . to match newlines
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)

    if matches:
        # Return last match found
        answer = matches[-1].strip()
        # Strip \boxed{...} if present
        if answer.startswith(r"\boxed{") and answer.endswith("}"):
            answer = answer[7:-1]
        return answer

    return None


# Test it:
if __name__ == "__main__":
    test_output = """<reasoning>
    Target is 13 using [3, 5, 2].
    First: 5 * 2 = 10
    Then: 3 + 10 = 13
    </reasoning>
    <answer>3+5*2</answer>"""

    equation = extract_countdown_answer(test_output)
    print(f"Extracted: {equation}")  # → "3+5*2"
