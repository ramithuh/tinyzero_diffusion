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
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)

    if matches:
        # Return last match (in case multiple <answer> tags)
        return matches[-1].strip()

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
