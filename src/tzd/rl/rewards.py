"""Reward functions for Countdown task."""
import re
from tzd.rl.parsing import extract_countdown_answer


def validate_equation(equation_str: str, available_numbers: list[int]) -> bool:
    """
    Check if equation uses only available numbers, each exactly once.

    Args:
        equation_str: E.g., "3+5*2"
        available_numbers: E.g., [3, 5, 2]

    Returns:
        True if valid, False otherwise
    """
    try:
        # Extract all numbers from equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

        # Check if same numbers (order doesn't matter)
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except (ValueError, TypeError):
        return False


def evaluate_equation(equation_str: str) -> float | None:
    """
    Safely evaluate arithmetic equation.

    Args:
        equation_str: E.g., "3+5*2"

    Returns:
        Result (13.0) or None if invalid
    """
    try:
        # Only allow safe characters
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation")

        # Evaluate with restricted builtins
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except (SyntaxError, NameError, TypeError, ZeroDivisionError):
        return None


def compute_countdown_score(
    solution_str: str,
    target: int,
    numbers: list[int],
    format_score: float = 0.1,
    correct_score: float = 1.0,
    verbose: bool = False
) -> float:
    """
    Score a countdown solution.

    Args:
        solution_str: Model output with <answer> tags
        target: Target number (e.g., 13)
        numbers: Available numbers (e.g., [3, 5, 2])
        format_score: Reward for valid format but wrong answer
        correct_score: Reward for correct answer
        verbose: Print debug info

    Returns:
        Score: 0.0 (no answer), 0.1 (format ok), or 1.0 (correct)
    """
    equation = extract_countdown_answer(solution_str)

    if verbose:
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted: {equation}")

    # No equation found
    if equation is None:
        if verbose:
            print("❌ No equation found")
        return 0.0

    # Check if uses correct numbers
    if not validate_equation(equation, numbers):
        if verbose:
            print("❌ Invalid equation (wrong numbers)")
        return format_score

    # Evaluate equation
    result = evaluate_equation(equation)
    if result is None:
        if verbose:
            print("❌ Could not evaluate equation")
        return format_score

    # Check if matches target
    if abs(result - target) < 1e-5:  # Float precision
        if verbose:
            print(f"✅ Correct: {equation} = {result}")
        return correct_score
    else:
        if verbose:
            print(f"❌ Wrong result: {equation} = {result} ≠ {target}")
        return format_score


def countdown_reward_batch(
    completions: list[str],
    targets: list[int],
    numbers: list[list[int]]
) -> list[float]:
    """
    Compute rewards for a batch of completions.

    Args:
        completions: List of model outputs
        targets: List of target numbers
        numbers: List of available number lists

    Returns:
        List of scores (same length as completions)
    """
    scores = []
    for completion, target, nums in zip(completions, targets, numbers):
        score = compute_countdown_score(
            completion,
            target,
            nums,
            verbose=False  # Set True for debugging
        )
        scores.append(score)

    return scores


# Test it:
if __name__ == "__main__":
    # Test case 1: Correct answer
    output1 = "<reasoning>\n5*2=10, 3+10=13\n</reasoning>\n<answer>3+5*2</answer>"
    score1 = compute_countdown_score(output1, target=13, numbers=[3, 5, 2], verbose=True)
    assert score1 == 1.0, "Should be correct!"

    # Test case 2: Wrong answer
    output2 = "<answer>3+5+2</answer>"  # = 10, not 13
    score2 = compute_countdown_score(output2, target=13, numbers=[3, 5, 2], verbose=True)
    assert score2 == 0.1, "Should get format score only"

    # Test case 3: Invalid numbers
    output3 = "<answer>3+7</answer>"  # 7 not available
    score3 = compute_countdown_score(output3, target=10, numbers=[3, 5, 2], verbose=True)
    assert score3 == 0.1, "Should get format score only"

    print("\n✅ All tests passed!")
