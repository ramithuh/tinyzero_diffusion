"""
Test script to compare TinyZero reward logic with ours.
"""
import re
import pytest
from tzd.rl.parsing import extract_countdown_answer
from tzd.rl.rewards import compute_countdown_score

# =============================================================================
# TinyZero Implementation (Copied from TinyZero/verl/utils/reward_score/countdown.py)
# =============================================================================

def tinyzero_extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    
    # CRITICAL: TinyZero seems to only look at the last line?
    # This line was in the file I read:
    # solution_str = solution_str.split('\n')[-1]
    # But let's verify if that's actually what we want to test against.
    # If the user's example has newlines, this might break.
    # Let's include it because it's in the source file.
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer

def tinyzero_validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False

def tinyzero_evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None

def tinyzero_compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = tinyzero_extract_solution(solution_str=solution_str)
    
    if equation is None:
        return 0
    
    # Validate equation uses correct numbers
    if not tinyzero_validate_equation(equation, numbers):
        return format_score
        
    # Evaluate equation
    try:
        result = tinyzero_evaluate_equation(equation)
        if result is None:
            return format_score
            
        if abs(result - target) < 1e-5:
            return score
        else:
            return format_score
    except:
        return format_score

# =============================================================================
# Tests
# =============================================================================

def test_user_example_alignment():
    """
    Test the specific example provided by the user.
    """
    sample_text = """system
You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.
user
Using the numbers [79, 17, 60], create an equation that equals 36. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
assistant
Let me solve this step by step.
<think> We have the numbers 79, 17, and 60. We want to use them only once and use each number once in the equation. One possible solution is: </think>
<answer>60 * (17 - 79 + 17) / 3</answer>"""

    target = 36
    numbers = [79, 17, 60]
    
    # 1. Run TinyZero Logic
    ground_truth = {'target': target, 'numbers': numbers}
    tiny_score = tinyzero_compute_score(sample_text, ground_truth)
    
    # 2. Run Our Logic
    our_score = compute_countdown_score(sample_text, target, numbers)
    
    print(f"\nTinyZero Score: {tiny_score}")
    print(f"Our Score: {our_score}")
    
    # 3. Analyze why
    # TinyZero splits by "Assistant:" or "<|im_start|>assistant".
    # The sample text uses "assistant" (lowercase).
    # TinyZero's extract_solution should return None because it doesn't find the marker.
    # So TinyZero score should be 0.
    
    # Our logic splits by "Assistant:" or "<|im_start|>assistant".
    # It also doesn't find the marker.
    # BUT, our logic falls back to searching the whole string (if I recall correctly).
    # Let's check parsing.py:
    # if "<|im_start|>assistant" in solution_str: ...
    # elif "Assistant:" in solution_str: ...
    # (No else block that returns None)
    # Then it runs re.findall on solution_str.
    # So our logic SHOULD find the answer.
    
    # However, the answer is "60 * (17 - 79 + 17) / 3".
    # Numbers used: 60, 17, 79, 17.
    # Available: 79, 17, 60.
    # 17 is used twice.
    # So validate_equation should return False.
    # So our score should be 0.1 (format_score).
    
    # So: TinyZero -> 0.0 (Parsing failed)
    # Ours -> 0.1 (Parsing succeeded, validation failed)
    
    # This is a MISMATCH!
    assert tiny_score == our_score, f"Mismatch! TinyZero: {tiny_score}, Ours: {our_score}"

def test_lowercase_assistant_handling():
    """Test if our parser handles lowercase 'assistant' better (or worse) than TinyZero."""
    text = "assistant\n<answer>1+1</answer>"
    
    # TinyZero
    tiny_extracted = tinyzero_extract_solution(text)
    # Ours
    our_extracted = extract_countdown_answer(text)
    
    print(f"\nTinyZero Extracted: {tiny_extracted}")
    print(f"Our Extracted: {our_extracted}")
    
    # TinyZero should be None
    assert tiny_extracted is None
    
    # Ours should be "1+1" because we don't enforce the split if marker is missing
    assert our_extracted == "1+1"

if __name__ == "__main__":
    # Run tests manually
    try:
        test_user_example_alignment()
        print("User example test passed!")
    except AssertionError as e:
        print(f"User example test failed: {e}")
        
    try:
        test_lowercase_assistant_handling()
        print("Lowercase assistant test passed!")
    except AssertionError as e:
        print(f"Lowercase assistant test failed: {e}")

def test_missing_tags_in_completion():
    """
    Verify that if we pass ONLY the completion (no prompt) and it lacks tags,
    we get None (and thus 0.0 reward), even if the prompt HAD tags.
    """
    # Simulate completion where model failed to output tags
    completion = "I cannot solve this."
    
    extracted = extract_countdown_answer(completion)
    assert extracted is None, f"Expected None, got {extracted}"
    
    score = compute_countdown_score(completion, 13, [3, 5, 2])
    assert score == 0.0, f"Expected 0.0, got {score}"
    print("\nMissing tags test passed!")

if __name__ == "__main__":
    # Run tests manually
    try:
        test_user_example_alignment()
        print("User example test passed!")
    except AssertionError as e:
        print(f"User example test failed: {e}")
        
    try:
        test_lowercase_assistant_handling()
        print("Lowercase assistant test passed!")
    except AssertionError as e:
        print(f"Lowercase assistant test failed: {e}")

    try:
        test_missing_tags_in_completion()
    except AssertionError as e:
        print(f"Missing tags test failed: {e}")

def test_multiple_answer_tags():
    """
    Verify behavior when multiple <answer> tags are present.
    It should take the LAST one.
    """
    completion = """
Using the numbers 79, 17, and 60:
<answer>17 - 60 + 79 </answer>
The equation:
<answer> -43 + 79 </answer> equals 36.
"""
    # Expected behavior: Extract "-43 + 79"
    extracted = extract_countdown_answer(completion)
    print(f"\nMultiple tags extracted: {extracted}")
    assert extracted == "-43 + 79"
    
    # Reward check
    # Target: 36, Numbers: [79, 17, 60]
    # "-43 + 79" uses numbers [-43, 79]
    # -43 is NOT in [79, 17, 60]
    # So validation should fail.
    score = compute_countdown_score(completion, 36, [79, 17, 60])
    print(f"Multiple tags score: {score}")
    assert score == 0.1 # Format score only
    print("Multiple tags test passed!")

if __name__ == "__main__":
    # Run tests manually
    try:
        test_user_example_alignment()
        print("User example test passed!")
    except AssertionError as e:
        print(f"User example test failed: {e}")
        
    try:
        test_lowercase_assistant_handling()
        print("Lowercase assistant test passed!")
    except AssertionError as e:
        print(f"Lowercase assistant test failed: {e}")

    try:
        test_missing_tags_in_completion()
    except AssertionError as e:
        print(f"Missing tags test failed: {e}")

    try:
        test_multiple_answer_tags()
    except AssertionError as e:
        print(f"Multiple tags test failed: {e}")
