import random
import json
import argparse
from typing import List, Tuple, Optional
import itertools

def solve_countdown(numbers: List[int], target: int) -> Optional[Tuple[str, str]]:
    """
    Find a solution for the countdown problem.
    Returns (expression, reasoning_steps) or None.
    """
    ops = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: x // y if y != 0 and x % y == 0 else None
    }

    # We need to use all numbers exactly once? 
    # SPG prompt says: "each number must be used exactly once"
    # But standard countdown allows using subset.
    # Let's assume we must use all for now, or at least try to.
    # Actually, for 3-4 numbers, it's easy to use all.
    
    # Recursive search
    # State: list of (value, expression_string)
    
    initial_state = [(n, str(n)) for n in numbers]
    
    def search(current_state):
        if len(current_state) == 1:
            val, expr = current_state[0]
            if val == target:
                return expr, []
            return None

        # Try combining any two numbers
        for i in range(len(current_state)):
            for j in range(len(current_state)):
                if i == j: continue
                
                v1, e1 = current_state[i]
                v2, e2 = current_state[j]
                
                # Optimization: commutative ops
                if i > j: # Avoid duplicates for + and *
                    pass 
                
                remaining = [current_state[k] for k in range(len(current_state)) if k != i and k != j]
                
                for op_sym, op_func in ops.items():
                    # Optimization: Order matters for - and /
                    # For + and *, we can enforce order to reduce search space
                    if op_sym in ['+', '*'] and i > j:
                        continue
                        
                    res = op_func(v1, v2)
                    if res is not None:
                        # Valid operation
                        new_expr = f"({e1} {op_sym} {e2})"
                        new_state = remaining + [(res, new_expr)]
                        
                        result = search(new_state)
                        if result:
                            expr, steps = result
                            # Add this step to reasoning
                            step_str = f"{e1} {op_sym} {e2} = {res}"
                            return expr, [step_str] + steps
        return None

    # Try to find solution using all numbers
    # Note: The recursive search above naturally uses all numbers because it stops when len==1
    return search(initial_state)

def generate_dataset(num_samples: int, output_file: str, min_nums: int = 3, max_nums: int = 4):
    data = []
    count = 0
    
    print(f"Generating {num_samples} samples...")
    
    while count < num_samples:
        n_nums = random.randint(min_nums, max_nums)
        numbers = [random.randint(1, 20) for _ in range(n_nums)]
        
        # Generate a reachable target by simulating operations
        # This ensures a solution exists
        current = list(numbers)
        while len(current) > 1:
            a = current.pop(random.randint(0, len(current)-1))
            b = current.pop(random.randint(0, len(current)-1))
            op = random.choice(['+', '-', '*']) # Avoid division for target generation to keep it integer/simple
            if op == '+': res = a + b
            elif op == '-': res = a - b
            elif op == '*': res = a * b
            current.append(res)
        
        target = current[0]
        
        # Now solve it to get the canonical solution/reasoning
        # (The random generation path might be one solution, but we want to verify)
        sol = solve_countdown(numbers, target)
        
        if sol:
            expr, steps = sol
            # Format reasoning
            reasoning = " ".join([f"{s}." for s in steps])
            # Remove outer parens from expr if present
            if expr.startswith("(") and expr.endswith(")"):
                expr = expr[1:-1]
            
            entry = {
                "input": ",".join(map(str, numbers)),
                "output": str(target),
                "solution": expr,
                "reasoning": reasoning
            }
            data.append(entry)
            count += 1
            if count % 100 == 0:
                print(f"Generated {count}/{num_samples}")
    
    with open(output_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    generate_dataset(1000, "data/countdown/countdown_sft_train.jsonl")
    generate_dataset(100, "data/countdown/countdown_sft_val.jsonl")
