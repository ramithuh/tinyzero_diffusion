"""
Countdown task dataset for diffusion language models.

The countdown task involves creating an arithmetic expression using provided numbers
to reach a target value. This follows the SPG format with <reasoning> and <answer> tags.
"""

import json
import os
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset


class CountdownDataset(Dataset):
    """
    Dataset for the Countdown task following SPG's format.
    
    Task: Given a target number and a list of numbers, create an arithmetic expression
    that uses each number exactly once to reach the target.
    
    Format:
        Prompt: System instructions + "Numbers: [a, b, c]\nTarget: X"
        Expected output: <reasoning>...</reasoning>\n<answer>\\boxed{expression}</answer>
    """
    
    SYSTEM_PROMPT = (
        "Using only the provided numbers, create an arithmetic expression that evaluates to exactly "
        "the provided target number. You may use the operations +, -, *, and / as needed, but each "
        "number must be used exactly once. Think step-by-step. After reasoning, provide only your "
        "final expression inside \\boxed{} tags without including an equals sign or the target number. "
        "For example: \\boxed{a + b * c}\n"
        "Respond in the following format:\n"
        "<reasoning>\n"
        "Your reasoning here\n"
        "</reasoning>\n"
        "<answer>\n"
        "\\boxed{...}\n"
        "</answer>"
    )
    
    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        add_reasoning_tag: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Initialize the Countdown dataset.
        
        Args:
            data_path: Path to JSONL file with countdown examples
            tokenizer: Tokenizer for the model (optional, for compatibility)
            add_reasoning_tag: Whether to prefill <reasoning> tag in prompt
            max_samples: Maximum number of samples to load (None = all)
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.add_reasoning_tag = add_reasoning_tag
        self.examples = self._load_data(data_path, max_samples)
        
    def _load_data(self, data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
        """Load countdown examples from JSONL file."""
        examples = []
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Countdown dataset not found at {data_path}")
        
        with open(data_path, 'r') as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break
                example = json.loads(line)
                examples.append(example)
        
        print(f"Loaded {len(examples)} countdown examples from {data_path}")
        return examples
    
    def _format_prompt(self, numbers: List[int], target: int) -> str:
        """
        Format the countdown prompt following SPG's template.
        """
        # Create the question
        question = f"Numbers: {numbers}\nTarget: {target}"
        
        # Combine system prompt and question
        full_prompt = f"{self.SYSTEM_PROMPT}\n\n{question}"
        
        # Optionally prefill <reasoning> tag to encourage structured output
        if self.add_reasoning_tag:
            full_prompt += "\n<reasoning>"
        
        return full_prompt
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)

    def _format_completion(self, reasoning: str, solution: str) -> str:
        """Format the completion (reasoning + answer)."""
        # If prompt already has <reasoning>, we start with the content
        # reasoning should be the content inside tags
        # solution should be the content inside \boxed{}
        
        return f"\n{reasoning}\n</reasoning>\n<answer>\n\\boxed{{{solution}}}\n</answer>"

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single example.
        """
        example = self.examples[idx]
        
        # Parse the example
        numbers_str = example["input"]
        target_str = example["output"]
        
        # Convert to integers
        numbers = [int(n.strip()) for n in numbers_str.split(",")]
        target = int(target_str)
        
        # Format the prompt
        prompt = self._format_prompt(numbers, target)
        
        # If training data (solution/reasoning) is available, format full text
        full_text = prompt
        if "solution" in example and "reasoning" in example:
            completion = self._format_completion(example["reasoning"], example["solution"])
            full_text = prompt + completion
        
        return {
            "prompt": prompt,
            "full_text": full_text,
            "numbers": numbers,
            "target": target,
            "raw_example": example
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict], tokenizer=None):
        """
        Collate function for DataLoader with prompt masking.
        """
        prompts = [item["prompt"] for item in batch]
        full_texts = [item["full_text"] for item in batch]
        numbers = [item["numbers"] for item in batch]
        targets = [item["target"] for item in batch]
        
        batch_dict = {
            "prompts": prompts,
            "numbers": numbers,
            "targets": targets,
        }
        
        # If tokenizer provided, tokenize the full texts
        if tokenizer is not None:
            # Tokenize full texts (prompt + completion)
            encoded = tokenizer(
                full_texts,
                padding="longest",
                return_tensors="pt",
                truncation=True
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            
            batch_dict["input_ids"] = input_ids
            batch_dict["attention_mask"] = attention_mask
            
            # Create loss mask (0 for prompt tokens, 1 for completion tokens)
            loss_mask = torch.ones_like(input_ids)
            
            # Mask out padding tokens
            loss_mask = loss_mask * attention_mask
            
            # Mask out prompt tokens
            # We need to find the length of the prompt for each example
            # Since we don't have the exact token count of the prompt part within the full tokenization,
            # we can approximate it or tokenize the prompt separately.
            # Tokenizing separately is safer.
            
            prompt_encoded = tokenizer(
                prompts,
                padding=False, # Don't pad here, we just want lengths
                add_special_tokens=True # Should match how full_texts was tokenized
            )
            
            for i, prompt_ids in enumerate(prompt_encoded["input_ids"]):
                # The prompt corresponds to the first len(prompt_ids) tokens
                # Note: This assumes tokenizer(A+B) starts with tokenizer(A). 
                # This is generally true for sentencepiece/BPE unless there's a merge at the boundary.
                # Given the prompt ends with "\n<reasoning>", it's likely a clean boundary.
                prompt_len = len(prompt_ids)
                
                # Ensure we don't mask everything if prompt_len >= seq_len
                prompt_len = min(prompt_len, input_ids.size(1))
                
                loss_mask[i, :prompt_len] = 0
            
            batch_dict["loss_mask"] = loss_mask
        
        return batch_dict


def download_countdown_dataset(save_dir: str) -> str:
    """
    Download the countdown dataset from SPG repository or HuggingFace.
    
    Args:
        save_dir: Directory to save the dataset
        
    Returns:
        Path to the downloaded dataset file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Option 1: Try to load from HuggingFace
    try:
        from datasets import load_dataset
        print("Attempting to download countdown dataset from HuggingFace...")
        
        # Load the countdown dataset (cd3 = 3-4 numbers)
        dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')
        
        # Save to JSONL format matching SPG's format
        output_path = os.path.join(save_dir, "countdown_cd3.jsonl")
        
        with open(output_path, 'w') as f:
            for example in dataset:
                # Convert to SPG format: {"input": "num1,num2,num3", "output": "target"}
                numbers = ','.join(map(str, example['nums']))
                target = str(example['target'])
                f.write(json.dumps({"input": numbers, "output": target}) + '\n')
        
        print(f"Downloaded {len(dataset)} examples to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error downloading from HuggingFace: {e}")
        print("Please manually download the dataset from SPG repository")
        return None


if __name__ == "__main__":
    # Example usage and testing
    print("Testing CountdownDataset...")
    
    # Create a sample dataset file for testing
    test_file = "/tmp/test_countdown.jsonl"
    with open(test_file, 'w') as f:
        f.write('{"input": "30,100,93", "output": "23"}\n')
        f.write('{"input": "83,18,75", "output": "10"}\n')
        f.write('{"input": "76,41,59", "output": "24"}\n')
    
    # Test dataset
    dataset = CountdownDataset(test_file)
    
    print(f"\nDataset size: {len(dataset)}")
    print("\nExample 0:")
    example = dataset[0]
    print(f"Numbers: {example['numbers']}")
    print(f"Target: {example['target']}")
    print(f"\nPrompt:\n{example['prompt']}")
    
    # Test collate function
    batch = [dataset[i] for i in range(2)]
    collated = CountdownDataset.collate_fn(batch)
    print(f"\nCollated batch keys: {collated.keys()}")
    print(f"Batch size: {len(collated['prompts'])}")
    
    # Clean up
    os.remove(test_file)
    print(f"Batch size: {len(collated['prompts'])}")
    
    # Clean up
    os.remove(test_file)
    print("\nTest completed successfully!")
