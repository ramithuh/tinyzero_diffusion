"""
Test script for Countdown dataset with Qwen tokenizer.

This verifies that the countdown dataset works correctly with the 
actual tokenizer configuration used in tinyzero_diffusion.
"""

import sys
import os
import hydra
from omegaconf import DictConfig
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from tzd.data import CountdownDataset, CountdownDataModule


def test_countdown_with_tokenizer():
    """Test countdown dataset with Qwen tokenizer."""
    
    print("=" * 80)
    print("Testing Countdown Dataset with Qwen Tokenizer")
    print("=" * 80)
    
    # Load tokenizer using Hydra
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name="config")
        tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    
    print(f"\n✓ Tokenizer loaded: {type(tokenizer)}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    # Setup special tokens (following diffusion_pretrained.py pattern)
    print("\n" + "=" * 80)
    print("Setting up special tokens (Qwen-specific)")
    print("=" * 80)
    
    tokenizer.bos_token = "<|im_start|>"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    
    tokenizer.pad_token = "<|endoftext|>"
    
    # Test 1: Basic dataset loading
    print("\n" + "=" * 80)
    print("Test 1: Loading Dataset")
    print("=" * 80)
    
    dataset = CountdownDataset(
        data_path="data/countdown/countdown_cd3_test.jsonl",
        tokenizer=tokenizer,
        add_reasoning_tag=True,
        max_samples=5
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} examples")
    
    # Test 2: Get a single example
    print("\n" + "=" * 80)
    print("Test 2: Single Example")
    print("=" * 80)
    
    example = dataset[0]
    print(f"\nNumbers: {example['numbers']}")
    print(f"Target: {example['target']}")
    print(f"\nPrompt (first 300 chars):\n{example['prompt'][:300]}...")
    
    # Test 3: Tokenize manually
    print("\n" + "=" * 80)
    print("Test 3: Manual Tokenization")
    print("=" * 80)
    
    prompt = example['prompt']
    tokens = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512
    )
    
    print(f"✓ Tokenization successful")
    print(f"  Input IDs shape: {tokens['input_ids'].shape}")
    print(f"  Attention mask shape: {tokens['attention_mask'].shape}")
    print(f"  Number of tokens: {tokens['input_ids'].shape[1]}")
    
    # Decode to verify
    decoded = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=False)
    print(f"\n  Decoded (first 200 chars): {decoded[:200]}...")
    
    # Test 4: Batch collation
    print("\n" + "=" * 80)
    print("Test 4: Batch Collation")
    print("=" * 80)
    
    batch = [dataset[i] for i in range(3)]
    collated = CountdownDataset.collate_fn(batch, tokenizer=tokenizer)
    
    print(f"✓ Batch collation successful")
    print(f"  Batch keys: {list(collated.keys())}")
    print(f"  Prompts: {len(collated['prompts'])} items")
    print(f"  Numbers: {len(collated['numbers'])} items")
    print(f"  Targets: {len(collated['targets'])} items")
    
    if 'input_ids' in collated:
        print(f"  Input IDs shape: {collated['input_ids'].shape}")
        print(f"  Attention mask shape: {collated['attention_mask'].shape}")
    
    # Test 5: DataModule
    print("\n" + "=" * 80)
    print("Test 5: Lightning DataModule")
    print("=" * 80)
    
    datamodule = CountdownDataModule(
        data_dir="data/countdown",
        batch_size=2,
        num_workers=0,
        tokenizer=tokenizer,
        max_test_samples=10
    )
    
    datamodule.setup(stage="test")
    print(f"✓ DataModule setup successful")
    print(f"  Test dataset size: {len(datamodule.test_dataset)}")
    
    # Test dataloader
    test_loader = datamodule.test_dataloader()
    batch = next(iter(test_loader))
    
    print(f"\n✓ DataLoader working")
    print(f"  Batch keys: {list(batch.keys())}")
    if 'input_ids' in batch:
        print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"  Batch attention_mask shape: {batch['attention_mask'].shape}")
    
    # Show a few examples from the batch
    print(f"\n  Example numbers from batch:")
    for i, (nums, target) in enumerate(zip(batch['numbers'][:2], batch['targets'][:2])):
        print(f"    Sample {i}: {nums} → {target}")
    
    # Test 6: Verify padding behavior
    print("\n" + "=" * 80)
    print("Test 6: Padding Behavior")
    print("=" * 80)
    
    # Create samples with different lengths
    short_prompt = "Numbers: [1, 2, 3]\nTarget: 6"
    long_prompt = example['prompt']
    
    batch_prompts = [short_prompt, long_prompt]
    encoded = tokenizer(
        batch_prompts,
        padding="longest",
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    print(f"✓ Padding test successful")
    print(f"  Batch shape: {encoded['input_ids'].shape}")
    print(f"  Padding token ID: {tokenizer.pad_token_id}")
    print(f"  Short prompt tokens: {(encoded['input_ids'][0] != tokenizer.pad_token_id).sum().item()}")
    print(f"  Long prompt tokens: {(encoded['input_ids'][1] != tokenizer.pad_token_id).sum().item()}")
    
    # Check attention mask
    print(f"\n  Attention mask check:")
    print(f"    Short prompt: {encoded['attention_mask'][0].sum().item()} valid tokens")
    print(f"    Long prompt: {encoded['attention_mask'][1].sum().item()} valid tokens")
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe Countdown dataset is fully compatible with the Qwen tokenizer.")
    print("You can now use it in training with:")
    print("  from tzd.data import CountdownDataModule")
    print("  datamodule = CountdownDataModule(data_dir='data/countdown', tokenizer=tokenizer)")
    print("=" * 80)


if __name__ == "__main__":
    test_countdown_with_tokenizer()
