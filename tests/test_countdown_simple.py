"""
Simple test for Countdown dataset with Qwen tokenizer.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from transformers import AutoTokenizer
from tzd.data import CountdownDataset, CountdownDataModule

def test_countdown():
    print("="*80)
    print("Countdown Dataset Test with Qwen Tokenizer")
    print("="*80)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("checkpoints/Qwen/Qwen2.5-3B")
    
    # Setup special tokens (following diffusion_pretrained.py)
    print("2. Setting up special tokens...")
    tokenizer.bos_token = "<|im_start|>"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    tokenizer.pad_token = "<|endoftext|>"
