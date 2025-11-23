"""
Lightning DataModule for the Countdown task.
"""

import os
from typing import Optional, List
import torch
import lightning as L
from torch.utils.data import DataLoader

from tzd.data.countdown import CountdownDataset


class CountdownDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for the Countdown task.
    
    This provides a standardized interface for loading train/val/test datasets
    and creating dataloaders for the countdown reasoning task.
    """
    
    def __init__(
        self,
        data_dir: str = "data/countdown",
        batch_size: int = 4,
        num_workers: int = 0,
        tokenizer: Optional[callable] = None,
        train_file: str = "countdown_cd3_train.jsonl",
        val_file: str = "countdown_cd3_val.jsonl",
        test_file: str = "countdown_cd3_test.jsonl",
        add_reasoning_tag: bool = True,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the Countdown DataModule.
        
        Args:
            data_dir: Directory containing countdown dataset files
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
            tokenizer: Tokenizer for encoding prompts (optional)
            train_file: Name of training data file
            val_file: Name of validation data file
            test_file: Name of test data file
            add_reasoning_tag: Whether to prefill <reasoning> tag in prompts
            max_train_samples: Maximum training samples to load (None = all)
            max_val_samples: Maximum validation samples to load (None = all)
            max_test_samples: Maximum test samples to load (None = all)
        """
        super().__init__()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.add_reasoning_tag = add_reasoning_tag
        
        # File paths
        self.train_file = os.path.join(data_dir, train_file)
        self.val_file = os.path.join(data_dir, val_file)
        self.test_file = os.path.join(data_dir, test_file)
        
        # Sample limits
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        
        # Datasets (initialized in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        """
        Download or prepare data if needed.
        Called only on 1 GPU/TPU in distributed mode.
        """
        # Check if data files exist
        if not os.path.exists(self.test_file):
            print(f"Warning: Test file not found at {self.test_file}")
            print("You may need to download the countdown dataset.")
            print("Run: from tzd.data import download_countdown_dataset; download_countdown_dataset('data/countdown')")
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training, validation, and testing.
        Called on every GPU in distributed mode.
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage == "fit" or stage is None:
            # Training dataset
            if os.path.exists(self.train_file):
                self.train_dataset = CountdownDataset(
                    data_path=self.train_file,
                    tokenizer=self.tokenizer,
                    add_reasoning_tag=self.add_reasoning_tag,
                    max_samples=self.max_train_samples
                )
            else:
                # Fallback: use test set for training (for quick experiments)
                print(f"Warning: {self.train_file} not found. Using test set for training.")
                self.train_dataset = CountdownDataset(
                    data_path=self.test_file,
                    tokenizer=self.tokenizer,
                    add_reasoning_tag=self.add_reasoning_tag,
                    max_samples=self.max_train_samples
                )
            
            # Validation dataset
            if os.path.exists(self.val_file):
                self.val_dataset = CountdownDataset(
                    data_path=self.val_file,
                    tokenizer=self.tokenizer,
                    add_reasoning_tag=self.add_reasoning_tag,
                    max_samples=self.max_val_samples
                )
            else:
                # Fallback: use subset of test set
                print(f"Warning: {self.val_file} not found. Using test set for validation.")
                self.val_dataset = CountdownDataset(
                    data_path=self.test_file,
                    tokenizer=self.tokenizer,
                    add_reasoning_tag=self.add_reasoning_tag,
                    max_samples=self.max_val_samples or 50
                )
        
        if stage == "test" or stage is None:
            # Test dataset
            self.test_dataset = CountdownDataset(
                data_path=self.test_file,
                tokenizer=self.tokenizer,
                add_reasoning_tag=self.add_reasoning_tag,
                max_samples=self.max_test_samples
            )
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=lambda batch: CountdownDataset.collate_fn(batch, tokenizer=self.tokenizer),
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=lambda batch: CountdownDataset.collate_fn(batch, tokenizer=self.tokenizer),
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=lambda batch: CountdownDataset.collate_fn(batch, tokenizer=self.tokenizer),
            pin_memory=True,
        )
    
    def get_sample_prompt(self, split: str = "test", idx: int = 0) -> str:
        """
        Get a sample prompt from the dataset.
        
        Args:
            split: 'train', 'val', or 'test'
            idx: Index of the sample
            
        Returns:
            Formatted prompt string
        """
        if split == "train" and self.train_dataset:
            return self.train_dataset[idx]["prompt"]
        elif split == "val" and self.val_dataset:
            return self.val_dataset[idx]["prompt"]
        elif split == "test" and self.test_dataset:
            return self.test_dataset[idx]["prompt"]
        else:
            raise ValueError(f"Invalid split '{split}' or dataset not initialized")
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Decode a sequence of tokens into a string."""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


if __name__ == "__main__":
    # Quick test
    print("Testing CountdownDataModule...")
    
    datamodule = CountdownDataModule(
        data_dir="../../data/countdown",
        batch_size=2,
        num_workers=0,
    )
    
    # Setup
    datamodule.setup(stage="fit")
    
    # Get a sample
    if datamodule.test_dataset:
        print(f"\nTest dataset size: {len(datamodule.test_dataset)}")
        sample = datamodule.get_sample_prompt(split="test", idx=0)
        print(f"\nSample prompt:\n{sample[:200]}...")
    
    print("\nCountdownDataModule test completed!")
