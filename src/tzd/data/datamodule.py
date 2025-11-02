import os
import requests
import numpy as np
from typing import Optional, Dict, List

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn.utils.rnn import pad_sequence


class TinyShakespeareDataset(Dataset):
    """Character-level Shakespeare dataset for diffusion language modeling."""

    def __init__(self,
                 data_dir: str = "data/shakespeare",
                 block_size: int = 256,
                 tokenizer: callable = None,
                 **kwargs):
        """
        Initialize the Shakespeare dataset.

        Args:
            data_dir: Directory to store/load the text file
            block_size: Maximum sequence length. If black_size == -1, then random sample length
            tokenizer: Tokenizer to use for encoding text
        """

        self.tokenizer = tokenizer
        self.block_size = block_size - 1  # Reserve one token for the <bos> token
        self.data_dir = data_dir

        # Download and load the text
        self.text = self._load_shakespeare_data()
        self.data = []

        tokens = tokenizer.encode(self.text, add_special_tokens=False)
        self.data = [[tokenizer.bos_token_id] + tokens[i:i + self.block_size]
                     for i in range(0, len(tokens) - self.block_size, self.block_size)]

    def _load_shakespeare_data(self) -> str:
        """Download and load the tinyshakespeare dataset."""
        os.makedirs(self.data_dir, exist_ok=True)
        file_path = os.path.join(self.data_dir, "input.txt")

        # Download if not exists
        if not os.path.exists(file_path):
            print(f"Downloading tinyshakespeare to {file_path}...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            response = requests.get(url)
            response.raise_for_status()

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("Download completed!")

        # Load the text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        return text

    def __len__(self):
        """Number of sequences in the dataset."""
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get a sequence for training.

        Returns:
            Dict with 'input_ids' tensor of shape (block_size,)
        """
        return {"input_ids": torch.LongTensor(self.data[idx])}


class TinyShakespeareDataModule(L.LightningDataModule):
    """Lightning DataModule for TinyShakespeare dataset."""

    def __init__(self,
                 batch_size: int,
                 num_workers: int,
                 block_size: int,
                 data_dir: str = "data/shakespeare",
                 tokenizer: Optional[callable] = None,
                 **kwargs):
        """
        Initialize the DataModule.

        Args:
            batch_size: Training batch size
            num_workers: Number of data loading workers
            block_size: Maximum sequence length
            data_dir: Directory to store the dataset
            tokenizer: Tokenizer for encoding text
        """
        super().__init__()

        if tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-30b")
        else:
            self.tokenizer = tokenizer

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.block_size = block_size
        self.data_dir = data_dir
        self.vocab_size = self.tokenizer.vocab_size

        # Initialize the dataset and splits
        self.splits = {'train': [], 'val': [], 'test': []}
        self.dataset, self.train_dataset, self.val_dataset, self.test_dataset = None, None, None, None

    def setup(self, stage: Optional[str] = None, splits: Optional[Dict[str, List[int]]] = None):
        """Setup datasets for training, validation, and testing."""

        self.dataset = TinyShakespeareDataset(
            data_dir=self.data_dir,
            block_size=self.block_size,
            tokenizer=self.tokenizer
        )

        if splits is None:  # perform random splits if no splits provided
            random_inds = np.random.permutation(len(self.dataset))
            self.splits['train'] = list(random_inds[:int(0.8 * len(self.dataset))])
            self.splits['val'] = list(random_inds[int(0.8 * len(self.dataset)):int(0.9 * len(self.dataset))])
            self.splits['test'] = list(random_inds[int(0.9 * len(self.dataset)):])
        else:
            self.splits = splits

        if stage == "fit" or stage is None:
            # Create training dataset
            self.train_dataset = Subset(self.dataset, self.splits['train'])

            # Create validation dataset
            self.val_dataset = Subset(self.dataset, self.splits['val'])

        if stage == "test" or stage is None:
            # Create test dataset
            self.test_dataset = Subset(self.dataset, self.splits['test'])

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def decode(self, tokens: torch.Tensor) -> str:
        """Decode a sequence of tokens into a string."""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def collate_fn(batch, padding_value=0):
    """Collate function to pad sequences in a batch."""
    input_ids = [item["input_ids"] for item in batch]
    lengths = torch.tensor([len(seq) for seq in input_ids], dtype=torch.long)
    padded = pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
    return {"input_ids": padded, "lengths": lengths}
