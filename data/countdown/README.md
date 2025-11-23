# Countdown Dataset Usage
# Countdown Task Dataset

This directory contains datasets for the Countdown task in different formats.

## Dataset Files

### SFT Training Data (Synthetic, with reasoning)
- `countdown_sft_1k_train.jsonl` - 1000 problems with reasoning traces for SFT
- `countdown_sft_1k_val.jsonl` - 100 problems with reasoning traces for validation
- Generated locally using `generate_data.py`
- **Purpose**: Supervised Fine-Tuning (SFT) to teach format and basic reasoning

### Test Data (Questions only)
- `countdown_cd3_test.jsonl` - 256 test problems (numbers + target only)
- From SPG repository / HuggingFace
- **Purpose**: Evaluation

## Naming Convention

```
countdown_<purpose>_<size>_<split>.jsonl

Examples:
- countdown_sft_1k_train.jsonl     # SFT training, 1K samples
- countdown_rl_327k_train.jsonl    # RL training, 327K samples (future)
- countdown_cd3_test.jsonl         # Test set, 3-number problems
```

### Dataset Versions

| Name | Size | Has Reasoning? | Purpose | Source |
|------|------|----------------|---------|--------|
| `sft_1k` | 1,000 train / 100 val | ✅ Yes | SFT warm-up | Local generator |
| `rl_327k` | 327,680 train | ❌ No | RL training | HuggingFace (future) |
| `cd3_test` | 256 test | ❌ No | Evaluation | SPG/HuggingFace |

## Data Format

### SFT Data (with reasoning)
```json
{
  "input": "10,5,2",
  "output": "15",
  "solution": "10 + 5",
  "reasoning": "10 + 5 = 15."
}
```

### Test Data (questions only)
```json
{
  "input": "30,100,93",
  "output": "23"
}
```

## Generating New Data

```bash
# Generate 1K SFT dataset (default)
python generate_data.py

# Generate custom size
python generate_data.py --train_size 5000 --val_size 500
```

## Future: Downloading HuggingFace Data for RL

```python
from datasets import load_dataset

# Full dataset for RL training (no reasoning needed)
dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')
# Save as: countdown_rl_327k_train.parquet
```

## Quick Start

```python
from tzd.data import CountdownDataset

# Load the dataset
dataset = CountdownDataset(
    data_path="data/countdown/countdown_cd3_test.jsonl",
    add_reasoning_tag=True  # Prefills <reasoning> to encourage structured output
)

# Get an example
example = dataset[0]
print(f"Numbers: {example['numbers']}")  # [30, 100, 93]
print(f"Target: {example['target']}")    # 23
print(f"Prompt:\n{example['prompt']}")
```

## Format

The dataset follows SPG's format with:
- **System prompt**: Instructions on how to solve the task
- **Question**: `Numbers: [a, b, c]\nTarget: X`
- **Expected output**:
  ```xml
  <reasoning>
  Step-by-step reasoning...
  </reasoning>
  <answer>
  \boxed{expression}
  </answer>
  ```

## Using with DataLoader

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=lambda batch: CountdownDataset.collate_fn(batch, tokenizer=your_tokenizer),
    shuffle=True
)

for batch in dataloader:
    prompts = batch["prompts"]
    numbers = batch["numbers"]
    targets = batch["targets"]
    # If tokenizer was provided:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
```

## Downloading Additional Data

To download the full dataset from HuggingFace:

```python
from tzd.data import download_countdown_dataset

# Downloads ~100k examples
dataset_path = download_countdown_dataset(save_dir="data/countdown")
```
