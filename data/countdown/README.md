# Countdown Dataset Usage

The countdown dataset has been integrated into `tinyzero_diffusion` following SPG's format.

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

## Dataset Location

- **Test set**: `data/countdown/countdown_cd3_test.jsonl` (256 examples)
- Format: `{"input": "num1,num2,num3", "output": "target"}`

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
