from datasets import load_dataset
import json
import os

save_dir = "data/countdown"
os.makedirs(save_dir, exist_ok=True)

print("Downloading Jiayi-Pan/Countdown-Tasks-3to4...")
dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')

# TinyZero splits: Train=327680, Test=1024
TRAIN_SIZE = 327680
TEST_SIZE = 1024

print(f"Splitting dataset: Train={TRAIN_SIZE}, Val={TEST_SIZE}")

# Train split
train_dataset = dataset.select(range(TRAIN_SIZE))
train_output_path = os.path.join(save_dir, "countdown_hf_train.jsonl")
with open(train_output_path, 'w') as f:
    for example in train_dataset:
        numbers = ','.join(map(str, example['nums']))
        target = str(example['target'])
        f.write(json.dumps({"input": numbers, "output": target}) + '\n')
print(f"Saved {len(train_dataset)} training examples to {train_output_path}")

# Val split (using TinyZero's test split as val)
val_dataset = dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
val_output_path = os.path.join(save_dir, "countdown_hf_val.jsonl")
with open(val_output_path, 'w') as f:
    for example in val_dataset:
        numbers = ','.join(map(str, example['nums']))
        target = str(example['target'])
        f.write(json.dumps({"input": numbers, "output": target}) + '\n')
print(f"Saved {len(val_dataset)} validation examples to {val_output_path}")
