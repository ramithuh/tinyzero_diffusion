import torch
from transformers import AutoTokenizer
from tzd.models.diffusion_pretrained import from_pretrained
import time

def test_llada_generation():
    # 1. Configuration (Matching training config)
    MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
    MAX_MEMORY = {0: "40GiB"}
    QUANTIZE = "bnb.nf4"
    BLOCK_SIZE = 256
    
    print(f"Loading {MODEL_NAME}...")
    
    # 2. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # 3. Load Model (using our integration wrapper)
    model = from_pretrained(
        pretrained_model_name=MODEL_NAME,
        tokenizer=tokenizer,
        model_type="huggingface",
        quantize=QUANTIZE,
        max_memory=MAX_MEMORY,
        block_size=BLOCK_SIZE,
        generation_block_size=BLOCK_SIZE
    )
    model.eval()
    model.cuda()
    
    # 4. Prepare Prompt (using SPG format with apply_chat_template)
    SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

    user_content = SYSTEM_PROMPT + "\nUsing only the numbers [79, 17, 60], create an arithmetic expression that evaluates to exactly 36. You must use all numbers from the list, and each number must be used exactly once. You may use the operations +, -, *, and / as needed."

    # Use apply_chat_template - this is critical!
    messages = [{"role": "user", "content": user_content}]
    prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    print("Prompt:")
    print(repr(prompt_str))
    print(prompt_str)
    print("-" * 50)

    # 5. Tokenize
    input_ids = tokenizer(prompt_str, return_tensors="pt")['input_ids'].to(model.device)
    prompt_len = input_ids.shape[1]
    print(f"Prompt length: {prompt_len} tokens")
    
    # 6. Generate (using proper parameters)
    print(f"\nGenerating with semi-autoregressive blocks...")
    start_time = time.time()

    with torch.no_grad():
        gen_length = 128  # Generate 128 new tokens
        block_length = 32  # SPG uses 32 - critical for quality!
        steps = 128  # More steps = better quality

        # Ensure steps is divisible by num_blocks
        num_blocks = gen_length // block_length  # 128 / 32 = 4
        if steps % num_blocks != 0:
            steps = num_blocks * (steps // num_blocks)
            print(f"Adjusted steps to {steps} for {num_blocks} blocks")

        print(f"Configuration: gen_length={gen_length}, block_length={block_length}, steps={steps}, temperature=0.0")

        samples = model.sample(
            batch_size=1,
            prompts=input_ids,
            seq_len=gen_length,        # Generate 128 NEW tokens
            num_steps=steps,           # 128 total steps
            temperature=0.0,           # Greedy (deterministic)
            repo="LLaDA",              # Explicit
            block_length=block_length  # Smaller blocks = better quality
        )
        
    print(f"Generation took {time.time() - start_time:.2f}s")

    # 7. Decode
    print("-" * 50)
    print("Full Output (with special tokens):")
    decoded_full = tokenizer.decode(samples[0], skip_special_tokens=False)
    print(decoded_full)
    print("-" * 50)

    # 8. Decode only the completion
    print("Completion Only:")
    completion = tokenizer.decode(samples[0, prompt_len:], skip_special_tokens=True)
    print(completion)
    print("-" * 50)

if __name__ == "__main__":
    test_llada_generation()
