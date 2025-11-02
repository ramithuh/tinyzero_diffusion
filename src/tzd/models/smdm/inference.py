
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from safetensors.torch import load_file

from tzd.models.smdm.diffmodel import TransEncoder
from tzd.models.smdm.config import Config

# source: SMDM
def add_gumbel_noise(logits, temperature):
    '''
    As suggested by https://arxiv.org/pdf/2409.02908, we use float64 for the gumbel max method.
    '''
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

# source: SMDM
@torch.no_grad()
def diff_sample(model, tokenizer, prompt=None, batch_size=1, alg='origin', steps=512, temperature=1., cfg_scale=2.,
                context_length=2048, eps=1e-5, dim=32000, device='cuda'):
        
    batch_size = batch_size if prompt is None else prompt.shape[0]
    x = torch.full((batch_size, context_length), dim, dtype=torch.long).to(device)

    if(prompt is not None):
        x[:, :prompt.shape[1]] = prompt.clone()

    timesteps = torch.linspace(1, eps, steps + 1, device='cuda')
    for i in range(steps):
        mask_index = (x == dim)
        # if not mask_index.any():
        #     break
        # with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        if cfg_scale > 0.:
            un_x = x.clone()

            if(prompt is not None):
                un_x[:, :prompt.shape[1]] = dim
                
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_)
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits, un_logits = logits[mask_index], un_logits[mask_index]
        else:
            logits = model(x)[mask_index]

        if cfg_scale > 0.:
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

        t = timesteps[i]
        s = timesteps[i + 1]

        if alg == 'origin':
            p_transfer = 1 - s / t if i < steps - 1 else 1
            x0 = torch.zeros_like(x[mask_index], device=device, dtype=torch.long) + dim
            transfer_index_t_s = torch.rand(*x0.shape, device='cuda') < p_transfer
            logits_with_noise = add_gumbel_noise(logits[transfer_index_t_s], temperature=temperature)
            x0[transfer_index_t_s] = torch.argmax(logits_with_noise, dim=-1)
            x[mask_index] = x0.clone()
        elif alg == 'greddy':
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            logits = logits.to(torch.float64)
            p = F.softmax(logits, dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            num_mask_token = mask_index.sum()
            number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token
            if number_transfer_tokens > 0:
                _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                x0_ = torch.zeros_like(x0, device=device, dtype=torch.long) + dim
                x0_[transfer_index] = x0[transfer_index].clone()
                x[mask_index] = x0_
        else:
            raise NotImplementedError(alg)

    return x

if __name__ == "__main__":
    model_name = "Diff_LLaMA_1028M" #"Diff_LLaMA_113M"
    ckpt_path = "mdm-1028M-1600e18-sharegpt.safetensors" #"./mdm-1028M-60e18.safetensors" #"./mdm-113M-30e18.safetensors"
    prompt = "USER: How are you?\nASSISTANT:"
    max_new_tokens = 128
    temperature = 0.2
    steps = 128
    cfg_scale = 0
    device = "cuda"

    # Load the model
    config = Config.from_name(model_name)
    model = TransEncoder(config).to(device)
    model.load_state_dict(load_file(ckpt_path))

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to('cuda')

    context_length = max_new_tokens
    print("Needed context Length:", context_length)

    # Generate text

    output_ids = diff_sample(model, tokenizer, input_ids, steps=steps,temperature=temperature,cfg_scale=cfg_scale, context_length=context_length, device='cuda')

    print("## START ##")
    print("#########")
    print("Generated Text (including prompt):")
    # We print the whole sequence first to see both prompt and generation
    full_output = tokenizer.decode(
        output_ids[0],
        spaces_between_special_tokens=False,
    )
    print(full_output)
    print("#########")
    print("## END ###")
