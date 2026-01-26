# importing stuff
import transformer_lens
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_id = "meta-llama/Llama-2-7b-hf"

# # Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bigcode/santacoder")
# hf_model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="auto",
#     # torch_dtype="float16"
# )

# model = transformer_lens.HookedTransformer.from_pretrained(
#     model_id,

#     hf_model=hf_model,
#     tokenizer=tokenizer,
#     device="cuda"
# )
model = transformer_lens.HookedTransformer.from_pretrained("bigcode/santacoder")



def fill_in_middle(prefix: str, suffix: str):
    
    return f"▁<PRE> {prefix} ▁<SUF>{suffix} ▁<MID>"

# print the loss of running a prompt
prefix = """def """
suffix = """(x, y):
    return x + y

sum = addition(2, 3)
"""
prompt = fill_in_middle(prefix, suffix)

loss = model(prompt, return_type="loss")
print("Model loss:", loss)

# caching activations
prompt_tokens = model.to_tokens(prompt)
prompt_logits, prompt_cache = model.run_with_cache(prompt_tokens, remove_batch_dim=True)
# print the shape of the logits and cache
print("Logits shape:", prompt_logits.shape)
activations = prompt_cache["mlp_out", 1]
print("Activations shape:", activations.shape)



@torch.inference_mode()
def get_residual_activations(
    model: transformer_lens.HookedTransformer,
    data: list[str],
    layer: int,
    resid_type: str = "mlp_out",
    batch_size: int = 8,
    max_length: int = 256,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Returns activations suitable for linear probing.

    Output shape: [N, L, D]
        N = number of sequences
        L = number of token positions per sequence
        D = d_model (hidden size)

    Each token position gets its own vector.
    """

    all_acts = []

    # Process data in batches to save GPU memory
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]

        # Convert text → token IDs
        # Padding ensures all sequences in batch have same L
        # Tokenize with HF tokenizer (this supports padding/truncation)
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        tokens = enc["input_ids"].to(device)

        # Run the model and cache *all* intermediate activations
        _, cache = model.run_with_cache(tokens)

        # Extract a specific activation:
        # e.g. ("mlp_out", layer) → [batch, pos, d_model]
        acts = cache[(resid_type, layer)]

        # Move to CPU so GPU memory can be freed
        all_acts.append(acts.cpu())

    # Stack all batches → [N, L, D]
    return torch.cat(all_acts, dim=0)

res_activations = get_residual_activations(
    model,
    data=[prompt],
    layer=1,
    resid_type="mlp_out"
)

print("Residual activations shape:", res_activations.shape)
