# importing stuff
import transformer_lens
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_id = "meta-llama/Llama-2-7b-hf"

# Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(model_id)
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
print("Cache shape:", prompt_cache.shape)
