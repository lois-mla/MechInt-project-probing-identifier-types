# importing stuff
import transformer_lens
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "codellama/CodeLlama-7b-hf"
# model_id = "llama-7b-hf"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # torch_dtype="float16"
)

# load a model
model = transformer_lens.HookedTransformer.from_pretrained(model_id)

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