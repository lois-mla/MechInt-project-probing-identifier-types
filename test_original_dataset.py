from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "codellama/CodeLlama-7b-hf"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # torch_dtype="float16"
)

def get_prompt(prefix: str, suffix: str):
    """
    returns prompt
    """
    return f"▁<PRE>{prefix}▁<SUF>{suffix}▁<MID>"

