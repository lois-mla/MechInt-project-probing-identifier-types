# https://huggingface.co/codellama/CodeLlama-7b-Python-hf?library=transformers

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
# model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Python-hf")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "codellama/CodeLlama-7b-Python-hf"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # torch_dtype="float16"
)

# # Fill-in-the-middle example
# prefix = "def "   # Start of function
# suffix = "(x, y):\n    return x + y\n\nsum = addition(2, 3)"

# CodeLlama FIM convention: use special <fim-prefix> and <fim-suffix> tokens
# The model supports <fim-prefix> and <fim-suffix> for infilling
# input_text = f"<fim-prefix>{prefix}<fim-suffix>{suffix}<fim-middle>"

prompt = (
    "<fim-prefix>def "
    "<fim-middle>"
    "<fim-suffix>(x, y):\n"
    "    return x + y\n\n"
    "sum = addition(2, 3)"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate the missing middle
outputs = model.generate(
    **inputs,
    max_new_tokens=5,
    do_sample=False,
    # temperature=0.7,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))



# # function that adds two numbers
# def ....(x, y):
#     return x + y
# sum = add(2, 3)


# x = 1
# y = 'str'
# z = 2 + ...

