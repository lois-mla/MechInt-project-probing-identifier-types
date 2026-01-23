# https://huggingface.co/codellama/CodeLlama-7b-Python-hf?library=transformers

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
# model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Python-hf")

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

# # Fill-in-the-middle example
# prefix = "# function that adds two numbers \ndef "   # Start of function
# suffix = "(x, y):\n    return x + y\n\n# add 2 and 3 together\nsum = addition(2, 3)"

prefix = "# set var to 0\n"   # Start of function
suffix = " = 0\n# add 1 to var\n var +=1 "

# CodeLlama FIM convention: use special <fim-prefix> and <fim-suffix> tokens
# The model supports <fim-prefix> and <fim-suffix> for infilling
prompt = f"<PRE> {prefix} <SUF>{suffix} <MID>"

# prompt = (
#     "<fim-prefix>def "
#     "<fim-middle>"
#     "<fim-suffix>(x, y):\n"
#     "    return x + y\n\n"
#     "sum = addition(2, 3)"
# )

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate the missing middle
outputs = model.generate(
    **inputs,
    max_new_tokens=10,
    do_sample=False,
    # early_stopping=False,
    # eos_token_id=None,   # allow generation past EOS prediction
    # temperature=0.7,
)

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(tokenizer.decode(outputs[0]))


# with torch.no_grad():
#     logits = model(**inputs).logits[:, -1]
#     probs = logits.softmax(dim=-1)

# topk = torch.topk(probs, 10)
# tokens = tokenizer.convert_ids_to_tokens(topk.indices)
# scores = topk.values

# for t, p in zip(tokens, scores):
#     print(f"{t:15s} {p.item():.3f}")

# # function that adds two numbers
# def ....(x, y):
#     return x + y
# sum = add(2, 3)


# x = 1
# y = 'str'
# z = 2 + ...

