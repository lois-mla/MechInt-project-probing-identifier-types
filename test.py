# https://huggingface.co/codellama/CodeLlama-7b-Python-hf?library=transformers

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
# model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Python-hf")
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
from typing import List, Dict

model_id = "codellama/CodeLlama-7b-hf"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # torch_dtype="float16"
)

print(tokenizer.special_tokens_map)
print(tokenizer.additional_special_tokens)

print(tokenizer.convert_tokens_to_ids("▁<PRE>"))  # correct
print(tokenizer.convert_tokens_to_ids("▁<MID>"))  # correct
print(tokenizer.convert_tokens_to_ids("▁<SUF>"))  # correct

def read_fim_dataset(path: str) -> List[Dict[str, str]]:
    """
    Reads a file where:
      - datapoints are separated by '#####'
      - each datapoint contains one 'FIM' and one '>>>'

    Returns a list of dicts:
        {   "identifier_type": int
            "prefix":  '',
            "suffix":  '',
            "correct": ''
        }
    """
    text = Path(path).read_text(encoding="utf-8")
    blocks = text.split("#####")

    examples = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        if block.count("FIM") != 1:
            raise ValueError("Block must contain exactly one 'FIM':\n" + block)

        if block.count(">>>") != 1:
            raise ValueError("Block must contain exactly one '>>>' :\n" + block)

        # Split at FIM
        before_fim, rest = block.split("FIM", 1)

        # Split rest at >>>
        middle, after_arrow = rest.split("\n>>>", 1)

        correct, ID = after_arrow.split("\nID:")

        prefix = f'"""{before_fim}"""'
        suffix = f'"""{middle}"""'
        correct = f'"""{after_arrow}"""'

        examples.append({
            "identifier_type": int(ID),
            "prefix": prefix,
            "suffix": suffix,
            "correct": correct})

    return examples


def get_prompt(prefix: str, suffix: str):
    """
    returns prompt
    """
    return f"▁<PRE> {prefix} ▁<SUF>{suffix} ▁<MID>"

def fill_in_middle(file):

    data = read_fim_dataset(file)

    for item in data:
        prompt = get_prompt(prefix=item["prefix"], suffix=item["suffix"])
        
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

        print("begin")
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print(tokenizer.decode(outputs[0]))
        print("end")

# fill_in_middle("training_data/template.txt")

data = read_fim_dataset("training_data/template.txt")

for item in data:
    prompt = get_prompt(prefix=item["prefix"], suffix=item["suffix"])
    print(prompt)

# prefix = """def """
# suffix = """(x, y):
#     return x + y

# sum = addition(2, 3)
# """

# fill_in_middle(prefix, suffix)

# prefix = """# function that adds two numbers
# def """
# suffix = """(x, y):
#     return x + y

# # add 2 and 3 together
# sum = addition(2, 3)
# """

# fill_in_middle(prefix, suffix)

# prefix = """# set var to 0
# """
# suffix = """ = 0
# # add 1 to var
# var += 1
# """

# fill_in_middle(prefix, suffix)

# prefix = ""
# suffix = """ = 0
# var += 1
# """

# fill_in_middle(prefix, suffix)


# prefix = """class """ 
# suffix = """:
#     def __init__(self):
#         self.data = []

#     def add(self, x):
#         self.data.append(x)

# bag = Bag()"""

# fill_in_middle(prefix, suffix)


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

