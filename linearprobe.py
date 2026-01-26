# importing stuff
import transformer_lens

# load a model
model = transformer_lens.HookedTransformer.from_pretrained("Llama-2-7b-hf")

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