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

# print(tokenizer.convert_tokens_to_ids("<fim_prefix>"))
# print(tokenizer.convert_tokens_to_ids("<fim_suffix>"))
# print(tokenizer.convert_tokens_to_ids("<fim_middle>"))

# print(tokenizer.convert_tokens_to_ids("<PRE>"))
# print(tokenizer.convert_tokens_to_ids("<SUF>"))
# print(tokenizer.convert_tokens_to_ids("<MID>"))

# print(tokenizer.convert_tokens_to_ids("_<PRE>"))
# print(tokenizer.convert_tokens_to_ids("_<SUF>"))
# print(tokenizer.convert_tokens_to_ids("_<MID>"))

print(tokenizer.special_tokens_map)
print(tokenizer.additional_special_tokens)

print(tokenizer.convert_tokens_to_ids("▁<PRE>"))  # correct
print(tokenizer.convert_tokens_to_ids("▁<MID>"))  # correct
print(tokenizer.convert_tokens_to_ids("▁<SUF>"))  # correct

def fill_in_middle(prefix: str, suffix: str):
    # CodeLlama FIM convention: use special <fim-prefix> and <fim-suffix> tokens
    # The model supports <fim-prefix> and <fim-suffix> for infilling
    prompt = f"▁<PRE> {prefix} ▁<SUF>{suffix} ▁<MID>"

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


prefix = """def """
suffix = """(x, y):
    return x + y

sum = g(2, 3)
"""

fill_in_middle(prefix, suffix)
print("correct: g")


prefix = """# function that adds two numbers
def """
suffix = """(x, y):
    return x + y

# add 2 and 3 together
sum = g(2, 3)
"""

fill_in_middle(prefix, suffix)

prefix = """# set var to 0
"""
suffix = """ = 0
# add 1 to var
t += 1
"""

fill_in_middle(prefix, suffix)
print("correct: t")


prefix = ""
suffix = """ = 0
t += 1
"""

fill_in_middle(prefix, suffix)
print("correct: t")



prefix = """class """ 
suffix = """:
    def __init__(self):
        self.data = []

    def add(self, x):
        self.data.append(x)

bag = e()"""

fill_in_middle(prefix, suffix)
print("correct: e")

prefix = '''def p(a,b):
    return(a+b)
class c:
    x=5
x=3
y=5
#new variable z equal to 8
z='''
suffix = '''(x,y)'''

fill_in_middle(prefix, suffix)
print("correct: p")


prefix =  '''def s(x):
    return x * x
class c:
    def _init_(self, start=0):  
        self.value = start
    def inc(self):
        self.value += 1
        return self.value'''
suffix = '''  = True
sq = s(2)
ctr = c(0)
after = ctr.inc()
total = ctr.value
neg = not f
both = f and False'''

fill_in_middle(prefix, suffix)
print("correct: f")


prefix = '''class i:
    def _init_(self, name):
        self.name = name
        def greet(self):
            return f'Hi {self.name}'
def a(a, b):
    return a + b
n = 10
res = a(1, 7)
g = '''
suffix = '''('name3')
msg = g.greet()
n = g.name
m = n + 3
check = n > 5
'''

fill_in_middle(prefix, suffix)
print("correct: i")


prefix = '''def i(j,c):
    return j+c
class r:
    a=5
k=3
h ='''
suffix = ''' ()
o=h.a
at=i(o,k)
d=k+3'''

fill_in_middle(prefix, suffix)
print("correct: r")


prefix = '''
def p(j,c):
    return j+c
o=7
k=32
w='''
suffix = '''(o,k)'''

fill_in_middle(prefix, suffix)
print("correct: p")

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

