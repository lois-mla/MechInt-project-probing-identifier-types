# examples from the website https://huggingface.co/codellama/CodeLlama-7b-Python-hf?library=transformers
# idk which way is better

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Python-hf")


# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("text-generation", model="codellama/CodeLlama-7b-Python-hf")