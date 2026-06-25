"""
This file contains functions for loading the model and dataset"""
# https://huggingface.co/codellama/CodeLlama-7b-Python-hf?library=transformers

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
# model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Python-hf")
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import transformer_lens
from pathlib import Path
from typing import List, Dict
from linearprobe_new import LinearProbe
import os
import json

model_id = "codellama/CodeLlama-7b-hf"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # torch_dtype="float16"
)

# print(tokenizer.special_tokens_map)
# print(tokenizer.additional_special_tokens)

# print(tokenizer.convert_tokens_to_ids("▁<PRE>"))  # correct
# print(tokenizer.convert_tokens_to_ids("▁<MID>"))  # correct
# print(tokenizer.convert_tokens_to_ids("▁<SUF>"))  # correct

# def read_fim_dataset(path: str) -> List[Dict[str, str]]:
#     """
#     Reads a file where:
#       - datapoints are separated by '#####'
#       - each datapoint contains one 'FIM' and one '>>>'

#     Returns a list of dicts:
#         {   "identifier_type": int
#             "prefix":  '',
#             "suffix":  '',
#             "correct": ''
#         }
#     """
#     text = Path(path).read_text(encoding="utf-8")
#     blocks = text.split("#####")

#     examples = []

#     for block in blocks:
#         block = block.strip()
#         if not block:
#             continue

#         if block.count("FIM") != 1:
#             raise ValueError("Block must contain exactly one 'FIM':\n" + block)

#         if block.count(">>>") != 1:
#             raise ValueError("Block must contain exactly one '>>>' :\n" + block)

#         # Split at FIM
#         before_fim, rest = block.split("FIM", 1)

#         # Split rest at >>>
#         middle, after_arrow = rest.split("\n>>>", 1)

#         correct, ID = after_arrow.split("\nID:")

#         prefix = before_fim
#         suffix = middle
#         correct = after_arrow

#         examples.append({
#             "identifier_type": int(ID),
#             "prefix": prefix,
#             "suffix": suffix,
#             "correct": correct})

#     return examples

def read_fim_dataset(path: str) -> List[Dict]:
    """
    Reads the new JSONL FIM dataset format:

    Each line is:
        {
            "text": "... <FIM> ...",
            "label": int,
            "target": str,
            "mask_mode": "definition" | "usage",
            "mixed": optional bool
        }

    Returns:
        List of dicts with:
        {
            "identifier_type": int,
            "prefix": str,
            "suffix": str,
            "correct": str,
            "mask_mode": str
        }
    """

    examples = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            ex = json.loads(line)

            text = ex["text"]
            label = ex["label"]
            target = ex["target"]
            mask_mode = ex.get("mask_mode", "unknown")
            variable = ex.get("var")
            function = ex.get("fun")
            class_ = ex.get("cls")

            if "<FIM>" not in text:
                raise ValueError(f"Missing <FIM> in example:\n{text}")

            if text.count("<FIM>") != 1:
                raise ValueError(f"More than one <FIM> in example:\n{text}")

            prefix, suffix = text.split("<FIM>")

            examples.append({
                "identifier_type": int(label),
                "prefix": prefix,
                "suffix": suffix,
                "correct": target,
                "mask_mode": mask_mode,
                "0": variable,
                "1": function,
                "2": class_,
            })

            # print(examples[0])

    return examples


def strip_string(s: str, to_strip: str) -> str:
    s = s.strip()
    if s.startswith(to_strip):
        return s.removeprefix(to_strip).strip()
    return s


def read_steering_dataset(path: str) -> List[Dict[str, str]]:
    """
    Reads a file where:
      - datapoints are separated by '#####'
      - each datapoint contains one 'FIM' and one '>>>'

    Returns a list of dicts:
        {
            "identifier_type": int,
            "prefix": '',
            "suffix": '',
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

        if ">>>" not in block:
            raise ValueError("Block missing '>>>' marker:\n" + block)

        # --- split core parts ---
        before_fim, rest = block.split("FIM", 1)
        after_fim, after_arrow = rest.split("\n>>>", 1)

        after_arrow_lines = [
            line.strip()
            for line in after_arrow.strip().splitlines()
            if line.strip()
        ]

        if len(after_arrow_lines) < 2:
            raise ValueError("Malformed >>> section:\n" + block)

        correct = after_arrow_lines[0]

        example = {
            "prefix": before_fim.rstrip(),
            "suffix": after_fim.lstrip(),
            "correct": correct,
        }

        # --- parse ID lines ---
        keys = ["ID:", "0:", "1:", "2:"]
        for key, line in zip(keys, after_arrow_lines[1:]):
            example[key[:-1]] = strip_string(line, key)

        examples.append(example)

    return examples


def get_prompt(prefix: str, suffix: str):
    """
    returns prompt
    """
    return f"▁<PRE>{prefix}▁<SUF>{suffix}▁<MID>"


def get_prompts_and_IDS(data):
    prompts = []
    ids = []
    for item in data:
        prompt = get_prompt(prefix=item["prefix"], suffix=item["suffix"])
        prompts.append(prompt)
        ids.append(item["identifier_type"])

    print (prompts[0], ids[0])

    return prompts, ids


def load_model(model_id="codellama/CodeLlama-7b-hf", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = transformer_lens.HookedTransformer.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device=device,
    )

    return model, tokenizer


def randomize_model_weights(model):
    """Iterates through all model parameters and randomizes them."""
    with torch.no_grad():
        count = 0
        for name, param in model.named_parameters():
            # Catch standard TransformerLens weight matrices (W_E, W_Q, W_in, etc.) 
            # and any lingering standard 'weight' matrices
            if "W_" in name or "weight" in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
                count += 1 
                
            # Catch biases (b_Q, b_in, etc.)
            elif "b_" in name or "bias" in name:
                torch.nn.init.zeros_(param)
                count += 1
                
            # Catch --- (usually named 'w' or 'scale')
            # Initialize them to 1.0 (standard for normal scale)
            elif name.endswith(".w") or "scale" in name:
                torch.nn.init.ones_(param)
            
        # print the amount of weights changed
        print(f"Randomized {count} weights")
    return model

def load_random_model(model_id="codellama/CodeLlama-7b-hf", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 1. Get the configuration for the model architecture
    cfg = transformer_lens.loading_from_pretrained.get_pretrained_model_config(
        model_id,
        torch_dtype=torch.float16,
    )
    
    # 2. Force the device in the config
    cfg.device = device

    # 3. Initialize a blank model with randomized weights based on that config
    model = transformer_lens.HookedTransformer(cfg, tokenizer=tokenizer)

    return model, tokenizer

def load_dataset(data_def, data_call, part="FULL"):
    def_fim_dict = read_fim_dataset(data_def)
    call_fim_dict = read_fim_dataset(data_call)

    def_prompts, def_ids = get_prompts_and_IDS(def_fim_dict)
    call_prompts, call_ids = get_prompts_and_IDS(call_fim_dict)

    if part == "FULL":
        prompts = def_prompts + call_prompts
        ids = def_ids + call_ids
    elif part == "DEF":
        prompts = def_prompts
        ids = def_ids
    elif part == "CALL":
        prompts = call_prompts
        ids = call_ids

    return prompts, torch.tensor(ids, dtype=torch.long)


def train_test_split(
    X: torch.Tensor,
    y: torch.Tensor,
    test_frac: float = 0.2,
    seed: int = 0,
):
    """
    Randomly split tensors into train and test sets.
    """
    assert X.size(0) == y.size(0)
    N = X.size(0)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)

    n_test = int(test_frac * N)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    return (
        X[train_idx],
        y[train_idx],
        X[test_idx],
        y[test_idx],
    )

    
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


def save_probe(probe, path, d_model, num_classes):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "state_dict": probe.state_dict(),
        "d_model": d_model,
        "num_classes": num_classes,
    }, path)


def load_probe(path, device="cuda"):
    checkpoint = torch.load(path, map_location=device)

    probe = LinearProbe(
        d_model=checkpoint["d_model"],
        num_classes=checkpoint["num_classes"],
    )

    probe.load_state_dict(checkpoint["state_dict"])
    probe.to(device)
    probe.eval()

    return probe

def evaluate_first_token_accuracy(model, tokenizer, data_path: str, save_path: str = None):
    """
    Evaluates how often the decoded top-1 next token prediction matches 
    the decoded first token of the correct target string.
    Prints execution logs for the first 10 and last 10 examples.
    Saves all correctly predicted items back into the original raw text format.
    """
    print(f"\nEvaluating first-token accuracy on: {data_path}")
    
    data = read_fim_dataset(data_path)
    total_examples = len(data)
    
    # Identify which indices to inspect (prevents overlapping prints if total < 20)
    indices_to_print = set(range(min(10, total_examples))).union(
        set(range(max(0, total_examples - 10), total_examples))
    )
    
    correct_predictions = 0
    total_predictions = total_examples
    correct_examples = []  # List to store the correct dataset entries

    for i, item in enumerate(data):
        # 1. Construct the prompt
        prompt = get_prompt(prefix=item["prefix"], suffix=item["suffix"])
        
        # 2. Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.cfg.device)

        # 3. Tokenize target
        # Splitting by \nID: bypasses the minor bug in read_fim_dataset
        target_text_raw = item["correct"].split("\nID:")[0].strip()
        target_tokens = tokenizer.encode(target_text_raw, add_special_tokens=False)
        
        if not target_tokens:
            total_predictions -= 1
            continue
            
        target_token_id = target_tokens[0]

        # 4. Forward pass
        with torch.no_grad():
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]
            predicted_token_id = torch.argmax(next_token_logits).item()

        # 5. Decode BOTH tokens to strings and strip whitespace/SentencePiece artifacts
        predicted_str = tokenizer.decode([predicted_token_id]).strip()
        target_str = tokenizer.decode([target_token_id]).strip()

        # --- DEBUG PRINT FOR FIRST 10 & LAST 10 ---
        if i in indices_to_print:
            print(f"--- Example {i+1} / {total_examples} ---")
            print(f"Input: '{prompt}'")
            print(f"Ground Truth: '{target_str}'")
            print(f"Prediction:   '{predicted_str}'")
            print("-" * 40)

        # 6. Compare the decoded strings instead of the raw IDs
        if predicted_str == target_str:
            correct_predictions += 1
            correct_examples.append(item)  # Capture the successful dataset entry

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"First-Token Accuracy: {accuracy * 100:.2f}% ({correct_predictions}/{total_predictions})\n")
    
    # 7. Save the dataset of correct examples in the original raw text format
    if save_path and correct_examples:
        # Create directories if they don't exist
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "w", encoding="utf-8") as f:
            for entry in correct_examples:
                # FIXED: Added '\n' right before '>>>' to match your parser's exact split behavior
                f.write(f"{entry['prefix']}FIM{entry['suffix']}\n>>>{entry['correct']}\n#####\n\n")
                
        print(f"Saved {len(correct_examples)} correct examples in raw text format to: {save_path}\n")
    
    return accuracy

# def evaluate_first_token_accuracy(model, tokenizer, data_path: str):
#     """
#     Evaluates how often the decoded top-1 next token prediction matches 
#     the decoded first token of the correct target string.
#     Prints execution logs for the first 10 and last 10 examples.
#     """
#     print(f"\nEvaluating first-token accuracy on: {data_path}")
    
#     data = read_fim_dataset(data_path)
#     total_examples = len(data)
    
#     # Identify which indices to inspect (prevents overlapping prints if total < 20)
#     indices_to_print = set(range(min(10, total_examples))).union(
#         set(range(max(0, total_examples - 10), total_examples))
#     )
    
#     correct_predictions = 0
#     total_predictions = total_examples

#     for i, item in enumerate(data):
#         # 1. Construct the prompt
#         prompt = get_prompt(prefix=item["prefix"], suffix=item["suffix"])
        
#         # 2. Tokenize prompt
#         input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.cfg.device)

#         # 3. Tokenize target
#         # Splitting by \nID: bypasses the minor bug in read_fim_dataset
#         target_text_raw = item["correct"].split("\nID:")[0].strip()
#         target_tokens = tokenizer.encode(target_text_raw, add_special_tokens=False)
        
#         if not target_tokens:
#             total_predictions -= 1
#             continue
            
#         target_token_id = target_tokens[0]

#         # 4. Forward pass
#         with torch.no_grad():
#             logits = model(input_ids)
#             next_token_logits = logits[0, -1, :]
#             predicted_token_id = torch.argmax(next_token_logits).item()

#         # 5. Decode BOTH tokens to strings and strip whitespace/SentencePiece artifacts
#         predicted_str = tokenizer.decode([predicted_token_id]).strip()
#         target_str = tokenizer.decode([target_token_id]).strip()

#         # --- DEBUG PRINT FOR FIRST 10 & LAST 10 ---
#         if i in indices_to_print:
#             # We show the last 100 characters of the input so you can see the immediate context
            
#             print(f"--- Example {i+1} / {total_examples} ---")
#             print(f"Input: '{prompt}'")
#             print(f"Ground Truth: '{target_str}'")
#             print(f"Prediction:   '{predicted_str}'")
#             print("-" * 40)

#         # 6. Compare the decoded strings instead of the raw IDs
#         if predicted_str == target_str:
#             correct_predictions += 1

#     accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
#     print(f"First-Token Accuracy: {accuracy * 100:.2f}% ({correct_predictions}/{total_predictions})\n")
    
#     return accuracy