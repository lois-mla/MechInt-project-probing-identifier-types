
import json
from typing import List, Dict
from pathlib import Path
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_fim_prompt(text: str) -> str:
    prefix, suffix = text.split("<FIM>")

    return (
        "▁<PRE>"
        + prefix
        + "▁<SUF>"
        + suffix
        + "▁<MID>"
    )


def extract_identifier(text: str) -> str:
    """
    Extract first valid Python identifier from generation.
    """
    m = re.search(r"[A-Za-z_][A-Za-z0-9_]*", text)
    return m.group(0) if m else ""


@torch.no_grad()
def predict_identifier(
    model,
    tokenizer,
    example,
    max_new_tokens=20,
):
    prompt = build_fim_prompt(example["text"])
    print(prompt)
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
    )

    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=False,
    )

    # print(tokenizer.decode(outputs[0]))

    prediction = extract_identifier(generated)

    return prediction, generated


def evaluate_dataset(
    dataset_path,
    model_name="codellama/CodeLlama-7b-hf",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(tokenizer.special_tokens_map)
    # tokenizer.additional_special_tokens

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    total = 0
    correct = 0

    for line in open(dataset_path):
        ex = json.loads(line)

        pred, raw = predict_identifier(
            model,
            tokenizer,
            ex,
        )

        ok = pred == ex["target"]

        total += 1
        correct += int(ok)

        print(
            f"target={ex['target']:10s}"
            f" pred={pred:10s}"
            f" {'✓' if ok else '✗'}"
        )

    acc = correct / total

    print()
    print(f"Accuracy: {correct}/{total} = {acc:.4f}")

    return acc

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

    # examples = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            ex = json.loads(line)

            text = ex["text"]
            label = ex["label"]
            target = ex["target"]

            print(text)
            print(label)
            print(target)
            print("------------------------------------------")


# read_fim_dataset("datasets/simple/letters/mixed_definition.jsonl")

evaluate_dataset("datasets/simple/letters/mixed_usage.jsonl")