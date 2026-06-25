
import json
from typing import List, Dict
from pathlib import Path


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


read_fim_dataset("datasets/letters/mixed_definition.jsonl")