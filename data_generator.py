import json
import random
import string
from enum import Enum
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

EXAMPLES_PER_CLASS = 500
random.seed(42)

VARIABLE = 0
FUNCTION = 1
CLASS = 2


# ============================================================
# IDENTIFIER SOURCES
# ============================================================

class IdentifierSource(Enum):
    LETTERS = "letters"
    TOKENIZER = "tokenizer"
    COMMON = "common"


LETTER_POOL = list(string.ascii_lowercase)

COMMON_VARIABLES = ["data", "x", "y", "value", "result", "state"]
COMMON_FUNCTIONS = ["process", "compute", "load", "parse", "run"]
COMMON_CLASSES = ["User", "Model", "Dataset", "Config", "Manager"]

TOKENIZER_POOL = None


def build_tokenizer_pool():
    global TOKENIZER_POOL
    if TOKENIZER_POOL is not None:
        return TOKENIZER_POOL

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

        vocab = [
            v for v in tok.get_vocab().keys()
            if v.isidentifier() and len(v) > 1
        ]

        TOKENIZER_POOL = vocab if len(vocab) > 100 else COMMON_FUNCTIONS

    except:
        TOKENIZER_POOL = COMMON_FUNCTIONS

    return TOKENIZER_POOL


# ============================================================
# RANDOM LITERALS (FIXED MISSING PIECE)
# ============================================================

def rand_int():
    return random.randint(-100, 100)

def rand_float():
    return round(random.uniform(-100, 100), 3)

def rand_string():
    return ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))

def rand_literal():
    return random.choice([
        rand_int(),
        rand_float(),
        rand_string(),
        True,
        False,
        None
    ])


# ============================================================
# IDENTIFIER CONTEXT (CONSISTENT STYLE PER EXAMPLE)
# ============================================================

class IdentifierContext:

    def __init__(self, source):

        self.source = source

        if source == IdentifierSource.LETTERS:
            self.var_pool = LETTER_POOL
            self.func_pool = LETTER_POOL
            self.cls_pool = list(string.ascii_uppercase)

        elif source == IdentifierSource.COMMON:
            self.var_pool = COMMON_VARIABLES
            self.func_pool = COMMON_FUNCTIONS
            self.cls_pool = COMMON_CLASSES

        else:
            pool = build_tokenizer_pool()
            self.var_pool = pool
            self.func_pool = pool
            self.cls_pool = pool

    def variable(self):
        return random.choice(self.var_pool)

    def function(self):
        return random.choice(self.func_pool)

    def class_(self):
        return random.choice(self.cls_pool)


# ============================================================
# CORE BINDING GENERATION (UNCHANGED LOGIC, CLEANED)
# ============================================================

def build_binding_example(label, ctx):

    name = (
        ctx.variable() if label == VARIABLE
        else ctx.function() if label == FUNCTION
        else ctx.class_()
    )

    if label == VARIABLE:
        definition = f"{name} = {rand_float()}"
        usage = f"q = {name} + 10"

    elif label == FUNCTION:
        definition = f"def {name}(x): return x * 2"
        usage = f"i = {name}(9)"

    else:
        definition = f"class {name}: pass"
        usage = f"j = {name}()"

    return {
        "label": label,
        "name": name,
        "definition": definition,
        "usage": usage
    }


# ============================================================
# DIVERSITY: ADDITIONAL CONTEXT (SAFE DRASTICALLY REDUCED RISK)
# ============================================================

def add_mixed_distractors(ctx):

    v = ctx.variable()
    f = ctx.function()
    c = ctx.class_()

    return "\n".join([
        f"{v}_aux = {rand_literal()}",
        f"def {f}_aux(x): return x",
        f"class {c}Aux: pass"
    ])


# ============================================================
# MASKING (CORRECT + CONSISTENT)
# ============================================================

def apply_mask(example, mask_mode):

    if mask_mode == "definition":
        definition = example["definition"].replace(example["name"], "<FIM>")
        usage = example["usage"]

    else:
        definition = example["definition"]
        usage = example["usage"].replace(example["name"], "<FIM>")

    return {
        "text": definition + "\n\n" + usage,
        "label": example["label"],
        "target": example["name"],
        "mask_mode": mask_mode
    }


# ============================================================
# MIXED (FIXED: SINGLE SOURCE OF TRUTH)
# ============================================================

def generate_mixed_example(ctx, mask_mode):

    base = build_binding_example(
        label=random.choice([VARIABLE, FUNCTION, CLASS]),
        ctx=ctx
    )

    ex = apply_mask(base, mask_mode)

    # optional noise (does NOT affect target binding)
    ex["text"] += "\n\n" + add_mixed_distractors(ctx)

    ex["mixed"] = True

    return ex


# ============================================================
# CLEAN (SINGLE CONTEXT VERSION)
# ============================================================

def generate_example(label, source, mask_mode, mixed=False):

    ctx = IdentifierContext(source)

    base = build_binding_example(label, ctx)

    ex = apply_mask(base, mask_mode)

    if mixed:
        ex["text"] += "\n\n" + add_mixed_distractors(ctx)

    return ex


# ============================================================
# WRITERS
# ============================================================
def is_valid_fim_example(ex: dict) -> bool:
    return ex["text"].count("<FIM>") == 1


def write_dataset(path, source, mask_mode, mixed=False):

    out = []

    target_size = EXAMPLES_PER_CLASS * 3  # keep your original scale

    ctx = IdentifierContext(source)

    attempts = 0
    max_attempts = target_size * 50  # safety bound

    while len(out) < target_size and attempts < max_attempts:

        attempts += 1

        label = random.choice([VARIABLE, FUNCTION, CLASS])

        ex = generate_example(
            label=label,
            source=source,
            mask_mode=mask_mode,
            mixed=mixed
        )

        if not is_valid_fim_example(ex):
            continue  # ❌ discard bad sample

        out.append(ex)

    if len(out) < target_size:
        raise RuntimeError(
            f"Only generated {len(out)}/{target_size} valid samples for {path}"
        )

    random.shuffle(out)

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for ex in out:
            f.write(json.dumps(ex) + "\n")

    print("wrote", path, len(out), "attempts:", attempts)

def write_mixed_dataset(path, source, mask_mode):

    out = []

    target_size = EXAMPLES_PER_CLASS * 3
    attempts = 0
    max_attempts = target_size * 50  # safety bound

    while len(out) < target_size and attempts < max_attempts:

        attempts += 1

        ctx = IdentifierContext(source)

        ex = generate_mixed_example(ctx, mask_mode)

        # -------------------------
        # VALIDATION: EXACTLY ONE FIM
        # -------------------------
        if ex["text"].count("<FIM>") != 1:
            continue  # discard bad sample

        out.append(ex)

    if len(out) < target_size:
        raise RuntimeError(
            f"Only generated {len(out)}/{target_size} valid mixed samples for {path}"
        )

    random.shuffle(out)

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for ex in out:
            f.write(json.dumps(ex) + "\n")

    print("wrote", path, len(out), "attempts:", attempts)

# def write_mixed_dataset(path, source, mask_mode):

#     out = []

#     for _ in range(EXAMPLES_PER_CLASS * 3):

#         ctx = IdentifierContext(source)

#         ex = generate_mixed_example(ctx, mask_mode)

#         out.append(ex)

#     random.shuffle(out)

#     Path(path).parent.mkdir(parents=True, exist_ok=True)

#     with open(path, "w") as f:
#         for ex in out:
#             f.write(json.dumps(ex) + "\n")

#     print("wrote", path, len(out))


# def write_dataset(path, source, mask_mode, mixed=False):

#     out = []

#     for label in [VARIABLE, FUNCTION, CLASS]:

#         for _ in range(EXAMPLES_PER_CLASS):

#             ex = generate_example(label, source, mask_mode, mixed)

#             out.append(ex)

#     random.shuffle(out)

#     Path(path).parent.mkdir(parents=True, exist_ok=True)

#     with open(path, "w") as f:
#         for ex in out:
#             f.write(json.dumps(ex) + "\n")

#     print("wrote", path, len(out))


# ============================================================
# MAIN
# ============================================================

def main():

    for source in [
        IdentifierSource.LETTERS,
        IdentifierSource.TOKENIZER,
        IdentifierSource.COMMON,
    ]:

        base = Path("datasets") / source.value

        write_dataset(base / "single_definition.jsonl", source, "definition")
        write_dataset(base / "single_usage.jsonl", source, "usage")

        write_mixed_dataset(base / "mixed_definition.jsonl", source, "definition")
        write_mixed_dataset(base / "mixed_usage.jsonl", source, "usage")


if __name__ == "__main__":
    main()