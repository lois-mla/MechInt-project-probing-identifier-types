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

# def rand_int():
#     return random.randint(-100, 100)

# def rand_float():
#     return round(random.uniform(-100, 100), 3)

# def rand_string():
#     return ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))

# def rand_literal():
#     return random.choice([
#         rand_int(),
#         rand_float(),
#         rand_string(),
#         True,
#         False,
#         None
#     ])


# ============================================================
# IDENTIFIER CONTEXT (CONSISTENT STYLE PER EXAMPLE)
# ============================================================
class IdentifierContext:
# kind of redundent code here & there. FIx this later
    def __init__(self, source):

        self.source = source

        if source == IdentifierSource.LETTERS:
            self.var_pool = LETTER_POOL
            self.func_pool = LETTER_POOL
            self.cls_pool = LETTER_POOL

        elif source == IdentifierSource.COMMON:
            self.var_pool = COMMON_VARIABLES
            self.func_pool = COMMON_FUNCTIONS
            self.cls_pool = COMMON_CLASSES

        else:
            pool = build_tokenizer_pool()
            self.var_pool = pool
            self.func_pool = pool
            self.cls_pool = pool

    # -------------------------
    # identifiers
    # -------------------------

    def variable(self):
        return random.choice(self.var_pool)

    def function(self):
        return random.choice(self.func_pool)

    def class_(self):
        return random.choice(self.cls_pool)

    
    def identifiers(self):
        if self.source == IdentifierSource.COMMON:
            var = random.choice(COMMON_VARIABLES)
            func = random.choice(COMMON_FUNCTIONS)
            cls_ = random.choice(COMMON_CLASSES)

            return {
                "var": var,
                "func": func,
                "cls": cls_,
            }

        elif self.source == IdentifierSource.LETTERS:
            pool = LETTER_POOL
        else:
            pool = build_tokenizer_pool()

        # sample 3 so that we don't get the same tokens!
        names = random.sample(pool, 3)

        return {
            "var": names[0],
            "func": names[1],
            "cls": names[2],
        }


    # -------------------------
    # literals (NOW CONSISTENTLY TIED TO CONTEXT)
    # -------------------------
    def int(self):
        return random.randint(-100, 100)

    def float(self):
        return round(random.uniform(-100, 100), 3)

    def string(self):
        # IMPORTANT: now depends on identifier style
        if self.source == IdentifierSource.LETTERS:
            return random.choice(LETTER_POOL)
        elif self.source == IdentifierSource.COMMON:
            return random.choice(COMMON_VARIABLES)
        else:
            pool = build_tokenizer_pool()
            return random.choice(pool)

# ============================================================
# CORE BINDING GENERATION (UNCHANGED LOGIC, CLEANED)
# ============================================================
# ============================================================
# RANDOM LITERALS (as you already use)
# ============================================================

# def rand_int():
#     return random.randint(-100, 100)

# def rand_float():
#     return round(random.uniform(-100, 100), 3)

# def rand_string():
#     return ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))


# def template_vars():
#     return {
#         "int1": rand_int(),
#         "int2": rand_int(),
#         "int3": rand_int(),
#         "float1": rand_float(),
#         "str1": rand_string(),
#         "str2": rand_string(),
#     }


# ============================================================
# VARIABLE DEFINITIONS
# ============================================================
VAR_DEFINITION_TEMPLATES = [

    # scalars
    {"tpl": "<FIM> = {int1}", "type": "int"},
    {"tpl": "<FIM> = {float1}", "type": "float"},
    {"tpl": "<FIM> = {str1}", "type": "str"},
    {"tpl": "<FIM> = True", "type": "bool"},
    {"tpl": "<FIM> = False", "type": "bool"},
    {"tpl": "<FIM> = None", "type": "none"},

    # arithmetic variants
    {"tpl": "<FIM> = {int1} + {int2}", "type": "int"},
    {"tpl": "<FIM> = {int1} * {int2}", "type": "int"},
    {"tpl": "<FIM> = ({int1} - {int2}) / {int3}", "type": "float"},
    {"tpl": "<FIM> = -{int1}", "type": "int"},
    {"tpl": "<FIM> = {int1} ** 2", "type": "int"},

    # conditional
    {"tpl": "<FIM> = {int1} if {int2} > 0 else {int3}", "type": "int"},

    # containers
    {"tpl": "<FIM> = [{int1}, {int2}]", "type": "list_int"},
    {"tpl": "<FIM> = []", "type": "list_int"},

    {"tpl": "<FIM> = ({int1}, {int2})", "type": "tuple_int"},
    {"tpl": "<FIM> = ()", "type": "tuple_int"},

    {"tpl": "<FIM> = {{'{str1}': {int1}}}", "type": "dict"},
    {"tpl": "<FIM> = {{}}", "type": "dict"},

    # mixed literals
    {"tpl": "<FIM> = [{float1}, {int1}]", "type": "list_int"},
    {"tpl": "<FIM> = {{'{str1}': {float1}}}", "type": "dict"},
]


# ============================================================
# FUNCTION DEFINITIONS
# ============================================================
FUNC_DEFINITION_TEMPLATES = [

    # unary
    {"tpl": "def <FIM>(x): return x * {int1}", "args": 1},
    {"tpl": "def <FIM>(x): return x / {int1}", "args": 1},
    {"tpl": "def <FIM>(x): return x - {int1}", "args": 1},
    {"tpl": "def <FIM>(x): return x ** {int1}", "args": 1},

    # binary
    {"tpl": "def <FIM>(x, y): return x + y", "args": 2},
    {"tpl": "def <FIM>(x, y): return x * y", "args": 2},

    # zero-arg
    {"tpl": "def <FIM>(): return {int1}", "args": 0},
    {"tpl": "def <FIM>(): return True", "args": 0},

    # # ternary
    # {"tpl": "def <FIM>(x, y, z): return x + y - z", "args": 3},

    # branching
    {"tpl": "def <FIM>(x): return x if x > {int1} else {int2}", "args": 1},

    # structured return
    {"tpl": "def <FIM>(x): return [x, {int1}]", "args": 1},
]

# ============================================================
# CLASS DEFINITIONS
# ============================================================
CLASS_DEFINITION_TEMPLATES = [

    {"tpl": "class <FIM>: pass", "args": 0},

    {"tpl": "class <FIM>:\n    VERSION = {int1}", "args": 0},

    {"tpl": "class <FIM>:\n    value = {float1}", "args": 0},

    {"tpl": "class <FIM>:\n    config = {{'a': {int1}}}", "args": 0},

    {"tpl": "class <FIM>:\n    def run(self): return {int1}", "args": 0},

    {"tpl": "class <FIM>(object): pass", "args": 0},
    {"tpl": "class <FIM>(Exception): pass", "args": 0},

    # constructors
    {"tpl": "class <FIM>:\n    def __init__(self, x): self.x = x", "args": 1},
    {"tpl": "class <FIM>:\n    def __init__(self, x, y): self.x = x; self.y = y", "args": 2},

    # slightly richer structure
    {"tpl": "class <FIM>:\n    def __init__(self): self.data = []", "args": 0},
]

# ============================================================
# VARIABLE USAGE TEMPLATES
# ============================================================
VAR_TEMPLATES = [

    # arithmetic
    {"tpl": "{var1} = <FIM> + {int1}", "types": ["int", "float"]},
    {"tpl": "{var1} = <FIM> * {int1}", "types": ["int", "float"]},
    {"tpl": "{var1} = (<FIM> + {int1}) / {int2}", "types": ["int", "float"]},

    # comparisons
    {"tpl": "if <FIM> > {int1}: pass", "types": ["int", "float"]},
    {"tpl": "if <FIM> == {int1}: pass", "types": ["int", "float", "str", "bool"]},

    # truthiness
    {"tpl": "if <FIM>: {var1} = {int1}",
     "types": ["int", "float", "str", "list_int", "tuple_int", "dict", "bool", "none"]},

    {"tpl": "while <FIM> is not None: break",
     "types": ["int", "float", "str", "list_int", "tuple_int", "dict", "bool"]},

    # containers
    {"tpl": "{var1} = [<FIM>, {int1}, {int2}]", "types": ["int", "float"]},
    {"tpl": "{var1} = [[<FIM>]]", "types": ["int", "float"]},

    {"tpl": "{var1} = {{'{str1}': <FIM>}}",
     "types": ["int", "float", "str", "bool"]},

    # unpacking
    {"tpl": "{var1}, {var2} = <FIM>", "types": ["tuple_int"]},

    # mutation
    {"tpl": "<FIM>.append({int1})", "types": ["list_int"]},
    {"tpl": "<FIM>.extend([{int1}, {int2}])", "types": ["list_int"]},

    {"tpl": "<FIM>[0]", "types": ["list_int", "tuple_int", "str"]},
    {"tpl": "<FIM>[-1]", "types": ["list_int", "tuple_int", "str"]},
    {"tpl": "<FIM>[:2]", "types": ["list_int", "tuple_int", "str"]},

    {"tpl": "<FIM>['{str1}'] = {int1}", "types": ["dict"]},
    {"tpl": "<FIM>.update({{'{str1}': {int1}}})", "types": ["dict"]},

    # assertions
    {"tpl": "assert <FIM> is not None",
     "types": ["int", "float", "str", "list_int", "tuple_int", "dict", "bool"]},
]
# ============================================================
# FUNCTION USAGE TEMPLATES
# ============================================================
FUNC_TEMPLATES = [

    # calls
    {"tpl": "{var1} = <FIM>({int1})", "args": 1},
    {"tpl": "{var1} = <FIM>({var2}, {var3})", "args": 2},
    {"tpl": "{var1} = <FIM>()", "args": 0},

    # returns
    {"tpl": "return <FIM>()", "args": 0},
    {"tpl": "return <FIM>({var1})", "args": 1},

    # higher-order
    {"tpl": "map(<FIM>, {var1})", "args": 1},
    {"tpl": "filter(<FIM>, {var1})", "args": 1},
    {"tpl": "sorted({var1}, key=<FIM>)", "args": 1},

    # callbacks
    {"tpl": "{var1}.append(<FIM>)", "args": 1},
    {"tpl": "{var1}.submit(<FIM>)", "args": 1},
    {"tpl": "{var1}.add(<FIM>)", "args": 1},

    # decorators
    {"tpl": "@<FIM>\ndef {func1}(): pass", "args": 1},

    # lambdas
    {"tpl": "lambda {var1}: <FIM>({var1})", "args": 1},

    # async / threading
    {"tpl": "Thread(target=<FIM>)", "args": 1},
    {"tpl": "asyncio.create_task(<FIM>())", "args": 0},
]
# ============================================================
# CLASS USAGE TEMPLATES
# ============================================================
CLASS_TEMPLATES = [

    {"tpl": "{var1} = <FIM>()", "ctor_args": 0},
    {"tpl": "{var1} = <FIM>({int1})", "ctor_args": 1},
    {"tpl": "{var1} = <FIM>({int1}, {int2})", "ctor_args": 2},

    {"tpl": "class {cls1}(<FIM>): pass", "ctor_args": None},
    {"tpl": "class {cls1}({cls2}, <FIM>): pass", "ctor_args": None},

    {"tpl": "issubclass({cls1}, <FIM>)", "ctor_args": None},
    {"tpl": "isinstance({var1}, <FIM>)", "ctor_args": None},

    {"tpl": "raise <FIM>()", "ctor_args": 0},

    {"tpl": "<FIM>.VERSION", "ctor_args": None},
    {"tpl": "<FIM>.config", "ctor_args": None},

    {"tpl": "<FIM>.__name__", "ctor_args": None},
]


# def build_binding_example(label, ctx):

#     name = (
#         ctx.variable() if label == VARIABLE
#         else ctx.function() if label == FUNCTION
#         else ctx.class_()
#     )

#     if label == VARIABLE:
#         definition = f"{name} = {rand_float()}"
#         usage = f"q = {name} + 10"

#     elif label == FUNCTION:
#         definition = f"def {name}(x): return x * 2"
#         usage = f"i = {name}(9)"

#     else:
#         definition = f"class {name}: pass"
#         usage = f"j = {name}()"

#     return {
#         "label": label,
#         "name": name,
#         "definition": definition,
#         "usage": usage
#     }


# ============================================================
# DIVERSITY: ADDITIONAL CONTEXT (SAFE DRASTICALLY REDUCED RISK)
# ============================================================

# def add_mixed_distractors(ctx):

#     v = ctx.variable()
#     f = ctx.function()
#     c = ctx.class_()

#     return "\n".join([
#         f"{v}_aux = {rand_literal()}",
#         f"def {f}_aux(x): return x",
#         f"class {c}Aux: pass"
#     ])


def compatible_var(def_t, use_t):
    return def_t["type"] in use_t["types"]


def compatible_func(def_t, use_t):
    return def_t["args"] == use_t["args"]


def compatible_class(def_t, use_t):
    if use_t["ctor_args"] is None:
        return True

    return def_t["args"] == use_t["ctor_args"]

def sample_pair(def_template, usage_template, identifier_type):
    def_t = random.choice(def_template)

    if identifier_type == 0:
        compatible = compatible_var 
    elif identifier_type == 1:
        compatible = compatible_func
    else:
        compatible = compatible_class

    candidates = [
        t for t in usage_template
        if compatible(def_t, t)
    ]

    use_t = random.choice(candidates)

    return def_t, use_t


# def sample_identifier(ctx):
#     return {
#         "var": ctx.variable(),
#         "func": ctx.function(),
#         "cls": ctx.class_(),
#     }

def build_env(ctx):
    env = {
    "int1": ctx.int(),
    "int2": ctx.int(),
    "int3": ctx.int(),

    "float1": ctx.float(),

    "str1": ctx.string(),
    "str2": ctx.string(),

    "var1": ctx.variable(),
    "var2": ctx.variable(),
    "var3": ctx.variable(),

    "func1": ctx.function(),

    "cls1": ctx.class_(),
    "cls2": ctx.class_(),
    }

    return env

def instantiate_pair(def_tpl, use_tpl, name, ctx):

    if isinstance(def_tpl, dict):
        def_tpl = def_tpl["tpl"]

    if isinstance(use_tpl, dict):
        use_tpl = use_tpl["tpl"]

    env = build_env(ctx)

    definition = def_tpl.format(**env).replace("<FIM>", name)
    usage = use_tpl.format(**env).replace("<FIM>", name)

    return definition, usage


def instantiate_single(def_tpl, use_tpl, name, ctx, mask_mode="usage"):

    if isinstance(def_tpl, dict):
        def_tpl = def_tpl["tpl"]

    if isinstance(use_tpl, dict):
        use_tpl = use_tpl["tpl"]

    env = build_env(ctx)

    if mask_mode == "usage":
        definition = def_tpl.format(**env).replace("<FIM>", name)
        usage = use_tpl.format(**env)

    else:
        definition = def_tpl.format(**env)
        usage = use_tpl.format(**env).replace("<FIM>", name)

    return definition, usage


def valid(order):
    pos = {x: i for i, x in enumerate(order)}

    return (
        pos["Vd"] < pos["Vu"] and
        pos["Fd"] < pos["Fu"] and
        pos["Cd"] < pos["Cu"]
    )


def shuffle_with_constraints(var_def_s, var_use_s,
                             func_def_s, func_use_s,
                             cls_def_s, cls_use_s):

    items = {
        "Vd": var_def_s,
        "Vu": var_use_s,
        "Fd": func_def_s,
        "Fu": func_use_s,
        "Cd": cls_def_s,
        "Cu": cls_use_s,
    }

    keys = list(items.keys())

    for _ in range(100):  # retry limit
        order = keys[:]
        random.shuffle(order)

        if valid(order):
            return "\n".join(items[k] for k in order)

    raise RuntimeError("Failed to generate valid ordering")


def generate_mixed_example(ctx, identifier_type, mask_mode="usage"):

    ids = ctx.identifiers()

    # -------------------------
    # pick templates
    # -------------------------
    
    var_def, var_use = sample_pair(VAR_DEFINITION_TEMPLATES, VAR_TEMPLATES, 0)
    func_def, func_use = sample_pair(FUNC_DEFINITION_TEMPLATES, FUNC_TEMPLATES, 1)
    cls_def, cls_use = sample_pair(CLASS_DEFINITION_TEMPLATES, CLASS_TEMPLATES, 2)

    # -------------------------
    # instantiate 3 pairs
    # -------------------------
    if identifier_type == 0:
        var_def_s, var_use_s = instantiate_single(var_def["tpl"], var_use, ids["var"], ctx, mask_mode=mask_mode)
        func_def_s, func_use_s = instantiate_pair(func_def["tpl"], func_use, ids["func"], ctx)
        cls_def_s, cls_use_s = instantiate_pair(cls_def["tpl"], cls_use, ids["cls"], ctx)
        target = ids["var"]
    elif identifier_type == 1:
        var_def_s, var_use_s = instantiate_pair(var_def["tpl"], var_use, ids["var"], ctx)
        func_def_s, func_use_s = instantiate_single(func_def["tpl"], func_use, ids["func"], ctx, mask_mode=mask_mode)
        cls_def_s, cls_use_s = instantiate_pair(cls_def["tpl"], cls_use, ids["cls"], ctx)
        target = ids["func"]
    else: 
        var_def_s, var_use_s = instantiate_pair(var_def["tpl"], var_use, ids["var"], ctx)
        func_def_s, func_use_s = instantiate_pair(func_def["tpl"], func_use, ids["func"], ctx)
        cls_def_s, cls_use_s = instantiate_single(cls_def["tpl"], cls_use, ids["cls"], ctx, mask_mode=mask_mode)
        target = ids["cls"]

    text = shuffle_with_constraints(var_def_s, var_use_s,
                             func_def_s, func_use_s,
                             cls_def_s, cls_use_s)

    return {
        "text": text,
        "label": identifier_type,
        "target": target,
        "mask_mode": mask_mode,
    }


def write_mixed_dataset(path, source, mask_mode):

    out = []

    target_size = EXAMPLES_PER_CLASS * 3
    attempts = 0
    max_attempts = target_size * 50

    for identifier_type in [0, 1, 2]:
        generated = 0
    
        while generated < EXAMPLES_PER_CLASS and attempts < max_attempts:

            attempts += 1

            ctx = IdentifierContext(source)

            ex = generate_mixed_example(ctx, identifier_type, mask_mode)

            # check for fim
            if ex["text"].count("<FIM>") != 1:
                continue

            generated += 1
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



# # ============================================================
# # MASKING (CORRECT + CONSISTENT)
# # ============================================================

# def apply_mask(example, mask_mode):

#     if mask_mode == "definition":
#         definition = example["definition"].replace(example["name"], "<FIM>")
#         usage = example["usage"]

#     else:
#         definition = example["definition"]
#         usage = example["usage"].replace(example["name"], "<FIM>")

#     return {
#         "text": definition + "\n\n" + usage,
#         "label": example["label"],
#         "target": example["name"],
#         "mask_mode": mask_mode
#     }


# # ============================================================
# # MIXED (FIXED: SINGLE SOURCE OF TRUTH)
# # ============================================================

# def generate_mixed_example(ctx, mask_mode):

#     base = build_binding_example(
#         label=random.choice([VARIABLE, FUNCTION, CLASS]),
#         ctx=ctx
#     )

#     ex = apply_mask(base, mask_mode)

#     # optional noise (does NOT affect target binding)
#     ex["text"] += "\n\n" + add_mixed_distractors(ctx)

#     ex["mixed"] = True

#     return ex


# # ============================================================
# # CLEAN (SINGLE CONTEXT VERSION)
# # ============================================================

# def generate_example(label, source, mask_mode, mixed=False):

#     ctx = IdentifierContext(source)

#     base = build_binding_example(label, ctx)

#     ex = apply_mask(base, mask_mode)

#     if mixed:
#         ex["text"] += "\n\n" + add_mixed_distractors(ctx)

#     return ex


# # ============================================================
# # WRITERS
# # ============================================================
# def is_valid_fim_example(ex: dict) -> bool:
#     return ex["text"].count("<FIM>") == 1


# def write_dataset(path, source, mask_mode, mixed=False):

#     out = []

#     target_size = EXAMPLES_PER_CLASS * 3  # keep your original scale

#     ctx = IdentifierContext(source)

#     attempts = 0
#     max_attempts = target_size * 50  # safety bound

#     while len(out) < target_size and attempts < max_attempts:

#         attempts += 1

#         label = random.choice([VARIABLE, FUNCTION, CLASS])

#         ex = generate_example(
#             label=label,
#             source=source,
#             mask_mode=mask_mode,
#             mixed=mixed
#         )

#         if not is_valid_fim_example(ex):
#             continue  # ❌ discard bad sample

#         out.append(ex)

#     if len(out) < target_size:
#         raise RuntimeError(
#             f"Only generated {len(out)}/{target_size} valid samples for {path}"
#         )

#     random.shuffle(out)

#     Path(path).parent.mkdir(parents=True, exist_ok=True)

#     with open(path, "w") as f:
#         for ex in out:
#             f.write(json.dumps(ex) + "\n")

#     print("wrote", path, len(out), "attempts:", attempts)

# def write_mixed_dataset(path, source, mask_mode):

#     out = []

#     target_size = EXAMPLES_PER_CLASS * 3
#     attempts = 0
#     max_attempts = target_size * 50  # safety bound

#     while len(out) < target_size and attempts < max_attempts:

#         attempts += 1

#         ctx = IdentifierContext(source)

#         ex = generate_mixed_example(ctx, mask_mode)

#         # -------------------------
#         # VALIDATION: EXACTLY ONE FIM
#         # -------------------------
#         if ex["text"].count("<FIM>") != 1:
#             continue  # discard bad sample

#         out.append(ex)

#     if len(out) < target_size:
#         raise RuntimeError(
#             f"Only generated {len(out)}/{target_size} valid mixed samples for {path}"
#         )

#     random.shuffle(out)

#     Path(path).parent.mkdir(parents=True, exist_ok=True)

#     with open(path, "w") as f:
#         for ex in out:
#             f.write(json.dumps(ex) + "\n")

#     print("wrote", path, len(out), "attempts:", attempts)

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

        # write_dataset(base / "single_definition.jsonl", source, "definition")
        # write_dataset(base / "single_usage.jsonl", source, "usage")

        write_mixed_dataset(base / "mixed_definition.jsonl", source, "definition")
        write_mixed_dataset(base / "mixed_usage.jsonl", source, "usage")


if __name__ == "__main__":
    main()