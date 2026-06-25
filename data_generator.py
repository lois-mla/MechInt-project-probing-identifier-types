import json
import random
import string
from enum import Enum
from pathlib import Path
from transformers import AutoTokenizer
import keyword
import builtins

# ============================================================
# CONFIG
# ============================================================

EXAMPLES_PER_CLASS = 500
STEERING_EXAMPLES_PER_CLASS = 50
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
COMMON_VARIABLES = [
    # core primitives
    "data", "x", "y", "z", "i", "j", "k",
    "a", "b", "c", "n", "m",
    "value", "values", "result", "results",
    "res", "out", "output", "input",

    # state / flags
    "state", "status", "flag", "flags",
    "done", "ready", "valid", "invalid",
    "enabled", "disabled",

    # indexing / iteration
    "index", "idx", "i_idx", "j_idx", "k_idx",
    "count", "cnt", "num", "total", "size", "length",

    # structures
    "item", "items", "element", "elements",
    "node", "nodes", "edge", "edges",
    "entry", "entries",

    # key-value
    "key", "keys", "val", "value_map",
    "mapping", "map_data", "dict_data",

    # ML / data
    "model", "models", "dataset", "data_batch",
    "batch", "batches", "epoch", "step",
    "loss", "score", "scores", "accuracy",
    "prediction", "predictions",
    "label", "labels", "target", "targets",
    "feature", "features", "embedding", "embeddings",

    # training
    "train_data", "test_data", "val_data",
    "train_set", "test_set", "val_set",
    "optimizer_state", "grad", "grads",

    # IO / systems
    "request", "response", "req", "res",
    "client", "server", "connection", "conn",
    "socket", "stream", "buffer", "cache",

    # config / misc
    "config", "cfg", "settings", "options",
    "params", "hyperparams",

    # temporal
    "time", "timestamp", "start_time", "end_time",
    "duration", "elapsed",

    # temp / misc
    "tmp", "temp", "scratch", "workspace"
]

COMMON_FUNCTIONS = [
    # core operations
    "process", "compute", "calculate", "evaluate",
    "run", "execute", "apply", "perform",

    # lifecycle
    "init", "initialize", "setup", "reset", "cleanup",
    "destroy", "shutdown",

    # creation
    "create", "build", "make", "construct",
    "generate", "produce",

    # accessors
    "get", "set", "fetch", "retrieve",
    "load", "save", "store",

    # IO
    "read", "write", "open", "close",
    "send", "receive",

    # parsing / transformation
    "parse", "serialize", "deserialize",
    "transform", "convert", "encode", "decode",

    # ML
    "train", "test", "predict", "infer",
    "fit", "score", "evaluate_model",

    # data manipulation
    "filter", "sort", "merge", "split",
    "join", "group", "aggregate",

    # validation
    "validate", "check", "verify",
    "assert_valid", "ensure",

    # utilities
    "update", "modify", "patch",
    "remove", "delete", "clear",

    # math / logic
    "add", "subtract", "multiply", "divide",
    "normalize", "clip", "clamp",

    # iteration helpers
    "iterate", "loop", "traverse",

    # networking
    "connect", "disconnect", "request", "response",
    "send_request", "fetch_data"
]


COMMON_CLASSES = [
    # core app
    "User", "Account", "Session", "Profile",
    "Request", "Response", "Error", "ExceptionBase",

    # networking
    "Client", "Server", "Connection", "Socket",
    "Endpoint", "Router", "Handler",

    # ML / AI
    "Model", "Dataset", "DataLoader", "Trainer",
    "Evaluator", "Pipeline", "Transform",
    "Encoder", "Decoder", "Tokenizer",
    "EmbeddingModel", "Classifier", "Regressor",

    # config / infra
    "Config", "Settings", "Options", "Params",
    "Manager", "Controller", "Coordinator",
    "Factory", "Builder", "Registry",

    # data structures
    "Node", "Graph", "Tree", "Edge",
    "Queue", "Stack", "Heap", "Cache",
    "LinkedList", "BinaryTree",

    # storage
    "Database", "Table", "Record",
    "Repository", "Storage", "Index",

    # pipelines
    "Processor", "HandlerBase", "Service",
    "Worker", "Task", "Job",

    # misc architecture
    "Engine", "System", "Module",
    "Component", "Interface", "Adapter"
]

TOKENIZER_POOL = None


tok = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

HARD_BLOCK = set(keyword.kwlist) | {
    "True", "False", "None",
    "match", "case",
    "__name__", "__file__", "__doc__", "__package__",
}

BAD_IDENTIFIERS = (
    set(keyword.kwlist)
    | set(dir(builtins))
    | HARD_BLOCK
    | {
        "self", "cls",
        "__init__", "__call__", "__len__",
        "Exception", "BaseException",
        "PRE", "MID", "SUF", "FIM",
        "EOT", "FILL_ME",
    }
)

vocab = []

for s in tok.get_vocab():

    if not s.isidentifier():
        continue

    if len(s) < 2:
        continue

    if s in BAD_IDENTIFIERS:
        continue

    if not s.isascii():
        continue

    # must tokenize to exactly one token
    ids = tok.encode(s, add_special_tokens=False)

    if len(ids) != 1:
        continue

    vocab.append(s)


# def build_tokenizer_pool():
#     global TOKENIZER_POOL
#     if TOKENIZER_POOL is not None:
#         return TOKENIZER_POOL

#     try:
#         from transformers import AutoTokenizer
#         tok = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

#         vocab = [
#             v for v in tok.get_vocab().keys()
#             if v.isidentifier() and len(v) > 1
#         ]

#         TOKENIZER_POOL = vocab if len(vocab) > 100 else COMMON_FUNCTIONS
        
#     except:
#         TOKENIZER_POOL = COMMON_FUNCTIONS

#     return TOKENIZER_POOL


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
# kind of redundent code here. Fix this later
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
            pool = vocab
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
            pool = vocab

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
            pool = vocab
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
# ============================================================
# VARIABLE DEFINITIONS
# ============================================================
# ============================================================
# VARIABLE DEFINITIONS
# ============================================================
# ============================================================
# VARIABLE DEFINITIONS (SIMPLE / COMMON)
# ============================================================
# VAR_DEFINITION_TEMPLATES = [

#     # scalars (pure literal signal)
#     {"tpl": "<FIM> = {int1}", "type": "int"},
#     {"tpl": "<FIM> = {float1}", "type": "float"},
#     {"tpl": "<FIM> = '{str1}'", "type": "str"},
#     {"tpl": "<FIM> = True", "type": "bool"},
#     {"tpl": "<FIM> = False", "type": "bool"},

#     # containers (make constructor EXPLICIT and unique)
#     {"tpl": "<FIM> = list([1, 2])", "type": "list_int"},
#     {"tpl": "<FIM> = tuple([1, 2])", "type": "tuple_int"},
#     {"tpl": "<FIM> = dict(x=1)", "type": "dict"},
# ]

# FUNC_DEFINITION_TEMPLATES = [

#     {"tpl": "def <FIM>():\n    return 0", "args": 0},

#     {"tpl": "def <FIM>(x):\n    return x", "args": 1},

#     {"tpl": "def <FIM>(x):\n    return x + 1", "args": 1},

#     {"tpl": "def <FIM>(x):\n    return x * 2", "args": 1},

#     {"tpl": "def <FIM>(x, y):\n    return x + y", "args": 2},

#     {"tpl": "def <FIM>(x, y):\n    return x * y", "args": 2},
# ]


# CLASS_DEFINITION_TEMPLATES = [

#     {"tpl": """class <FIM>:
#     pass""", "args": 0},

#     {"tpl": """class <FIM>:
#     def __init__(self):
#         pass""", "args": 0},

#     {"tpl": """class <FIM>:
#     def __init__(self, x):
#         self.x = x""", "args": 1},

#     {"tpl": """class <FIM>:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y""", "args": 2},

#     {"tpl": """class <FIM>(Exception):
#     pass""", "args": 0},
# ]

# VAR_TEMPLATES = [

#     # scalar usage
#     {"tpl": "print(<FIM>)",
#      "types": ["int", "float", "str", "bool"]},

#     {"tpl": "x = <FIM>",
#      "types": ["int", "float", "str", "bool"]},

#     {"tpl": "if <FIM>:\n    pass",
#      "types": ["bool"]},

#     # numeric-only arithmetic (STRICT)
#     {"tpl": "x = <FIM> + 1",
#      "types": ["int", "float"]},

#     {"tpl": "x = <FIM> * 2",
#      "types": ["int", "float"]},

#     # container-only contexts (VERY IMPORTANT: no overlap)
#     {"tpl": "len_value = len(<FIM>)",
#      "types": ["list_int", "tuple_int", "dict", "str"]},

#     {"tpl": "for _ in <FIM>:\n    pass",
#      "types": ["list_int", "tuple_int"]},

#     {"tpl": "<FIM>.append(1)",
#      "types": ["list_int"]},

#     {"tpl": "<FIM>['k'] = 1",
#      "types": ["dict"]},
# ]

# FUNC_TEMPLATES = [

#     {"tpl": "result = <FIM>()", "args": 0},

#     {"tpl": "result = <FIM>(1)", "args": 1},

#     {"tpl": "result = <FIM>(a, b)", "args": 2},

#     {"tpl": "print(<FIM>(1))", "args": 1},

#     {"tpl": "return <FIM>(x)", "args": 1},
# ]

# CLASS_TEMPLATES = [

#     {"tpl": "obj = <FIM>()", "ctor_args": 0},

#     {"tpl": "obj = <FIM>(1)", "ctor_args": 1},

#     {"tpl": "instance = <FIM>()", "ctor_args": 0},

#     {"tpl": "isinstance(obj, <FIM>)", "ctor_args": None},

#     {"tpl": "class Child(<FIM>):\n    pass", "ctor_args": None},
# ]

# VAR_DEFINITION_TEMPLATES = [

#     {"tpl": "<FIM> = {int1}", "type": "int"},
#     {"tpl": "<FIM> = {float1}", "type": "float"},
#     {"tpl": "<FIM> = '{str1}'", "type": "str"},
#     {"tpl": "<FIM> = True", "type": "bool"},
#     {"tpl": "<FIM> = False", "type": "bool"},

#     {"tpl": "<FIM> = list([{int1}, {int2}])", "type": "list_int"},
#     {"tpl": "<FIM> = tuple([{int1}, {int2}])", "type": "tuple_int"},
#     {"tpl": "<FIM> = dict(k='{str1}')", "type": "dict"},
# ]


# FUNC_DEFINITION_TEMPLATES = [

#     {"tpl": """def <FIM>():
#     return {int1}""",
#      "args": 0},

#     {"tpl": """def <FIM>({var1}):
#     return {var1}""",
#      "args": 1},

#     {"tpl": """def <FIM>({var1}):
#     return {var1} + {int1}""",
#      "args": 1},

#     {"tpl": """def <FIM>({var1}):
#     return {var1} * {int1}""",
#      "args": 1},

#     {"tpl": """def <FIM>({var1}, {var2}):
#     return {var1} + {var2}""",
#      "args": 2},

#     {"tpl": """def <FIM>({var1}, {var2}):
#     return {var1} * {var2}""",
#      "args": 2},
# ]


# CLASS_DEFINITION_TEMPLATES = [

#     {"tpl": """class <FIM>:
#     pass""",
#      "args": 0},

#     {"tpl": """class <FIM>:
#     def __init__(self):
#         pass""",
#      "args": 0},

#     {"tpl": """class <FIM>:
#     def __init__(self, {var1}):
#         self.{var1} = {var1}""",
#      "args": 1},

#     {"tpl": """class <FIM>:
#     def __init__(self, {var1}, {var2}):
#         self.{var1} = {var1}
#         self.{var2} = {var2}""",
#      "args": 2},
# ]


# VAR_TEMPLATES = [

#     {"tpl": """print(<FIM>)""",
#      "types": ["int", "float", "str", "bool"]},

#     {"tpl": """{var1} = <FIM>""",
#      "types": ["int", "float", "str", "bool"]},

#     {"tpl": """if <FIM>:
#     pass""",
#      "types": ["bool"]},

#     {"tpl": """{var1} = <FIM> + {int1}""",
#      "types": ["int", "float"]},

#     {"tpl": """{var1} = <FIM> * {int1}""",
#      "types": ["int", "float"]},

#     {"tpl": """{var1} = len(<FIM>)""",
#      "types": ["list_int", "tuple_int", "dict", "str"]},

#     {"tpl": """for {var1} in <FIM>:
#     pass""",
#      "types": ["list_int", "tuple_int"]},

#     {"tpl": """<FIM>.append({int1})""",
#      "types": ["list_int"]},

#     {"tpl": """<FIM>['{str1}'] = {int1}""",
#      "types": ["dict"]},
# ]


# FUNC_TEMPLATES = [

#     {"tpl": """{var1} = <FIM>()""",
#      "args": 0},

#     {"tpl": """{var1} = <FIM>({int1})""",
#      "args": 1},

#     {"tpl": """{var1} = <FIM>({var2}, {var3})""",
#      "args": 2},

#     {"tpl": """print(<FIM>({int1}))""",
#      "args": 1},

#     {"tpl": """{var1} = <FIM>({var2})""",
#      "args": 1},
# ]


# CLASS_TEMPLATES = [

#     {"tpl": """{var1} = <FIM>()""",
#      "ctor_args": 0},

#     {"tpl": """{var1} = <FIM>({int1})""",
#      "ctor_args": 1},

#     {"tpl": """{var1} = <FIM>({int1}, {int2})""",
#      "ctor_args": 2},

#     {"tpl": """isinstance({var1}, <FIM>)""",
#      "ctor_args": None},

#     {"tpl": """class {cls1}(<FIM>):
#     pass""",
#      "ctor_args": None},
# ]



# ============================================================
# VARIABLE DEFINITIONS
# ============================================================

VAR_DEFINITION_TEMPLATES = [

    {"tpl": "<FIM> = {int1}", "type": "int"},
    {"tpl": "<FIM> = {int1} + {int2}", "type": "int"},
    {"tpl": "<FIM> = {int1} * {int2}", "type": "int"},

    {"tpl": "<FIM> = {float1}", "type": "float"},
    {"tpl": "<FIM> = {float1} * 2.0", "type": "float"},

    {"tpl": "<FIM> = '{str1}'", "type": "str"},
    {"tpl": "<FIM> = '{str1}' + '{str2}'", "type": "str"},

    {"tpl": "<FIM> = True", "type": "bool"},
    {"tpl": "<FIM> = False", "type": "bool"},

    {"tpl": "<FIM> = [{int1}, {int2}, {int3}]", "type": "list_int"},
    {"tpl": "<FIM> = list([{int1}, {int2}])", "type": "list_int"},

    {"tpl": "<FIM> = ({int1}, {int2})", "type": "tuple_int"},
    {"tpl": "<FIM> = tuple([{int1}, {int2}])", "type": "tuple_int"},

    {"tpl": "<FIM> = {{'{str1}': {int1}}}", "type": "dict"},
    {"tpl": "<FIM> = dict(k={int1})", "type": "dict"},
]


# ============================================================
# FUNCTION DEFINITIONS
# ============================================================

FUNC_DEFINITION_TEMPLATES = [

    {"tpl": """def <FIM>():
    return {int1}""",
     "args": 0},

    {"tpl": """def <FIM>():
    return '{str1}'""",
     "args": 0},

    {"tpl": """def <FIM>({var1}):
    return {var1}""",
     "args": 1},

    {"tpl": """def <FIM>({var1}):
    return {var1} + {int1}""",
     "args": 1},

    {"tpl": """def <FIM>({var1}):
    return {var1} * {int1}""",
     "args": 1},

    {"tpl": """def <FIM>({var1}):
    return len({var1})""",
     "args": 1},

    {"tpl": """def <FIM>({var1}, {var2}):
    return {var1} + {var2}""",
     "args": 2},

    {"tpl": """def <FIM>({var1}, {var2}):
    return {var1} * {var2}""",
     "args": 2},

    {"tpl": """def <FIM>({var1}, {var2}, {var3}):
    return {var1}""",
     "args": 3},

    {"tpl": """def <FIM>({var1}, {var2}, {var3}):
    return {var1} + {var2} + {var3}""",
     "args": 3},
]


# ============================================================
# CLASS DEFINITIONS
# ============================================================

CLASS_DEFINITION_TEMPLATES = [

    {"tpl": """class <FIM>:
    pass""",
     "args": 0},

    {"tpl": """class <FIM>:
    def __init__(self):
        pass""",
     "args": 0},

    {"tpl": """class <FIM>:
    def __init__(self, {var1}):
        self.{var1} = {var1}""",
     "args": 1},

    {"tpl": """class <FIM>:
    def __init__(self, {var1}, {var2}):
        self.{var1} = {var1}
        self.{var2} = {var2}""",
     "args": 2},

    {"tpl": """class <FIM>:
    def __init__(self):
        self.value = {int1}""",
     "args": 0},

    {"tpl": """class <FIM>:
    def __init__(self, {var1}):
        self.data = {var1}

    def get(self):
        return self.data""",
     "args": 1},

    {"tpl": """class <FIM>:
    def __init__(self, {var1}, {var2}, {var3}):
        self.a = {var1}
        self.b = {var2}
        self.c = {var3}""",
     "args": 3},
]


# ============================================================
# VARIABLE USAGE
# ============================================================

VAR_TEMPLATES = [

    {"tpl": "print(<FIM>)",
     "types": ["int", "float", "str", "bool"]},

    {"tpl": "{var1} = <FIM>",
     "types": ["int", "float", "str", "bool"]},

    {"tpl": "if <FIM>:\n    pass",
     "types": ["bool"]},

    {"tpl": "{var1} = <FIM> + {int1}",
     "types": ["int", "float"]},

    {"tpl": "{var1} = <FIM> * {int1}",
     "types": ["int", "float"]},

    {"tpl": "{var1} = len(<FIM>)",
     "types": ["list_int", "tuple_int", "dict", "str"]},

    {"tpl": "{var1} = len(<FIM>) + {int1}",
     "types": ["list_int", "tuple_int", "dict", "str"]},

    {"tpl": "for {var1} in <FIM>:\n    pass",
     "types": ["list_int", "tuple_int"]},

    {"tpl": "{var1} = <FIM>[0]",
     "types": ["list_int", "tuple_int"]},

    {"tpl": "<FIM>.append({int1})",
     "types": ["list_int"]},

    {"tpl": "<FIM>['{str1}'] = {int1}",
     "types": ["dict"]},

    {"tpl": "{var1} = list(<FIM>.keys())",
     "types": ["dict"]},
]


# ============================================================
# FUNCTION USAGE
# ============================================================

FUNC_TEMPLATES = [

    {"tpl": "{var1} = <FIM>()",
     "args": 0},

    {"tpl": "{var1} = <FIM>({int1})",
     "args": 1},

    {"tpl": "{var1} = <FIM>({var2})",
     "args": 1},

    {"tpl": "{var1} = <FIM>({var2}, {var3})",
     "args": 2},

    {"tpl": "{var1} = <FIM>({var1}, {var2}, {var3})",
     "args": 3},

    {"tpl": "print(<FIM>())",
     "args": 0},

    {"tpl": "print(<FIM>({int1}))",
     "args": 1},

    {"tpl": "print(<FIM>({var1}, {var2}))",
     "args": 2},

    {"tpl": "{var1} = <FIM>({int1}, {int2})",
     "args": 2},
]


# ============================================================
# CLASS USAGE
# ============================================================

CLASS_TEMPLATES = [

    {"tpl": "{var1} = <FIM>()",
     "ctor_args": 0},

    {"tpl": "{var1} = <FIM>({int1})",
     "ctor_args": 1},

    {"tpl": "{var1} = <FIM>({int1}, {int2})",
     "ctor_args": 2},

    {"tpl": "{var1} = <FIM>({int1}, {int2}, {int3})",
     "ctor_args": 3},

    {"tpl": "isinstance({var1}, <FIM>)",
     "ctor_args": None},

    {"tpl": "issubclass({cls1}, <FIM>)",
     "ctor_args": None},

    {"tpl": "class {cls1}(<FIM>):\n    pass",
     "ctor_args": None},
]






















# # ============================================================
# # VARIABLE DEFINITIONS (UNAMBIGUOUS)
# # ============================================================

# VAR_DEFINITION_TEMPLATES = [

#     # identity assignment (strongest signal)
#     {"tpl": "<FIM> = {int1}", "type": "int"},
#     {"tpl": "<FIM> = {float1}", "type": "float"},
#     {"tpl": "<FIM> = '{str1}'", "type": "str"},
#     {"tpl": "<FIM> = True", "type": "bool"},
#     {"tpl": "<FIM> = False", "type": "bool"},

#     # STRUCTURAL containers ONLY (no arithmetic ambiguity)
#     {"tpl": "<FIM> = list([{int1}, {int2}])", "type": "list_int"},
#     {"tpl": "<FIM> = tuple([{int1}, {int2}])", "type": "tuple_int"},
#     {"tpl": "<FIM> = dict(key='{str1}', value={int1})", "type": "dict"},

#     {"tpl": "<FIM> = list()", "type": "list_int"},
#     {"tpl": "<FIM> = dict()", "type": "dict"},
# ]


# # ============================================================
# # FUNCTION DEFINITIONS (UNAMBIGUOUS)
# # ============================================================

# FUNC_DEFINITION_TEMPLATES = [

#     {"tpl": """def <FIM>():
#     return {int1}""", "args": 0},

#     {"tpl": """def <FIM>(x):
#     return x""", "args": 1},

#     {"tpl": """def <FIM>(x):
#     return x + {int1}""", "args": 1},

#     {"tpl": """def <FIM>(x):
#     return x * {int1}""", "args": 1},

#     {"tpl": """def <FIM>(x, y):
#     return x + y""", "args": 2},

#     {"tpl": """def <FIM>(x, y):
#     return x * y""", "args": 2},
# ]


# # ============================================================
# # CLASS DEFINITIONS (UNAMBIGUOUS)
# # ============================================================

# CLASS_DEFINITION_TEMPLATES = [

#     {"tpl": """class <FIM>:
#     pass""", "args": 0},

#     {"tpl": """class <FIM>:
#     def __init__(self):
#         pass""", "args": 0},

#     {"tpl": """class <FIM>:
#     def __init__(self, x):
#         self.x = x""", "args": 1},

#     {"tpl": """class <FIM>:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y""", "args": 2},

#     {"tpl": """class <FIM>(Exception):
#     pass""", "args": 0},
# ]


# # ============================================================
# # VARIABLE USAGE (STRICTLY NON-CALL POSITION)
# # ============================================================
# VAR_TEMPLATES = [

#     # pure evaluation contexts
#     {"tpl": "print(<FIM>)",
#      "types": ["int", "float", "str", "bool"]},

#     {"tpl": "log_value = str(<FIM>)",
#      "types": ["int", "float", "str", "bool"]},

#     {"tpl": "if bool(<FIM>):\n    pass",
#      "types": ["int", "float", "str", "bool"]},

#     # container-only contexts
#     {"tpl": "len_value = len(<FIM>)",
#      "types": ["list_int", "tuple_int", "dict", "str"]},

#     {"tpl": "for _ in <FIM>:\n    pass",
#      "types": ["list_int", "tuple_int"]},

#     {"tpl": "<FIM>.append(1)",
#      "types": ["list_int"]},

#     {"tpl": "first = <FIM>[0]",
#      "types": ["list_int", "tuple_int", "str"]},

#     {"tpl": "<FIM>['k'] = 1",
#      "types": ["dict"]},

#     {"tpl": "assert <FIM> is not None",
#      "types": ["int", "float", "str", "bool", "list_int", "tuple_int", "dict"]},
# ]

# # ============================================================
# # FUNCTION USAGE (STRICT CALL CONTEXT ONLY)
# # ============================================================
# FUNC_TEMPLATES = [

#     {"tpl": "<FIM>()", "args": 0},

#     {"tpl": "<FIM>({int1})", "args": 1},

#     {"tpl": "<FIM>({var2}, {var3})", "args": 2},

#     {"tpl": "print(<FIM>())", "args": 0},

#     {"tpl": "return <FIM>({var1})", "args": 1},

#     {"tpl": "result = <FIM>({int1})", "args": 1},
# ]


# # ============================================================
# # CLASS USAGE (STRICTLY INSTANTIATION / TYPE CONTEXT)
# # ============================================================
# CLASS_TEMPLATES = [

#     {"tpl": "obj = <FIM>()", "ctor_args": 0},

#     {"tpl": "obj = <FIM>({int1})", "ctor_args": 1},

#     {"tpl": "instance = <FIM>()", "ctor_args": 0},

#     {"tpl": "type(obj) is <FIM>", "ctor_args": None},

#     {"tpl": "isinstance(obj, <FIM>)", "ctor_args": None},

#     {"tpl": "class Child(<FIM>):\n    pass", "ctor_args": None},
# ]
# VAR_DEFINITION_TEMPLATES = [

#     {"tpl": """<FIM> = {int1}""", "type": "int"},
#     {"tpl": """<FIM> = {float1}""", "type": "float"},
#     {"tpl": """<FIM> = {str1}""", "type": "str"},
#     {"tpl": """<FIM> = True""", "type": "bool"},
#     {"tpl": """<FIM> = False""", "type": "bool"},
#     # {"tpl": """<FIM> = None""", "type": "none"},

#     {"tpl": """<FIM> = {int1} + {int2}""", "type": "int"},
#     {"tpl": """<FIM> = {int1} * {int2}""", "type": "int"},

#     {"tpl": """<FIM> = [{int1}, {int2}]""", "type": "list_int"},
#     {"tpl": """<FIM> = []""", "type": "list_int"},

#     {"tpl": """<FIM> = ({int1}, {int2})""", "type": "tuple_int"},

#     {"tpl": """<FIM> = {{'{str1}': {int1}}}""", "type": "dict"},
#     {"tpl": """<FIM> = {{}}""", "type": "dict"},
# ]


# # ============================================================
# # FUNCTION DEFINITIONS (VERY COMMON)
# # ============================================================

# FUNC_DEFINITION_TEMPLATES = [

#     {"tpl": """def <FIM>():
#     return {int1}""",
#      "args": 0},

#     {"tpl": """def <FIM>(x):
#     return x""",
#      "args": 1},

#     {"tpl": """def <FIM>(x):
#     return x + {int1}""",
#      "args": 1},

#     {"tpl": """def <FIM>(x):
#     return x * {int1}""",
#      "args": 1},

#     {"tpl": """def <FIM>(x, y):
#     return x + y""",
#      "args": 2},

#     {"tpl": """def <FIM>(x, y):
#     return x * y""",
#      "args": 2},
# ]


# # ============================================================
# # CLASS DEFINITIONS (VERY COMMON)
# # ============================================================

# CLASS_DEFINITION_TEMPLATES = [

#     {"tpl": """class <FIM>:
#     pass""",
#      "args": 0},

#     {"tpl": """class <FIM>:
#     def __init__(self):
#         pass""",
#      "args": 0},

#     {"tpl": """class <FIM>:
#     def __init__(self, x):
#         self.x = x""",
#      "args": 1},

#     {"tpl": """class <FIM>:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y""",
#      "args": 2},

#     {"tpl": """class <FIM>(Exception):
#     pass""",
#      "args": 0},
# ]


# # ============================================================
# # VARIABLE USAGE TEMPLATES
# # ============================================================

# VAR_TEMPLATES = [

#     {"tpl": """print(<FIM>)""",
#      "types": ["int", "float", "str", "bool"]},

#     {"tpl": """{var1} = <FIM>""",
#      "types": ["int", "float", "str", "bool",
#                "list_int", "tuple_int", "dict"]},

#     {"tpl": """if <FIM>:
#     pass""",
#      "types": ["int", "float", "str", "bool",
#                "list_int", "tuple_int", "dict"]},

#     {"tpl": """{var1} = <FIM> + {int1}""",
#      "types": ["int", "float"]},

#     {"tpl": """{var1} = <FIM> * {int1}""",
#      "types": ["int", "float"]},

#     {"tpl": """if <FIM> > {int1}:
#     pass""",
#      "types": ["int", "float"]},

#     {"tpl": """{var1} = len(<FIM>)""",
#      "types": ["list_int", "tuple_int", "dict", "str"]},

#     {"tpl": """for {var1} in <FIM>:
#     pass""",
#      "types": ["list_int", "tuple_int"]},

#     {"tpl": """<FIM>.append({int1})""",
#      "types": ["list_int"]},

#     {"tpl": """{var1}, {var2} = <FIM>""",
#      "types": ["tuple_int"]},

#     {"tpl": """<FIM>['{str1}'] = {int1}""",
#      "types": ["dict"]},

#     {"tpl": """assert <FIM> is not None""",
#      "types": ["int", "float", "str", "bool",
#                "list_int", "tuple_int", "dict"]},
# ]


# # ============================================================
# # FUNCTION USAGE TEMPLATES
# # ============================================================

# FUNC_TEMPLATES = [

#     {"tpl": """{var1} = <FIM>()""",
#      "args": 0},

#     {"tpl": """{var1} = <FIM>({int1})""",
#      "args": 1},

#     {"tpl": """{var1} = <FIM>({var2}, {var3})""",
#      "args": 2},

#     {"tpl": """result = <FIM>()""",
#      "args": 0},

#     {"tpl": """result = <FIM>({int1})""",
#      "args": 1},

#     {"tpl": """result = <FIM>({var1})""",
#      "args": 1},

#     {"tpl": """print(<FIM>({int1}))""",
#      "args": 1},

#     {"tpl": """return <FIM>({var1})""",
#      "args": 1},

#     {"tpl": """x = <FIM>({int1})""",
#      "args": 1},
# ]


# # ============================================================
# # CLASS USAGE TEMPLATES
# # ============================================================

# CLASS_TEMPLATES = [

#     {"tpl": """obj = <FIM>()""",
#      "ctor_args": 0},

#     {"tpl": """obj = <FIM>({int1})""",
#      "ctor_args": 1},

#     {"tpl": """obj = <FIM>({int1}, {int2})""",
#      "ctor_args": 2},

#     {"tpl": """instance = <FIM>()""",
#      "ctor_args": 0},

#     {"tpl": """instance = <FIM>({int1})""",
#      "ctor_args": 1},

#     {"tpl": """instance = <FIM>({int1}, {int2})""",
#      "ctor_args": 2},

#     {"tpl": """if isinstance(obj, <FIM>):
#     pass""",
#      "ctor_args": None},

#     {"tpl": """class {cls1}(<FIM>):
#     pass""",
#      "ctor_args": None},

#     {"tpl": """raise <FIM>()""",
#      "ctor_args": 0},
# ]

# VAR_TEMPLATES = [

#     {"tpl": """{var1} = <FIM> + 5""",
#      "types": ["int"]},

#     {"tpl": """{var1} = <FIM> * 1.5""",
#      "types": ["float"]},

#     {"tpl": """{var1} = <FIM>.replace('a', 'b')""",
#      "types": ["str"]},

#     {"tpl": """<FIM>.append(3)""",
#      "types": ["list_int"]},

#     {"tpl": """{var1} = <FIM>[0]""",
#      "types": ["list_int"]},

#     {"tpl": """{var1} = len(<FIM>)""",
#      "types": ["list_int", "str"]},
# ]

# FUNC_TEMPLATES = [

#     {"tpl": """{var1} = <FIM>({int1})""",
#      "args": 1},

#     {"tpl": """{var1} = <FIM>({int1}, {int2})""",
#      "args": 2},

#     {"tpl": """{var1} = <FIM>({var2}, {var3}, {var4})""",
#      "args": 3},

#     {"tpl": """{var1} = <FIM>('a', 'b', 'c', 'd')""",
#      "args": 4},
# ]

# CLASS_TEMPLATES = [

#     {"tpl": """{var1} = <FIM>()""",
#      "ctor_args": 0},

#     {"tpl": """{var1} = <FIM>('x')""",
#      "ctor_args": 1},

#     {"tpl": """{var1} = <FIM>('x', 'y')""",
#      "ctor_args": 2},

#     {"tpl": """{var1} = <FIM>(1, 2)""",
#      "ctor_args": 2},

#     {"tpl": """{var1} = <FIM>('a')
# {var2} = {var1}.{method1}('b')""",
#      "ctor_args": 1},

#     {"tpl": """{var1} = <FIM>('a', 'b')
# {var2} = {var1}.{method1}('c')""",
#      "ctor_args": 2},
# ]



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
    print(def_t)

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
        "var": ids["var"],
        "fun": ids["func"],
        "cls": ids["cls"],
    }


def generate_single_example(ctx, identifier_type, mask_mode="usage"):

    ids = ctx.identifiers()

    if identifier_type == 0:
        var_def, var_use = sample_pair(VAR_DEFINITION_TEMPLATES, VAR_TEMPLATES, 0)
        def_s, use_s = instantiate_single(var_def["tpl"], var_use, ids["var"], ctx, mask_mode=mask_mode)
        target = ids["var"]
    elif identifier_type == 1:
        func_def, func_use = sample_pair(FUNC_DEFINITION_TEMPLATES, FUNC_TEMPLATES, 1)
        def_s, use_s = instantiate_single(func_def["tpl"], func_use, ids["func"], ctx, mask_mode=mask_mode)
        target = ids["func"]
    else: 
        cls_def, cls_use = sample_pair(CLASS_DEFINITION_TEMPLATES, CLASS_TEMPLATES, 2)
        def_s, use_s = instantiate_single(cls_def["tpl"], cls_use, ids["cls"], ctx, mask_mode=mask_mode)
        target = ids["cls"]

    text = '\n'.join([def_s, use_s])

    return {
        "text": text,
        "label": identifier_type,
        "target": target,
        "mask_mode": mask_mode,
    }



def write_dataset(path, source, mask_mode, examples_per_class, mixed=True):

    out = []

    target_size = examples_per_class * 3
    attempts = 0
    max_attempts = target_size * 50

    for identifier_type in [0, 1, 2]:
        generated = 0
    
        while generated < examples_per_class and attempts < max_attempts:

            attempts += 1

            ctx = IdentifierContext(source)

            if mixed == True:
                ex = generate_mixed_example(ctx, identifier_type, mask_mode)
            else:
                ex = generate_single_example(ctx, identifier_type, mask_mode)

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



# -------------------------------------------------------------------------------------------
def main():

    for source in [
        IdentifierSource.LETTERS,
        IdentifierSource.TOKENIZER,
        IdentifierSource.COMMON,
    ]:
        
        base = Path("datasets/final") / source.value

        write_dataset(base / "single_definition.jsonl", source, "definition", EXAMPLES_PER_CLASS, mixed=False)
        write_dataset(base / "single_usage.jsonl", source, "usage", EXAMPLES_PER_CLASS, mixed=False)

        write_dataset(base / "mixed_definition.jsonl", source, "definition", EXAMPLES_PER_CLASS, mixed=True)
        write_dataset(base / "mixed_usage.jsonl", source, "usage", EXAMPLES_PER_CLASS, mixed=True)
        write_dataset(base / "steering_definition.jsonl", source, "definition", STEERING_EXAMPLES_PER_CLASS, mixed=True)
        write_dataset(base / "steering_usage.jsonl", source, "usage", STEERING_EXAMPLES_PER_CLASS, mixed=True)


if __name__ == "__main__":
    main()