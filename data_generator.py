import random
import string

# --------------------------------------------------
# Random utilities
# --------------------------------------------------

def rand_name():
    return random.choice(string.ascii_lowercase)

def rand_cap():
    return random.choice(string.ascii_uppercase)

def rand_int():
    return str(random.randint(1, 999))

def rand_float():
    return f"{random.uniform(1, 100):.3f}"

def rand_string():
    return f"'{rand_cap()}'"

def rand_list():
    return f"[{random.randint(1,20)}, {random.randint(1,20)}]"

def rand_set():
    return f"{{{random.randint(1,20)}, {random.randint(1,20)}}}"

def rand_dict():
    return f"{{'{rand_name()}': {random.randint(1,5)}}}"

def rand_value():
    generators = [
        rand_int,
        rand_float,
        rand_string,
        rand_list,
        rand_set,
        rand_dict,
    ]
    return random.choice(generators)()

# --------------------------------------------------
# Function generation
# --------------------------------------------------

def generate_function():
    name = rand_name()

    n_args = random.randint(2, 5)
    args = [rand_name() for _ in range(n_args)]

    lines = [f"def {name}({', '.join(args)}):"]

    n_internal = random.randint(2, 5)

    for _ in range(n_internal):
        lhs = rand_cap()
        rhs_var = random.choice(args)

        op = random.choice([
            f"{rhs_var} + {random.randint(1,10)}",
            f"{rhs_var} - {random.randint(1,10)}",
            f"{rhs_var} * {random.randint(1,10)}",
            f"{rhs_var} / {random.randint(1,10)}",
            f"{rhs_var} ** {random.randint(1,3)}",
        ])

        lines.append(f"    {lhs} = {op}")

    lines.append(f"    return {random.choice(args)}")

    return name, "\n".join(lines)

# --------------------------------------------------
# Class generation
# --------------------------------------------------

def generate_class():
    cname = rand_name()

    n_fields = random.randint(2, 4)
    fields = [rand_name() for _ in range(n_fields)]

    lines = [f"class {cname}:"]

    init_args = ", ".join(fields)

    lines.append(f"    def __init__(self, {init_args}):")

    for f in fields:
        lines.append(f"        self.{f} = {f}")

    n_methods = random.randint(1, 3)

    for _ in range(n_methods):
        m = rand_name()

        op = random.choice([
            "x + 2",
            "x - 2",
            "x * 2",
            "x / 2"
        ])

        lines.append("")
        lines.append(f"    def {m}(self, x):")
        lines.append(f"        return {op}")

    n_attrs = random.randint(0, 2)

    for _ in range(n_attrs):
        lines.append(f"    {rand_name()} = {rand_value()}")

    return cname, "\n".join(lines)

# --------------------------------------------------
# Assignment generation
# --------------------------------------------------

def generate_assignment(symbols):
    lhs = rand_name()

    target = random.choice(symbols)

    if target["type"] == "class":

        n_args = random.randint(2, 4)

        args = ", ".join(rand_value() for _ in range(n_args))

        return f"{lhs} = {target['name']}({args})"

    else:

        n_args = random.randint(2, 4)

        args = ", ".join(rand_value() for _ in range(n_args))

        return f"{lhs} = {target['name']}({args})"

# --------------------------------------------------
# Example generation
# --------------------------------------------------

def generate_example():

    blocks = []
    entities = []

    n_functions = random.randint(1, 2)
    n_classes = random.randint(1, 2)

    for _ in range(n_functions):
        name, text = generate_function()
        blocks.append(text)

        entities.append({
            "name": name,
            "type": "function",
            "id": len(entities) + 1
        })

    for _ in range(n_classes):
        name, text = generate_class()
        blocks.append(text)

        entities.append({
            "name": name,
            "type": "class",
            "id": len(entities) + 1
        })

    random.shuffle(blocks)

    program = "\n\n".join(blocks)

    # random assignments
    n_assignments = random.randint(3, 6)

    assign_lines = []

    for _ in range(n_assignments):
        assign_lines.append(generate_assignment(entities))

    # choose target
    target = random.choice(entities)

    # generate FIM query
    if target["type"] == "class":

        fim = (
            f"{rand_name()} = <FIM>"
            f"({', '.join(rand_value() for _ in range(2))})"
        )

    else:

        fim = (
            f"{rand_name()} = <FIM>"
            f"({', '.join(rand_value() for _ in range(2))})"
        )

    assign_lines.append(fim)

    text = (
        program
        + "\n\n"
        + "\n".join(assign_lines)
        + "\n\n>>>"
        + target["name"]
        + f"\nID:{target['id']}"
    )

    return text

# --------------------------------------------------
# Dataset generation
# --------------------------------------------------

for _ in range(5):
    print(generate_example())
    print("\n" + "#" * 80 + "\n")