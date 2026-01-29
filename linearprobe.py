"""
This file contains the necessary class and functions to train and use a linear probe.

get_residual_activations() takes as input a list of strings (the prompts we will use to train the linear probe),
and returns the residual activations for the <MID> token from the specified layer. (so output shape: [N, d_model])

The LinearProbe class is a simple linear layer that takes the residual activations as input and outputs class logits.
In order to use the LinearProbe, we will need to feed the examples to get_residual_activations (this is x for training),
and get a tensor with corresponding label indices (this is y for training).

Still need to add how to use the learned weight matrix W, to get insight into possible
feature directions for the different identifier types.
"""

# importing stuff
import transformer_lens
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch.nn.functional as F
from utils import read_fim_dataset, get_prompts_and_IDS


# model_id = "meta-llama/Llama-2-7b-hf"

# Load tokenizer and model
# CodeLlama 7B has d_model=4096, n_layers = 32, n_heads=32, d_head=128
model_id = "codellama/CodeLlama-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # padding works with eos tokens

model = transformer_lens.HookedTransformer.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device="cuda"
)
# def fill_in_middle(prefix: str, suffix: str):
    
#     return f"▁<PRE> {prefix} ▁<SUF>{suffix} ▁<MID>"

# # print the loss of running a prompt
# prefix = """def """
# suffix = """(x, y):
#     return x + y

# sum = addition(2, 3)
# """
# prompt = fill_in_middle(prefix, suffix)

# loss = model(prompt, return_type="loss")
# print("Model loss:", loss)

# # caching activations
# prompt_tokens = model.to_tokens(prompt)
# prompt_logits, prompt_cache = model.run_with_cache(prompt_tokens, remove_batch_dim=True)
# # print the shape of the logits and cache
# print("Logits shape:", prompt_logits.shape)
# activations = prompt_cache["mlp_out", 1]
# print("Activations shape:", activations.shape)

# prefix = """# function that adds two numbers
# def """
# suffix = """(x, y):
#     return x + y

# # add 2 and 3 together
# sum = addition(2, 3)
# """

# prompt_2 = fill_in_middle(prefix, suffix)

# prefix = """# set var to 0
# """
# suffix = """ = 0
# # add 1 to var
# var += 1
# """

# prompt_3 = fill_in_middle(prefix, suffix)

# prefix = ""
# suffix = """ = 0
# var += 1
# """

# prompt_4 = fill_in_middle(prefix, suffix)


# prefix = """class """ 
# suffix = """:
#     def __init__(self):
#         self.data = []

#     def add(self, x):
#         self.data.append(x)

# bag = Bag()"""

# prompt_5 = fill_in_middle(prefix, suffix)


@torch.inference_mode()
def get_residual_activations(
    model: transformer_lens.HookedTransformer,
    data: list[str],
    layer: int,  
    resid_type: str = "mlp_out", # where in the layer you retrieve activations
    batch_size: int = 8, # for computational efficiency
    max_length: int = 256, # how long is a prompt allowed to be
    device: str = "cuda",
) -> torch.Tensor:
    """
    Returns activations suitable for linear probing.
    Returns the activations at the <MID> position. (last token before padding)

    Output shape: [N, D]
        N = number of sequences (# prompts)
        D = d_model (hidden size)

    """

    all_acts = []

    # Process data in batches to save GPU memory
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]

        # Convert text → token IDs
        # Padding ensures all sequences in batch have same L 
        # Tokenize with HF tokenizer (this supports padding/truncation)
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        tokens = enc["input_ids"].to(device) # shape: [batch, pos]

        hook_name = f"blocks.{layer}.hook_{resid_type}"

        _, cache = model.run_with_cache(
            tokens,
            return_type=None,
            stop_at_layer=layer + 1,
            names_filter=[hook_name],
        )

        acts = cache[hook_name]


        # Run the model and cache *all* intermediate activations up until necessary layer
        # _, cache = model.run_with_cache(tokens, return_type=None)
        # _, cache = model.run_with_cache(
        #     tokens,
        #     return_type=None,
        #     stop_at_layer=layer + 1,
        #     names_filter=[(resid_type, layer)],
        # )
        # # Extract a specific activation:
        # # e.g. ("mlp_out", layer) → [batch, pos, d_model]
        # # pos: sequence length after padding
        # acts = cache[(resid_type, layer)] 

        # Gather the activations at the ▁<MID> position
        mid_token_id = tokenizer.convert_tokens_to_ids("▁<MID>")
        # print("MID token ID:", mid_token_id)

        mid_acts = []

        # Iterate over each sequence in the batch
        for b in range(tokens.size(0)):
            # Find all positions in this sequence where the token ID equals <MID>
            # (tokens[b] has shape [pos])
            mid_pos = (tokens[b] == mid_token_id).nonzero(as_tuple=True)[0]

            # Sanity check: we expect exactly ONE <MID> token per prompt
            # If this fails, something is wrong with prompt construction or tokenization
            assert len(mid_pos) == 1, (
                f"Expected exactly one <MID> token, "
                f"but found {len(mid_pos)} in sequence {b}"
            )

            # Extract the position index of the <MID> token
            pos = mid_pos.item()

            # --- SANITY CHECK ---
            # Decode and print the token at this position to verify correctness
            # This should print "<MID>" for CodeLLaMA
            # found_token = tokenizer.decode(tokens[b, pos])
            # print(f"Sequence {b}: found token at <MID> position → {found_token}")

            # Extract the activation vector at the <MID> position
            # acts has shape [batch, pos, d_model]
            mid_acts.append(acts[b, pos])

        # Stack all <MID> activations into a single tensor
        # Final shape: [batch, d_model]
        mid_acts = torch.stack(mid_acts)

        # Move to CPU so GPU memory can be freed
        all_acts.append(mid_acts.cpu())

    # Stack all batches → [N, d_model]
    return torch.cat(all_acts, dim=0)

# res_activations = get_residual_activations(
#     model,
#     data=[prompt, prompt_2, prompt_3, prompt_4, prompt_5],
#     layer=30,
#     resid_type="mlp_out"
# )

# get the activations at the <MID> position (last token of the prompt)

# print("Residual activations shape for the last non-padding token position:", res_activations.shape)

class LinearProbe(nn.Module):
    """
    A linear probe for token-level representations.

    This probe takes a single residual stream vector
    (e.g. mlp_out at the <MID> position) and predicts
    an identifier type.
    """

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        # model consists of 1 linear layer, so one matrix W and vector b
        # self.linear.weight retrieves W
        # self.linear.bias retrieves b
        self.linear = nn.Linear(d_model, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, d_model]
        returns logits: [batch, num_classes]
        """
        return self.linear(x)

def train_probe(
    probe: LinearProbe,
    X: torch.Tensor, # [N (#examples), d_model]
    y: torch.Tensor, # [N] (class indices; 0, 1, 2)
    num_epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 32,
):
    probe.train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() # for multi-class classification
    # (PyTorch internally uses one-hot targets)

    for epoch in range(num_epochs):
        # Shuffle the data at the beginning of each epoch
        perm = torch.randperm(X.size(0))
        X_shuf = X[perm]
        y_shuf = y[perm]

        # Iterate over mini-batches
        for i in range(0, X.size(0), batch_size):
            xb = X_shuf[i:i+batch_size]
            yb = y_shuf[i:i+batch_size]

            logits = probe(xb) # calls probe.forward() implicitly
            loss = criterion(logits, yb)

            # Backpropagation, baby
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

@torch.inference_mode()
def predict_probe(
    probe: LinearProbe,
    X: torch.Tensor, # [N, d_model]
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Predict class labels using a trained linear probe.

    X: [N, d_model]
    Returns: [N] predicted class indices
    """
    probe.eval()
    preds = []

    for i in range(0, X.size(0), batch_size):
        xb = X[i:i+batch_size]
        logits = probe(xb)
        predicted = torch.argmax(logits, dim=-1)
        preds.append(predicted.cpu())

    return torch.cat(preds, dim=0)

@torch.inference_mode()
def evaluate_probe(
    probe: LinearProbe,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 32,
) -> float:
    """
    Evaluate probe accuracy.

    X: [N, d_model]
    y: [N]
    """
    probe.eval()
    correct = 0
    total = 0

    for i in range(0, X.size(0), batch_size):
        xb = X[i:i+batch_size]
        yb = y[i:i+batch_size]

        logits = probe(xb)
        preds = torch.argmax(logits, dim=-1)

        correct += (preds == yb).sum().item()
        total += yb.size(0)

    return correct / total


def get_class_steering_vector(
    probe: LinearProbe,
    class_id: int,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Returns steering vector s for a given class.
    s has shape [d_model].
    """
    W = probe.linear.weight.detach()   # [C, d_model]
    s = W[class_id].clone()

    if normalize:
        s = s / (s.norm() + 1e-8)

    return s


def get_contrastive_steering_vector(
    probe: LinearProbe,
    pos_class: int,
    neg_class: int,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Steering vector that pushes toward pos_class and away from neg_class.
    """
    W = probe.linear.weight.detach()
    s = W[pos_class] - W[neg_class]

    if normalize:
        s = s / (s.norm() + 1e-8)

    return s

def make_steering_hook(
    steering_vector: torch.Tensor,   # [d_model]
    alpha: float,
    mid_token_id: int,
):
    """
    Returns a hook function that adds alpha * steering_vector
    to the activation at the <MID> token position.
    """

    steering_vector = steering_vector.to("cuda")

    def hook_fn(acts: torch.Tensor, hook):
        """
        acts: [batch, seq_len, d_model]
        """
        # We need access to the tokens to find <MID>.
        # TransformerLens stores tokens on the hook context:
        tokens = hook.ctx["tokens"]   # [batch, seq_len]

        for b in range(tokens.size(0)):
            mid_pos = (tokens[b] == mid_token_id).nonzero(as_tuple=True)[0]
            if len(mid_pos) != 1:
                continue  # safety

            pos = mid_pos.item()
            acts[b, pos] += alpha * steering_vector

        return acts

    return hook_fn

@torch.inference_mode()
def run_with_steering(
    model: transformer_lens.HookedTransformer,
    prompt: str,
    steering_vector: torch.Tensor,
    alpha: float,
    layer: int,
    resid_type: str = "mlp_out",
):
    """
    Runs the model on a prompt while applying steering at a given layer.
    Returns logits.
    """

    tokens = model.to_tokens(prompt).to("cuda")
    mid_token_id = tokenizer.convert_tokens_to_ids("▁<MID>")
    hook_name = f"blocks.{layer}.hook_{resid_type}"

    steering_hook = make_steering_hook(
        steering_vector=steering_vector,
        alpha=alpha,
        mid_token_id=mid_token_id,
    )

    with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
        logits = model(tokens)

    return logits

# # Toy example to test the probe
# torch.manual_seed(0)

# # Create toy data
# N = 300
# D = 2
# C = 3

# X0 = torch.randn(N//3, D) + torch.tensor([2.0, 0.0])
# X1 = torch.randn(N//3, D) + torch.tensor([-2.0, 0.0])
# X2 = torch.randn(N//3, D) + torch.tensor([0.0, 2.0])

# X = torch.cat([X0, X1, X2], dim=0)
# y = torch.tensor([0]*(N//3) + [1]*(N//3) + [2]*(N//3))

# # Train probe
# probe = LinearProbe(d_model=D, num_classes=C)
# train_probe(probe, X, y, num_epochs=50, lr=1e-2)

# # Evaluate
# probe.eval()
# with torch.no_grad():
#     logits = probe(X)
#     preds = logits.argmax(dim=-1)
#     acc = (preds == y).float().mean()

# print("Toy accuracy:", acc.item())
# gives 0.89, is around the expected value 

########## very messy how to get the data
# should probably do this in other file idk
def_fim_dict = read_fim_dataset("training_data/def_FIM_data.txt")
call_fim_dict = read_fim_dataset("training_data/call_FIM_data.txt")

# get list of prompts to train on, and their corresponding identifier types (0: variable, 1: function, 2: class)
def_prompts, def_IDS = get_prompts_and_IDS(def_fim_dict)
call_prompts, call_IDS = get_prompts_and_IDS(call_fim_dict)

# prompts, ids = def_prompts[:100] + call_prompts[:100], def_IDS[:100] + call_IDS[:100]
prompts, ids = def_prompts + call_prompts, def_IDS + call_IDS

ids = torch.tensor(ids, dtype=torch.long)
device = "cuda"

res_activations = get_residual_activations(
    model,
    data=prompts,
    layer=30,
    resid_type="mlp_out"
)

# for the def and call subsets of the data
res_activations_def = get_residual_activations(
    model,
    data=def_prompts,
    layer=30,
    resid_type="mlp_out"
)
res_activations_def = res_activations_def.float().to(device)
def_IDS = torch.tensor(def_IDS, dtype=torch.long).to(device)

res_activations_call = get_residual_activations(
    model,
    data=call_prompts,
    layer=30,
    resid_type="mlp_out"
)

res_activations_call = res_activations_call.float().to(device)
call_IDS = torch.tensor(call_IDS, dtype=torch.long).to(device)



D = res_activations.shape[1]
C = 3

res_activations = res_activations.float().to(device)
ids = ids.to(device)

# Train probe
probe = LinearProbe(d_model=D, num_classes=C).to(device)
train_probe(probe, res_activations, ids, num_epochs=50, lr=1e-2)

# train probe on def data
probe_def = LinearProbe(d_model=D, num_classes=C).to(device)
train_probe(probe_def, res_activations_def, def_IDS, num_epochs=50, lr=1e-2)

# train probe on call data
probe_call = LinearProbe(d_model=D, num_classes=C).to(device)
train_probe(probe_call, res_activations_call, call_IDS, num_epochs=50, lr=1e-2)

# Evaluate; accuracy on the training set 
# Do we also want to evaluate on a test set?
probe.eval()
with torch.no_grad():
    logits = probe(res_activations)
    preds = logits.argmax(dim=-1)
    acc = (preds == ids).float().mean()
print("accuracy:", acc.item())


full_feature_direction_1 = get_class_steering_vector(probe, 0)
full_feature_direction_2 = get_class_steering_vector(probe, 1)
full_feature_direction_3 = get_class_steering_vector(probe, 2)
call_feature_direction_1 = get_class_steering_vector(probe_call, 0)
call_feature_direction_2 = get_class_steering_vector(probe_call, 1)
call_feature_direction_3 = get_class_steering_vector(probe_call, 2)
def_feature_direction_1 = get_class_steering_vector(probe_def, 0)
def_feature_direction_2 = get_class_steering_vector(probe_def, 1)
def_feature_direction_3 = get_class_steering_vector(probe_def, 2)


print("feature direction 1:", full_feature_direction_1)
print("feature direction 2:", full_feature_direction_2)
print("feature direction 3:", full_feature_direction_3)
print("Not normalised:", get_class_steering_vector(probe, 0, normalize=False))
print("Not normalised:", get_class_steering_vector(probe, 1, normalize=False))
print("Not normalised:", get_class_steering_vector(probe, 2, normalize=False))

print("feature direction 1 for def:", def_feature_direction_1)
print("feature direction 2 for def:", def_feature_direction_2)
print("feature direction 3 for def:", def_feature_direction_3)
print("feature direction 1 for call:", call_feature_direction_1)
print("feature direction 2 for call:", call_feature_direction_2)
print("feature direction 3 for call:", call_feature_direction_3)

# similarity feature direction 1 between all datasets
similarity_full_def = torch.cosine_similarity(full_feature_direction_1, def_feature_direction_1, dim=0)
similarity_def_call = torch.cosine_similarity(def_feature_direction_1, call_feature_direction_1, dim=0)
similarity_full_call = torch.cosine_similarity(full_feature_direction_1, call_feature_direction_1, dim=0)
print("Similarity between feature direction 1 for full and def:", similarity_full_def.item())
print("Similarity between feature direction 1 for def and call:", similarity_def_call.item())
print("Similarity between feature direction 1 for full and call:", similarity_full_call.item())

# same for feature direction 2
similarity_full_def = torch.cosine_similarity(full_feature_direction_2, def_feature_direction_2, dim=0)
similarity_def_call = torch.cosine_similarity(def_feature_direction_2, call_feature_direction_2, dim=0)
similarity_full_call = torch.cosine_similarity(full_feature_direction_2, call_feature_direction_2, dim=0)
print("Similarity between feature direction 2 for full and def:", similarity_full_def.item())
print("Similarity between feature direction 2 for def and call:", similarity_def_call.item())
print("Similarity between feature direction 2 for full and call:", similarity_full_call.item())

# same for feature direction 3
similarity_full_def = torch.cosine_similarity(full_feature_direction_3, def_feature_direction_3, dim=0)
similarity_def_call = torch.cosine_similarity(def_feature_direction_3, call_feature_direction_3, dim=0)
similarity_full_call = torch.cosine_similarity(full_feature_direction_3, call_feature_direction_3, dim=0)
print("Similarity between feature direction 3 for full and def:", similarity_full_def.item())
print("Similarity between feature direction 3 for def and call:", similarity_def_call.item())
print("Similarity between feature direction 3 for full and call:", similarity_full_call.item())


print("try steering")
# Example: steer toward CLASS identifiers (class_id = 0)
s_class = get_class_steering_vector(probe, class_id=0)

prompt = prompts[0]

# No steering
logits_base = model(prompt)

# With steering
logits_steered = run_with_steering(
    model=model,
    prompt=prompt,
    steering_vector=s_class,
    alpha=5.0,        # try 0.5 → 10.0
    layer=30,
    resid_type="mlp_out",
)

# Compare predictions / generations
print("Base next token:", model.to_string(logits_base.argmax(dim=-1)[0, -1]))
print("Steered next token:", model.to_string(logits_steered.argmax(dim=-1)[0, -1]))
