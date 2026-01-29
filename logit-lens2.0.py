import torch
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer
import plotly.express as px
import pandas as pd


torch.cuda.empty_cache()



torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("codellama/CodeLlama-7b-hf")
device = utils.get_device()

# FIM example
'''
example_prompt = "<PRE>def pet(a,b):\n   return(a+b)\nclass cat:\n  x=5\nx=3\ny=5\n#new variable z equal to 8\nz=<SUF>(x,y)<MID>"
correct_answer = "pet"
incorrect_answer = "cat" 
'''
'''
example_prompt = "<PRE>def square(x):\n   return x * x\nclass Counter:\ndef __init__(self, start=0):\n        self.value = start\n    def inc(self):\n      self.value += 1\n        return self.value\n<SUF> = True\nsq = square(2)\nctr = Counter(0)\nafter = ctr.inc()\ntotal = ctr.value\nneg = not flag\nboth = flag and False<MID>"

correct_answer = "flag"
incorrect_answer = "square"
'''

example_prompt = "class ing:\n    def __init__(self, name):\n        self.name = name\n    def greet(self):\n        return f'Hi {self.name}'\nn = 10\nres = add(1, 7)\ng = FIM('name3')\nmsg = g.greet()\nn = g.name\nm = n + 3\ncheck = n > 5\ndef add(a, b):\n    return a + b"

correct_answer = "ing"
incorrect_answer = "n"


# answer_tokens should be a tensor of shape [batch, 2]
correct_token = model.to_single_token(correct_answer)
incorrect_token = model.to_single_token(incorrect_answer)
answer_tokens = torch.tensor([[correct_token, incorrect_token]]).to(device)

# Check whether model predicts correctly
utils.test_prompt(example_prompt, correct_answer, model, prepend_bos=True)



# Logit Difference
# finds the vector in the residual stream that points from the wrong logit to the correct logit
answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)
logit_diff_direction = answer_residual_directions[0, 0] - answer_residual_directions[0, 1]

# run the model
# logits, cache = model.run_with_cache(example_prompt)

# Create a list to store only the residual stream values we need
accum_resid = []

# Define a hook function that only grabs the last token's residual stream
def hook_fn(residual_value, hook):
    # residual_value has shape [batch, pos, d_model]
    # We only want the last token [-1]
    accum_resid.append(residual_value[:, -1, :].detach().clone())

# Identify the names of the residual stream hooks
resid_post_hooks = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
# Also add the very beginning (embedding)
hooks = [("blocks.0.hook_resid_pre", hook_fn)] + [(name, hook_fn) for name in resid_post_hooks]

# Run the model with these hooks
with model.hooks(fwd_hooks=hooks):
    logits = model(example_prompt)

# Stack the results into a single tensor [n_layers + 1, d_model]
last_token_resid = torch.stack(accum_resid).squeeze(1)



# Perform the Logit Lens
# We extract the residual stream at the final token position across all layers
# Position -1 corresponds to the <MID> token
# accum_resid, labels = cache.accumulated_resid(layer=-1, incl_mid=True, return_labels=True)
# last_token_resid = accum_resid[:, 0, -1, :] # Shape: [n_layers + 1, d_model]

# Apply the final LayerNorm scaling
scaled_resid = model.ln_final(last_token_resid)

# Project the residual stream onto the logit difference direction
# This gives us a "score" for how much each layer prefers 'pet' over 'x'
logit_lens_values = (scaled_resid @ logit_diff_direction).cpu().numpy()


labels = ["Embed"] + [f"L{i}" for i in range(model.cfg.n_layers)]

full_logits = scaled_resid @ model.W_U # [33, vocab_size]

layer_stats = []
for i, label in enumerate(labels):
    # Get logits for this specific layer
    layer_logits = full_logits[i]
    
    # Calculate Rank of the correct token
    # We sort the logits and see where our 'correct_token' sits
    sorted_logits = torch.argsort(layer_logits, descending=True)
    rank = (sorted_logits == correct_token).nonzero().item() + 1
    
    # Get the top 1 token string for context at this layer
    top_token_id = sorted_logits[0].item()
    top_token_str = model.to_string(top_token_id)
    
    layer_stats.append({
        "Layer": label,
        "Logit Diff": logit_lens_values[i],
        "Rank": rank,
        "Top Prediction": top_token_str
    })

# Convert to DataFrame for a clear view
stats_df = pd.DataFrame(layer_stats)
print(stats_df)

# Plotting the evolution of Rank (Log scale is often better for Rank)
fig = px.line(stats_df, x="Layer", y="Logit Diff", title="Logit Difference Across Layers")
fig.show()

fig_rank = px.line(stats_df, x="Layer", y="Rank", title="Rank of Correct Token Across Layers")
fig_rank.update_yaxes(type="log", autorange="reversed") # Ranks are better viewed on log scale
fig_rank.show()


'''
df = pd.DataFrame({
    "Layer": labels,
    "Logit Difference": logit_lens_values
})

fig = px.line(df, x="Layer", y="Logit Difference", 
             title=f"Logit Lens: Preference for '{correct_answer}' vs '{incorrect_answer}'",
             template="plotly_white")
fig.update_traces(mode='lines+markers')
fig.show()
'''
