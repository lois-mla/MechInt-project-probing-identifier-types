import torch
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer
import plotly.express as px
import pandas as pd


torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("codellama/CodeLlama-7b-hf")
device = utils.get_device()

# FIM example
example_prompt = "<PRE>def pet(a,b):\n   return(a+b)\nclass cat:\n  x=5\nx=3\ny=5\n#new variable z equal to 8\nz=<SUF>(x,y)<MID>"
correct_answer = "pet"
incorrect_answer = "cat"


# answer_tokens should be a tensor of shape [batch, 2]
correct_token = model.to_single_token(correct_answer)
incorrect_token = model.to_single_token(incorrect_answer)
answer_tokens = torch.tensor([[correct_token, incorrect_token]]).to(device)

# Logit Difference
# finds the vector in the residual stream that points from the wrong logit to the correct logit
answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)
logit_diff_direction = answer_residual_directions[0, 0] - answer_residual_directions[0, 1]

# run the model
logits, cache = model.run_with_cache(example_prompt)

# Perform the Logit Lens
# We extract the residual stream at the final token position across all layers
# Position -1 corresponds to the <MID> token
accum_resid, labels = cache.accumulated_resid(layer=-1, incl_mid=True, return_labels=True)
last_token_resid = accum_resid[:, 0, -1, :] # Shape: [n_layers + 1, d_model]

# Apply the final LayerNorm scaling
scaled_resid = cache.apply_ln_to_stack(last_token_resid, layer=-1)

# Project the residual stream onto the logit difference direction
# This gives us a "score" for how much each layer prefers 'pet' over 'x'
logit_lens_values = (scaled_resid @ logit_diff_direction).cpu().numpy()


df = pd.DataFrame({
    "Layer": labels,
    "Logit Difference": logit_lens_values
})

fig = px.line(df, x="Layer", y="Logit Difference", 
             title=f"Logit Lens: Preference for '{correct_answer}' vs '{incorrect_answer}'",
             template="plotly_white")
fig.update_traces(mode='lines+markers')
fig.show()