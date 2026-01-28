from functools import partial
from typing import List, Optional, Union

import einops
import numpy as np
import plotly.express as px
import plotly.io as pio
import torch
from circuitsvis.attention import attention_heads
from fancy_einsum import einsum
from IPython.display import HTML, IFrame
from jaxtyping import Float

import transformer_lens.utils as utils
from transformer_lens import ActivationCache, HookedTransformer


torch.set_grad_enabled(False)
print("Disabled automatic differentiation")

answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)
print("Answer residual directions shape:", answer_residual_directions.shape)
logit_diff_directions = (
    answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
)
print("Logit difference directions shape:", logit_diff_directions.shape)


# NBVAL_IGNORE_OUTPUT
model = HookedTransformer.from_pretrained(
    "codellama/CodeLlama-7b-hf",
#    center_unembed=True,
#    center_writing_weights=True,
#    fold_ln=True,
#    refactor_factored_attn_matrices=True,
)

# Get the default device used
device: torch.device = utils.get_device()





example_prompt = "<▁PRE>def pet(a,b):\n   return(a+b)\nx=3\ny=5\n#new variable z equal to 8\nz=<▁SUF>(x,y)<▁MID>"
example_answer = "pet"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)






