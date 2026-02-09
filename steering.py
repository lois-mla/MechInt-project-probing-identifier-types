import torch
import transformer_lens


def get_class_steering_vector(
    probe,
    class_id: int,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Returns steering vector for a class.
    """
    W = probe.linear.weight.detach()   # [C, d_model]
    s = W[class_id].clone()

    if normalize:
        s = s / (s.norm() + 1e-8)

    return s

    # steering vector
    # steering_vector = get_class_steering_vector(
    #     probe,
    #     class_id=class_id,
    #     normalize=True,
    # )
def get_contrastive_steering_vector(
    probe,
    pos_class: int,
    neg_class: int,
    normalize: bool = True,
) -> torch.Tensor:
    """
    pos - neg steering direction.
    """
    W = probe.linear.weight.detach()
    s = W[pos_class] - W[neg_class]

    if normalize:
        s = s / (s.norm() + 1e-8)

    return s

def make_last_token_steering_hook(
    steering_vector: torch.Tensor,
    alpha: float,
):
    steering_vector = steering_vector.to("cuda")

    def hook_fn(acts: torch.Tensor, hook):
        # acts: [batch, seq_len, d_model]
        acts[:, -1] += alpha * steering_vector
        return acts

    return hook_fn

# def make_steering_hook(
#     steering_vector: torch.Tensor,
#     alpha: float,
#     mid_token_id: int,
# ):
#     steering_vector = steering_vector.to("cuda")

#     def hook_fn(acts: torch.Tensor, hook):
#         tokens = hook.ctx["tokens"]   # [batch, seq_len]

#         for b in range(tokens.size(0)):
#             mid_pos = (tokens[b] == mid_token_id).nonzero(as_tuple=True)[0]
#             if len(mid_pos) != 1:
#                 continue
#             pos = mid_pos.item()
#             acts[b, pos] += alpha * steering_vector

#         return acts

#     return hook_fn

@torch.inference_mode()
def run_with_last_token_steering(
    model,
    tokenizer,
    prompt: str,
    steering_vector: torch.Tensor,
    alpha: float,
    layer: int,
    resid_type: str = "mlp_out",
):
    tokens = model.to_tokens(prompt).to("cuda")

    hook_name = f"blocks.{layer}.hook_{resid_type}"

    hook_fn = make_last_token_steering_hook(
        steering_vector=steering_vector,
        alpha=alpha,
    )

    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        logits = model(tokens)

    return logits



@torch.inference_mode()
def compare_steering(
    model: transformer_lens.HookedTransformer,
    tokenizer,
    probe,
    prompt: str,
    id: int,
    contrastive_id: int,
    alpha: float = 5.0,
    layer: int = 24,
    resid_type: str = "mlp_out",
):
    model = model.to("cuda")

    # tokens
    tokens = model.to_tokens(prompt).to("cuda")

    steering_vec = get_contrastive_steering_vector(
        probe,
        pos_class=id,
        neg_class=contrastive_id,
    )

    logits_base = model(tokens)
    logits_steered = run_with_last_token_steering(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        steering_vector=steering_vec,
        alpha=alpha,
        layer=layer,
        resid_type="mlp_out",

    )

    token_id = logits_base[0, -1].argmax().item()

    print("Token id:     ", token_id)
    print("Token string: ", tokenizer.convert_ids_to_tokens(token_id))
    print("Decoded repr: ", repr(tokenizer.decode([token_id], skip_special_tokens=False)))


    # print("Baseline next token:", tokenizer.decode(logits_base[0, -1].argmax()))
    # print("Steered  next token:", tokenizer.decode(logits_steered[0, -1].argmax()))


# @torch.inference_mode()
# def run_with_steering(
#     model: transformer_lens.HookedTransformer,
#     tokenizer,
#     prompt: str,
#     steering_vector: torch.Tensor,
#     alpha: float,
#     layer: int,
#     resid_type: str = "mlp_out",
# ):
#     tokens = model.to_tokens(prompt).to("cuda")
#     mid_token_id = tokenizer.convert_tokens_to_ids("▁<MID>")
#     hook_name = f"blocks.{layer}.hook_{resid_type}"

#     steering_hook = make_steering_hook(
#         steering_vector=steering_vector,
#         alpha=alpha,
#         mid_token_id=mid_token_id,
#     )

#     with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
#         logits = model(tokens)

#     return logits


def decode_output(model, logits, tokens):
    # take argmax for simplicity
    next_tokens = logits.argmax(dim=-1)
    return model.to_string(next_tokens)


# @torch.inference_mode()
# def compare_steering(
#     model: transformer_lens.HookedTransformer,
#     tokenizer,
#     probe,
#     prompt: str,
#     id: int,
#     contrastive_id: int,
#     alpha: float = 5.0,
#     layer: int = 24,
#     resid_type: str = "mlp_out",
# ):
#     model = model.to("cuda")

#     # tokens
#     tokens = model.to_tokens(prompt).to("cuda")

#     # steering vector
#     # steering_vector = get_class_steering_vector(
#     #     probe,
#     #     class_id=class_id,
#     #     normalize=True,
#     # )
#     steering_vector = get_contrastive_steering_vector(
#     probe,
#     pos_class=id,
#     neg_class=contrastive_id,
#     normalize=True,
# )

#     # normal
#     logits_base = model(tokens)
#     text_base = decode_output(model, logits_base, tokens)

#     # steered
#     logits_steered = run_with_steering(
#         model=model,
#         tokenizer=tokenizer,
#         prompt=prompt,
#         steering_vector=steering_vector,
#         alpha=alpha,
#         layer=layer,
#         resid_type=resid_type,
#     )
#     text_steered = decode_output(model, logits_steered, tokens)

#     print("=== PROMPT ===")
#     print(prompt)
#     print("\n=== BASELINE ===")
#     print(text_base)
#     print("\n=== STEERED ===")
#     print(text_steered)
    
# def decode_output(model, logits, tokens):
#     # take argmax for simplicity
#     next_tokens = logits.argmax(dim=-1)

#     return model.to_string(next_tokens)


# @torch.inference_mode()
# def compare_steering(
#     model: transformer_lens.HookedTransformer,
#     tokenizer,
#     probe,
#     prompt: str,
#     id: int,
#     contrastive_id: int,
#     alpha: float = 5.0,
#     layer: int = 24,
#     resid_type: str = "mlp_out",
# ):
#     model = model.to("cuda")

#     # tokens
#     tokens = model.to_tokens(prompt).to("cuda")

#     # steering vector
#     # steering_vector = get_class_steering_vector(
#     #     probe,
#     #     class_id=class_id,
#     #     normalize=True,
#     # )
#     steering_vector = get_contrastive_steering_vector(
#     probe,
#     pos_class=id,
#     neg_class=contrastive_id,
#     normalize=True,
# )

#     # ---- baseline ----
#     logits_base = model(tokens)
#     text_base = decode_output(model, logits_base, tokens)

#     # ---- steered ----
#     logits_steered = run_with_steering(
#         model=model,
#         tokenizer=tokenizer,
#         prompt=prompt,
#         steering_vector=steering_vector,
#         alpha=alpha,
#         layer=layer,
#         resid_type=resid_type,
#     )
#     text_steered = decode_output(model, logits_steered, tokens)

#     print("=== PROMPT ===")
#     print(prompt)
#     print("\n=== BASELINE ===")
#     print(text_base)
#     print("\n=== STEERED ===")
#     print(text_steered)

