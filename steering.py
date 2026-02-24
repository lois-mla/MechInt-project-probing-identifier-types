import torch
import transformer_lens
import pandas as pd
import torch.nn.functional as F


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
    neg - pos steering direction.
    """
    W = probe.linear.weight.detach()
    s = W[neg_class] - W[pos_class]

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

#     steering_vec = get_contrastive_steering_vector(
#         probe,
#         pos_class=id,
#         neg_class=contrastive_id,
#     )

#     logits_base = model(tokens)
#     logits_steered = run_with_last_token_steering(
#         model=model,
#         tokenizer=tokenizer,
#         prompt=prompt,
#         steering_vector=steering_vec,
#         alpha=alpha,
#         layer=layer,
#         resid_type="mlp_out",

#     )

#     token_id = logits_base[0, -1].argmax().item()

#     print("=== BASELINE ===")
#     show_topk(logits_base, tokenizer, k=50)

#     print("\n=== STEERED ===")
#     show_topk(logits_steered, tokenizer, k=50)
#     # print("Token id:     ", token_id)
#     # print("Token string: ", tokenizer.convert_ids_to_tokens(token_id))
#     # print("Decoded repr: ", repr(tokenizer.decode([token_id], skip_special_tokens=False)))


#     # print("Baseline next token:", tokenizer.decode(logits_base[0, -1].argmax()))
#     # print("Steered  next token:", tokenizer.decode(logits_steered[0, -1].argmax()))

# def show_topk(logits, tokenizer, k=10):
#     vals, ids = torch.topk(logits[0, -1], k)
#     for v, i in zip(vals, ids):
#         tok_id = i.item()
#         tok = tokenizer.convert_ids_to_tokens(tok_id)
#         dec = tokenizer.decode([tok_id], skip_special_tokens=False)
#         print(f"{tok!r:12s} | {dec!r:12s} | logit={v.item():.2f}")



def get_token_id(tokenizer, target_token: str):
    ids = tokenizer.encode(target_token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"Target '{target_token}' is {len(ids)} tokens. "
            "This function expects a single token."
        )
    return ids[0]


def get_token_metrics(logits, target_id: int):
    """
    Returns research-grade metrics for the last-token distribution.
    """
    last_logits = logits[0, -1]
    probs = F.softmax(last_logits, dim=-1)
    log_probs = F.log_softmax(last_logits, dim=-1)

    prob = probs[target_id].item()
    log_prob = log_probs[target_id].item()

    # rank (1 = best)
    rank = (last_logits > last_logits[target_id]).sum().item() + 1

    return {
        "prob": prob,
        "log_prob": log_prob,
        "rank": rank,
    }



def get_topk_dict(logits, tokenizer, k=10):
    vals, ids = torch.topk(logits[0, -1], k)
    row = {}
    for rank, (v, i) in enumerate(zip(vals, ids), start=1):
        tok_id = i.item()
        dec = tokenizer.decode([tok_id], skip_special_tokens=False)
        row[f"top_{rank}"] = f"{dec} ({v.item():.2f})"
    return row


def compare_steering(
    model,
    tokenizer,
    results,          # dictionary from probe_all_layers
    prompt: str,
    id: int,
    contrastive_id: int,
    alpha: float = 10.0,
    resid_type: str = "mlp_out",
    k: int = 10,
):
    """"
    Returns df with baseline & each layer steered
    """
    device = "cuda"
    model = model.to(device)
    tokens = model.to_tokens(prompt).to(device)

    table = {}

    # baseline logits
    with torch.no_grad():
        logits_base = model(tokens)

    table["baseline"] = get_topk_dict(logits_base, tokenizer, k)

    # steer layers & get logits
    for layer, result in results.items():

        probe = result["probe"]

        # get contr steering vector for each layer
        steering_vec = get_contrastive_steering_vector(
            probe,
            pos_class=id,
            neg_class=contrastive_id,
        )

        with torch.no_grad():
            logits_steered = run_with_last_token_steering(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                steering_vector=steering_vec,
                alpha=alpha,
                layer=layer,
                resid_type=resid_type,
            )

        table[f"layer_{layer}"] = get_topk_dict(logits_steered, tokenizer, k)

        del steering_vec
        torch.cuda.empty_cache()

    df = pd.DataFrame.from_dict(table, orient="index")
    return df


def compare_steering_research(
    model,
    tokenizer,
    results,
    prompt: str,
    id: int,
    contrastive_id: int,
    target_token: str,
    alpha: float = 10.0,
    resid_type: str = "mlp_out",
    k: int = 10,
):
    device = "cuda"
    model = model.to(device)
    tokens = model.to_tokens(prompt).to(device)

    target_id = get_token_id(tokenizer, target_token)

    table = {}
    metrics = {}

    # ===== baseline =====
    with torch.no_grad():
        logits_base = model(tokens)

    table["baseline"] = get_topk_dict(logits_base, tokenizer, k)
    base_metrics = get_token_metrics(logits_base, target_id)
    metrics["baseline"] = base_metrics

    # ===== steered layers =====
    for layer, result in results.items():
        probe = result["probe"]

        steering_vec = get_contrastive_steering_vector(
            probe,
            pos_class=id,
            neg_class=contrastive_id,
        )

        with torch.no_grad():
            logits_steered = run_with_last_token_steering(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                steering_vector=steering_vec,
                alpha=alpha,
                layer=layer,
                resid_type=resid_type,
            )

        table[f"layer_{layer}"] = get_topk_dict(logits_steered, tokenizer, k)
        metrics[f"layer_{layer}"] = get_token_metrics(
            logits_steered, target_id
        )

        del steering_vec
        torch.cuda.empty_cache()

    df = pd.DataFrame.from_dict(table, orient="index")
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")

    # ===== derived research metrics =====
    base_log_prob = metrics_df.loc["baseline", "log_prob"]
    base_prob = metrics_df.loc["baseline", "prob"]

    metrics_df["delta_log_prob"] = metrics_df["log_prob"] - base_log_prob
    metrics_df["prob_ratio"] = metrics_df["prob"] / base_prob

    return df, metrics_df


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

