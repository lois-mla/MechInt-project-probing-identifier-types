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
    print(neg_class, pos_class, type(neg_class), type(pos_class))
    W = probe.linear.weight.detach()
    print(W, type(W))
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
    Metrics for the last-token distribution.
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

def compare_steering_with_gap(
    model,
    tokenizer,
    results,
    prompt: str,
    id: int,
    contrastive_id: int,
    token: str,
    contrastive_token: str,
    alpha: float = 10.0,
    resid_type: str = "mlp_out",
    k: int = 10,
):
    device = "cuda"
    model = model.to(device)
    tokens = model.to_tokens(prompt).to(device)

    token_id = get_token_id(tokenizer, token)
    contrastive_token_id = get_token_id(tokenizer, contrastive_token)

    table = {}
    gap_differences = {}

    # ----- Baseline -----
    with torch.no_grad():
        logits_base = model(tokens)

    table["baseline"] = get_topk_dict(logits_base, tokenizer, k)

    base_token = get_token_metrics(logits_base, token_id)
    base_contr = get_token_metrics(logits_base, contrastive_token_id)

    # ----- Steered layers -----
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

        layer_name = f"layer_{layer}"
        table[layer_name] = get_topk_dict(logits_steered, tokenizer, k)

        steered_token = get_token_metrics(logits_steered, token_id)
        steered_contr = get_token_metrics(logits_steered, contrastive_token_id)

        # ----- PROB TERMS -----
        prob_contr_shift = steered_contr["prob"] - base_contr["prob"]
        prob_true_shift = -steered_token["prob"] + base_token["prob"]
        prob_gap_shift = prob_contr_shift + prob_true_shift

        # ----- LOGPROB TERMS -----
        log_contr_shift = steered_contr["log_prob"] - base_contr["log_prob"]
        log_true_shift = -steered_token["log_prob"] + base_token["log_prob"]
        log_gap_shift = log_contr_shift + log_true_shift

        gap_differences[layer_name] = {
            "prob_gap": prob_gap_shift,
            "prob_contr": prob_contr_shift,
            "prob_true": prob_true_shift,
            "log_gap": log_gap_shift,
            "log_contr": log_contr_shift,
            "log_true": log_true_shift,
        }

        del steering_vec
        torch.cuda.empty_cache()

    df = pd.DataFrame.from_dict(table, orient="index")

    return df, gap_differences

# def compare_steering_research(
#     model,
#     tokenizer,
#     results,
#     prompt: str,
#     id: int,
#     contrastive_id: int,
#     token: str,
#     contrastive_token: str,
#     alpha: float = 10.0,
#     resid_type: str = "mlp_out",
#     k: int = 10,
# ):
#     device = "cuda"
#     model = model.to(device)
#     tokens = model.to_tokens(prompt).to(device)

#     token_id = get_token_id(tokenizer, token)
#     contrastive_token_id = get_token_id(tokenizer, contrastive_token)

#     table = {}
#     metrics = {
#         "target": {},
#         "contrastive": {},
#     }

#     # Helper functions

#     def store_results(name, logits):
#         table[name] = get_topk_dict(logits, tokenizer, k)
#         metrics["target"][name] = get_token_metrics(logits, token_id)
#         metrics["contrastive"][name] = get_token_metrics(logits, contrastive_token_id)

#     def add_relative_metrics(df):
#         base_log_prob = df.loc["baseline", "log_prob"]
#         base_prob = df.loc["baseline", "prob"]

#         df["delta_log_prob"] = df["log_prob"] - base_log_prob
#         df["prob_ratio"] = df["prob"] / base_prob
#         return df

#     # Baseline

#     with torch.no_grad():
#         logits_base = model(tokens)

#     store_results("baseline", logits_base)

#     # Steered layers 

#     for layer, result in results.items():
#         probe = result["probe"]

#         steering_vec = get_contrastive_steering_vector(
#             probe,
#             pos_class=id,
#             neg_class=contrastive_id,
#         )

#         with torch.no_grad():
#             logits_steered = run_with_last_token_steering(
#                 model=model,
#                 tokenizer=tokenizer,
#                 prompt=prompt,
#                 steering_vector=steering_vec,
#                 alpha=alpha,
#                 layer=layer,
#                 resid_type=resid_type,
#             )

#         store_results(f"layer_{layer}", logits_steered)

#         del steering_vec
#         torch.cuda.empty_cache()

#     # build DataFrames 

#     df = pd.DataFrame.from_dict(table, orient="index")

#     metrics_df = pd.DataFrame.from_dict(metrics["target"], orient="index")
#     contrastive_metrics_df = pd.DataFrame.from_dict(
#         metrics["contrastive"], orient="index"
#     )

#     metrics_df = add_relative_metrics(metrics_df)
#     contrastive_metrics_df = add_relative_metrics(contrastive_metrics_df)

#     return df, metrics_df, contrastive_metrics_df
# def compare_steering_research(
#     model,
#     tokenizer,
#     results,
#     prompt: str,
#     id: int,
#     contrastive_id: int,
#     token: str,
#     contrastive_token: str,
#     alpha: float = 10.0,
#     resid_type: str = "mlp_out",
#     k: int = 10,
# ):
#     device = "cuda"
#     model = model.to(device)
#     tokens = model.to_tokens(prompt).to(device)

#     token_id = get_token_id(tokenizer, token)
#     contrastive_token_id = get_token_id(tokenizer, contrastive_token)

#     table = {}
#     metrics_id = {}
#     metrics_contrastive_id = {}

#     # baseline 
#     with torch.no_grad():
#         logits_base = model(tokens)

#     table["baseline"] = get_topk_dict(logits_base, tokenizer, k)

#     base_metrics = get_token_metrics(logits_base, token_id)
#     base_metrics_contr = get_token_metrics(logits_base, contrastive_token_id)
    
#     metrics_id["baseline"] = base_metrics
#     metrics_contrastive_id["baseline"] = base_metrics_contr

#     # steered layers
#     for layer, result in results.items():
#         probe = result["probe"]

#         steering_vec = get_contrastive_steering_vector(
#             probe,
#             pos_class=id,
#             neg_class=contrastive_id,
#         )

#         with torch.no_grad():
#             logits_steered = run_with_last_token_steering(
#                 model=model,
#                 tokenizer=tokenizer,
#                 prompt=prompt,
#                 steering_vector=steering_vec,
#                 alpha=alpha,
#                 layer=layer,
#                 resid_type=resid_type,
#             )

#         table[f"layer_{layer}"] = get_topk_dict(logits_steered, tokenizer, k)
        
#         metrics_id[f"layer_{layer}"] = get_token_metrics(
#             logits_steered, token_id
#         )
#         metrics_contrastive_id[f"layer_{layer}"] = get_token_metrics(
#             logits_steered, contrastive_token_id
#         )

#         del steering_vec
#         torch.cuda.empty_cache()

#     df = pd.DataFrame.from_dict(table, orient="index")

#     contrastive_metrics_df = pd.DataFrame.from_dict(metrics_contrastive_id, orient="index")
#     metrics_df = pd.DataFrame.from_dict(metrics_id, orient="index")

#     base_log_prob = metrics_df.loc["baseline", "log_prob"]
#     base_prob = metrics_df.loc["baseline", "prob"]

#     metrics_df["delta_log_prob"] = metrics_df["log_prob"] - base_log_prob
#     metrics_df["prob_ratio"] = metrics_df["prob"] / base_prob

#     base_log_prob = contrastive_metrics_df.loc["baseline", "log_prob"]
#     base_prob = contrastive_metrics_df.loc["baseline", "prob"]

#     contrastive_metrics_df["delta_log_prob"] = contrastive_metrics_df["log_prob"] - base_log_prob
#     contrastive_metrics_df["prob_ratio"] = contrastive_metrics_df["prob"] / base_prob

#     return df, metrics_df, contrastive_metrics_df


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

