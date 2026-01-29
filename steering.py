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


def make_steering_hook(
    steering_vector: torch.Tensor,
    alpha: float,
    mid_token_id: int,
):
    steering_vector = steering_vector.to("cuda")

    def hook_fn(acts: torch.Tensor, hook):
        tokens = hook.ctx["tokens"]   # [batch, seq_len]

        for b in range(tokens.size(0)):
            mid_pos = (tokens[b] == mid_token_id).nonzero(as_tuple=True)[0]
            if len(mid_pos) != 1:
                continue
            pos = mid_pos.item()
            acts[b, pos] += alpha * steering_vector

        return acts

    return hook_fn


@torch.inference_mode()
def run_with_steering(
    model: transformer_lens.HookedTransformer,
    tokenizer,
    prompt: str,
    steering_vector: torch.Tensor,
    alpha: float,
    layer: int,
    resid_type: str = "mlp_out",
):
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

# if __name__ == "__main__":
