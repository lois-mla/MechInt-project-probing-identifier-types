import torch
import torch.nn as nn
import transformer_lens
from transformers import AutoTokenizer
from typing import List


class ResidualActivationExtractor:
    """
    Extracts residual stream activations at the <MID> token
    from a specified layer of a HookedTransformer.
    """

    def __init__(
        self,
        model: transformer_lens.HookedTransformer,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        batch_size: int = 8,
        max_length: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.mid_token_id = tokenizer.convert_tokens_to_ids("▁<MID>")


    @torch.inference_mode()
    def extract(
        self,
        prompts: List[str],
        layer: int,
        resid_type: str = "resid_mid",
    ) -> torch.Tensor:
        """
        Returns: [N, d_model] tensor of activations at <MID>.
        """
        all_acts = []
        hook_name = f"blocks.{layer}.hook_{resid_type}"

        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]

            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = enc["input_ids"].to(self.device)

            _, cache = self.model.run_with_cache(
                tokens,
                return_type=None,
                stop_at_layer=layer + 1,
                names_filter=[hook_name],
            )

            acts = cache[hook_name]     # [B, seq, d_model]
            mid_acts = []

            for b in range(tokens.size(0)):
                mid_pos = (tokens[b] == self.mid_token_id).nonzero(as_tuple=True)[0]
                assert len(mid_pos) == 1, "Each prompt must contain exactly one <MID> token."
                pos = mid_pos.item()
                mid_acts.append(acts[b, pos])

            mid_acts = torch.stack(mid_acts)
            all_acts.append(mid_acts.cpu())

        return torch.cat(all_acts, dim=0)


class LinearProbe(nn.Module):
    """
    Simple linear classifier on top of residual activations.
    """

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_probe(
    probe: LinearProbe,
    X: torch.Tensor,
    y: torch.Tensor,
    num_epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 32,
):
    probe.train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(num_epochs):
        perm = torch.randperm(X.size(0), device=X.device)
        X_shuf = X[perm]
        y_shuf = y[perm]

        for i in range(0, X.size(0), batch_size):
            xb = X_shuf[i:i+batch_size]
            yb = y_shuf[i:i+batch_size]

            logits = probe(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.inference_mode()
def evaluate_probe(
    probe: LinearProbe,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 64,
) -> float:
    probe.eval()
    correct = 0
    total = 0

    for i in range(0, X.size(0), batch_size):
        xb = X[i:i+batch_size]
        yb = y[i:i+batch_size]

        logits = probe(xb)
        preds = logits.argmax(dim=-1)

        correct += (preds == yb).sum().item()
        total += yb.size(0)

    return correct / total
