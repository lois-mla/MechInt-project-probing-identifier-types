import torch
import transformer_lens
from transformers import AutoTokenizer

from utils import read_fim_dataset, get_prompts_and_IDS, train_test_split, load_dataset, load_model
from linearprobe_new import (
    ResidualActivationExtractor,
    LinearProbe,
    train_probe,
    evaluate_probe,
)


def probe_layer(
    extractor,
    prompts,
    labels,
    layer: int,
    num_classes: int = 3,
    device: str = "cuda",
):
    print(f"Probing layer {layer}")

    # extract features
    X = extractor.extract(prompts, layer=layer)
    X = X.float().to(device)
    y = labels.to(device)

    # split data
    X_train, y_train, X_test, y_test = train_test_split(
        X, y, test_frac=0.2, seed=42
    )

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # train
    D = X.shape[1]
    probe = LinearProbe(d_model=D, num_classes=num_classes).to(device)
    train_probe(probe, X_train, y_train, num_epochs=30, lr=1e-3)

    # evaluate
    train_acc = evaluate_probe(probe, X_train, y_train)
    test_acc = evaluate_probe(probe, X_test, y_test)

    print(f"Train acc: {train_acc:.4f}")
    print(f"Test  acc: {test_acc:.4f}")

    return {
        "probe": probe,
        "train_acc": train_acc,
        "test_acc": test_acc,
    }


def probe_all_layers(
    extractor,
    prompts,
    labels,
    n_layers: int,
):
    results = {}

    for layer in range(n_layers):
        result = probe_layer(
            extractor=extractor,
            prompts=prompts,
            labels=labels,
            layer=layer,
        )
        results[layer] = result

    return results

def main():
    device = "cuda"

    model, tokenizer = load_model()
    prompts, labels = load_dataset()

    extractor = ResidualActivationExtractor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=8,
    )

    n_layers = model.cfg.n_layers

    results = probe_all_layers(
        extractor=extractor,
        prompts=prompts,
        labels=labels,
        n_layers=n_layers,
    )

    # print best layer
    best_layer = max(results, key=lambda k: results[k]["test_acc"])
    print("Best layer:", best_layer)
    print("Test accuracy:", results[best_layer]["test_acc"])

    print("All results:", results)


if __name__ == "__main__":
    main()
