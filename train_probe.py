import torch
import transformer_lens
from transformers import AutoTokenizer
from steering import get_class_steering_vector

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

    prompts_def, labels_def = load_dataset(part="DEF")
    prompts_call, labels_call = load_dataset(part="CALL")

    extractor = ResidualActivationExtractor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=8,
    )

    n_layers = model.cfg.n_layers
    layer = 25 # the layer to probe, found using logitlens

    results_def = probe_layer(
        extractor=extractor,
        prompts=prompts_def,
        labels=labels_def,
        layer=layer,
    )
    probe_def = results_def["probe"]

    results_call = probe_layer(
        extractor=extractor,
        prompts=prompts_call,
        labels=labels_call,
        layer=layer,
    )
    probe_call = results_call["probe"]

    results_full = probe_layer(
        extractor=extractor,
        prompts=prompts,
        labels=labels,
        layer=layer,
    )
    probe_full = results_full["probe"]

    for i in range(3):
        full_feature_direction_i = get_class_steering_vector(probe_full, i)
        call_feature_direction_i = get_class_steering_vector(probe_call, i)
        def_feature_direction_i = get_class_steering_vector(probe_def, i)
        print(f"feature direction {i} norm full:", full_feature_direction_i)
        print(f"feature direction {i} norm call:", call_feature_direction_i)
        print(f"feature direction {i} norm def:", def_feature_direction_i)

        # similarity between the feature directions
        similarity_full_call = torch.cosine_similarity(full_feature_direction_i, call_feature_direction_i, dim=0)
        similarity_full_def = torch.cosine_similarity(full_feature_direction_i, def_feature_direction_i, dim=0)
        similarity_def_call = torch.cosine_similarity(def_feature_direction_i, call_feature_direction_i, dim=0)

        print(f"Similarity between feature direction {i} for full and call:", similarity_full_call.item())
        print(f"Similarity between feature direction {i} for full and def:", similarity_full_def.item())
        print(f"Similarity between feature direction {i} for def and call:", similarity_def_call.item())



    # COMMENTED THIS OUT FOR NOW JUST UNCOMMENT IF U WANT TO RUN IT AGAIN
    # results = probe_all_layers(
    #     extractor=extractor,
    #     prompts=prompts,
    #     labels=labels,
    #     n_layers=n_layers,
    # )

    # # print best layer
    # best_layer = max(results, key=lambda k: results[k]["test_acc"])
    # print("Best layer:", best_layer)
    # print("Test accuracy:", results[best_layer]["test_acc"])

    # print("All results:", results)


if __name__ == "__main__":
    main()
