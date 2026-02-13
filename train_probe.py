import torch
import os
import transformer_lens
from transformers import AutoTokenizer
# import matplotlib.pyplot as plt

from utils import read_fim_dataset, get_prompt, get_prompts_and_IDS, train_test_split, load_dataset, load_model, save_probe, load_probe
from steering import compare_steering, get_class_steering_vector
from linearprobe_new import (
    ResidualActivationExtractor,
    LinearProbe,
    train_probe,
    evaluate_probe,
)

# def probe_layer(
#     extractor,
#     prompts,
#     labels,
#     layer: int,
#     num_classes: int = 3,
#     device: str = "cuda",
# ):
#     print(f"Probing layer {layer}")

#     # extract features
#     X = extractor.extract(prompts, layer=layer)
#     X = X.float().to(device)
#     y = labels.to(device)

#     # split data
#     X_train, y_train, X_test, y_test = train_test_split(
#         X, y, test_frac=0.2, seed=42
#     )

#     print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

#     # train
#     D = X.shape[1]
#     probe = LinearProbe(d_model=D, num_classes=num_classes).to(device)
#     train_probe(probe, X_train, y_train, num_epochs=30, lr=1e-3)

#     # evaluate
#     train_acc = evaluate_probe(probe, X_train, y_train)
#     test_acc = evaluate_probe(probe, X_test, y_test)

#     print(f"Train acc: {train_acc:.4f}")
#     print(f"Test  acc: {test_acc:.4f}")

#     return {
#         "probe": probe,
#         "train_acc": train_acc,
#         "test_acc": test_acc,
#     }

def probe_layer(
    extractor,
    prompts,
    labels,
    layer: int,
    num_classes: int = 3,
    device: str = "cuda",
    save_dir: str = "probes",
):
    print(f"\nProbing layer {layer}")

    save_path = os.path.join(save_dir, f"probe_layer_{layer}.pt")

    # Extract features (needed either way for eval)
    X = extractor.extract(prompts, layer=layer)
    X = X.float().to(device)
    y = labels.to(device)

    X_train, y_train, X_test, y_test = train_test_split(
        X, y, test_frac=0.2, seed=42
    )

    D = X.shape[1]

    if os.path.exists(save_path):
        print("Loading existing probe...")
        probe = load_probe(save_path, device=device)

    else:
        print("Training new probe...")
        probe = LinearProbe(d_model=D, num_classes=num_classes).to(device)

        train_probe(
            probe,
            X_train,
            y_train,
            num_epochs=30,
            lr=1e-3,
        )

        save_probe(
            probe,
            save_path,
            d_model=D,
            num_classes=num_classes,
        )

    # Evaluate (always useful)
    train_acc = evaluate_probe(probe, X_train, y_train)
    test_acc = evaluate_probe(probe, X_test, y_test)

    print(f"Train acc: {train_acc:.4f}")
    print(f"Test  acc: {test_acc:.4f}")

    return {
        "probe": probe,
        "train_acc": train_acc,
        "test_acc": test_acc,
    }


# def probe_all_layers(
#     extractor,
#     prompts,
#     labels,
#     n_layers: int,
# ):
#     results = {}

#     for layer in range(n_layers):
#         result = probe_layer(
#             extractor=extractor,
#             prompts=prompts,
#             labels=labels,
#             layer=layer,
#         )
#         results[layer] = result

#     return results

def probe_all_layers(
    extractor,
    prompts,
    labels,
    n_layers: int,
    save_dir: str = "probes",
):
    results = {}

    for layer in range(n_layers):
        result = probe_layer(
            extractor=extractor,
            prompts=prompts,
            labels=labels,
            layer=layer,
            save_dir=save_dir,
        )

        results[layer] = result

    return results


def main():

    #     ["""def a(x, y):
    #     return x * y

    # class q:
    #     b = 4
    #     def m(self, z):
    #         return z + 1

    # c = [1, 2]

    # o = q()
    # p = o.b
    # r = a(p, p)
    # s = FIM    
    #     """, """class FIM:
    #     k = 7

    # d = 3.5

    # def n(u, v):
    #     return u - v

    # o = FIM()
    # p = o.k
    # r = n(p, d)
    # s = d * 2
    #     """, """x = {1, 2, 3}

    # def FIM(a, b, c):
    #     a = a + b
    #     return a

    # class l:
    #     y = 5

    # o = l()
    # p = o.y
    # r = FIM(p, p, p)
    # s = x
    # """, """class t:
    #     a = 2
    #     def f(self, q):
    #         return q * 3

    # def w(e):
    #     return e + 1

    # FIM = {"x": 9}

    # o = t()
    # p = o.a
    # r = w(p)
    # s = FIM
    # """, """def k(a, b):
    #     return a + b

    # class s:
    #     m = 8

    # v = 10

    # o = FIM()
    # p = o.m
    # r = k(p, v)
    # u = v + 1
    # ""","""class j:
    #     r = 6

    # def c(x):
    #     return x * 2

    # FIM = [4, 5]

    # o = j()
    # p = o.r
    # q = c(p)
    # z = FIM
    # """]    


        # put new prompt here
        # answer should be m
    #     prompt_prefix = """class x:
    #     y = 'base'
    #     def __init__(self, z):
    #         self.z = z
    #     def a(self, b):
    #         return self.y + self.z + b

    # def c(d, e):
    #     return d - e

    # f = 100

    # g = c(f, 10)
    # h = 

    # """
        
    #     prompt_suffix = """('mid')
    # i = h.a('end')
    # j = f // 2
    #     """

    prompt_prefix = """class n:
    o = 3.14
    def p(self):
        return self.o


"""
    
    prompt_suffix = """ = {'z': 99}

def q(r):
    return r.capitalize()

s = q('word')
t = n()
u = t.p()
v = w['z']
    """

#     prompt = get_prompt(prompt_prefix, prompt_suffix)

#     print(prompt)
    data_def = "training_data/def_FIM_data_final.txt"
    data_call = "training_data/call_FIM_data_final.txt"

    model, tokenizer = load_model()
    prompts, labels = load_dataset(data_def, data_call)
    device = "cuda"

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

    # steering:

    prompt = get_prompt(prompt_prefix, prompt_suffix)

    alpha = 10.0

    df = compare_steering(
    model=model,
    tokenizer=tokenizer,
    results=results,
    prompt=prompt,
    id=0,
    contrastive_id=1,
    alpha=alpha,
    resid_type="mlp_out",
    k=20,
)

    print(prompt)
    print("alpha: ", alpha)
    print(df.to_markdown()) 


    # # print best layer
    # best_layer = max(results, key=lambda k: results[k]["test_acc"])
    # print("Best layer:", best_layer)
    # print("Test accuracy:", results[best_layer]["test_acc"])

    # print("All results:", results)

if __name__ == "__main__":
    main()

    