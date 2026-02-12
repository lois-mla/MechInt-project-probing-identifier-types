import torch
import transformer_lens
from transformers import AutoTokenizer
# import matplotlib.pyplot as plt

from utils import read_fim_dataset, get_prompt, get_prompts_and_IDS, train_test_split, load_dataset, load_model
from steering import compare_steering, get_class_steering_vector
from linearprobe_new import (
    ResidualActivationExtractor,
    LinearProbe,
    train_probe,
    evaluate_probe,
)
# from steering import 

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

    model, tokenizer = load_model()
    prompts, labels = load_dataset()
    device = "cuda"

    extractor = ResidualActivationExtractor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=8,
    )
    

#     # probe all layers
#     for layer in range (32):
#         results_full = probe_layer(
#             extractor=extractor,
#             prompts=prompts,
#             labels=labels,
#             layer=layer,
#         )
#         probe_full = results_full["probe"]
#         # print the train and test accuracy of this layer
#         print(f"Layer {layer} | Train acc: {results_full['train_acc']:.4f} | Test acc: {results_full['test_acc']:.4f}")

#         compare_steering(
#             model=model,
#             tokenizer=tokenizer,
#             probe=probe_full,        # trained on layer 25
#             prompt=prompt,
#             id=1,                    # positive class
#             contrastive_id=2,        # negative class
#             alpha=50.0,            # 'how much' you steer
#             layer=layer,
#             resid_type="mlp_out",    # must match extractor
#         )

#             # clean probe tensors
#         del probe_full
#         torch.cuda.empty_cache()





    # device = "cuda"

    # model, tokenizer = load_model()
    # prompts, labels = load_dataset()

    # extractor = ResidualActivationExtractor(
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=device,
    #     batch_size=8,
    # )

    n_layers = model.cfg.n_layers

    # results = probe_all_layers(
    #     extractor=extractor,
    #     prompts=prompts,
    #     labels=labels,
    #     n_layers=n_layers,
    # )
    # probe_full = results_full["probe"]

    # for i in range(3):
    #     full_feature_direction_i = get_class_steering_vector(probe_full, i)
    #     call_feature_direction_i = get_class_steering_vector(probe_call, i)
    #     def_feature_direction_i = get_class_steering_vector(probe_def, i)
    #     print(f"feature direction {i} norm full:", full_feature_direction_i)
    #     print(f"feature direction {i} norm call:", call_feature_direction_i)
    #     print(f"feature direction {i} norm def:", def_feature_direction_i)

    #     # similarity between the feature directions
    #     similarity_full_call = torch.cosine_similarity(full_feature_direction_i, call_feature_direction_i, dim=0)
    #     similarity_full_def = torch.cosine_similarity(full_feature_direction_i, def_feature_direction_i, dim=0)
    #     similarity_def_call = torch.cosine_similarity(def_feature_direction_i, call_feature_direction_i, dim=0)

    #     print(f"Similarity between feature direction {i} for full and call:", similarity_full_call.item())
    #     print(f"Similarity between feature direction {i} for full and def:", similarity_full_def.item())
    #     print(f"Similarity between feature direction {i} for def and call:", similarity_def_call.item())


    # COMMENTED THIS OUT FOR NOW JUST UNCOMMENT IF U WANT TO RUN IT AGAIN
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

    layer=0
    for layer, result in results.items():
        probe = result["probe"]

        compare_steering(
            model=model,
            tokenizer=tokenizer,
            probe=probe,        #
            prompt=prompt,
            id=0,                    # positive class
            contrastive_id=1,        # negative class
            alpha=10.0,
            layer=layer,                
            resid_type="mlp_out",    # must match extractor
        )

        layer += 1
        del probe
        torch.cuda.empty_cache()


    # # print best layer
    # best_layer = max(results, key=lambda k: results[k]["test_acc"])
    # print("Best layer:", best_layer)
    # print("Test accuracy:", results[best_layer]["test_acc"])

    # print("All results:", results)

if __name__ == "__main__":
    main()

    