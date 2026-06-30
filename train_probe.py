"""
This file is used for training the linear probe and running the steering experiments.
It contains the following functions: 
for probing: 
probe_layer: Probes a single layer of the model.
probe_all_layers: Probes all layers of the model.

for steering: 
save_average_to_csv: Saves the average results for the steering experiments to a CSV file.
steer_prompts_from_file: Runs the steering experiment for prompts from a file and plots the results.
It also contains the code to run the probing and steering experiments at the bottom of the file.
For the different dataset, different paths should be specified (this is explained in the comments)
"""
import torch
import os
import transformer_lens
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


from utils import evaluate_first_token_accuracy_jsonl, evaluate_first_token_accuracy, randomize_model_weights, read_steering_dataset, read_fim_dataset, get_prompt, get_prompts_and_IDS, train_test_split, load_dataset, load_model, save_probe, load_probe
from steering import compare_steering_with_gap, compare_steering_with_gap_non_contr
from linearprobe_new import (
    ResidualActivationExtractor,
    LinearProbe,
    train_probe,
    evaluate_probe,
)
from plotting import plot_average_gap, plot_probe_accuracies

def probe_layer(
    extractor,
    prompts,
    labels,
    layer: int,
    resid_type: str,
    num_classes: int = 3,
    device: str = "cuda",
    save_dir: str = "probes_stored/probes_no_cont",
    
):
    print(f"\nProbing layer {layer}")

    save_path = os.path.join(save_dir, f"probe_layer_{layer}.pt")

    # Extract features (needed either way for eval)
    X = extractor.extract(prompts, layer=layer, resid_type=resid_type)
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


def save_accuracies_to_csv(
    results,
    probe_name,
    base_path = "accuracies"
):
    accuracies = {"train": [],
    		  "test": []}
    for layer in results:
        train = results[layer]["train_acc"]
        test = results[layer]["test_acc"]
        accuracies["train"].append(train)
        accuracies["test"].append(test)
   
    save_dir = os.path.join(base_path, f"accuracies_{probe_name}")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"accuracies_{probe_name}.csv")
    pd.DataFrame(accuracies).to_csv(save_path, index=False)
    print(f"Saved CSV: {save_path}")





def probe_all_layers(
    extractor,
    prompts,
    labels,
    n_layers: int,
    resid_type: str,
    save_dir: str = "probes",
):
    results = {}

    for layer in range(n_layers):
        result = probe_layer(
            extractor=extractor,
            prompts=prompts,
            labels=labels,
            layer=layer,
            resid_type=resid_type,
            save_dir=save_dir,
        )

        results[layer] = result

    return results


def save_average_to_csv(averaged_results, id, contrastive_id, alpha, base_path="figures"):
    key = (id, contrastive_id)
    if key not in averaged_results:
        return

    layers = sorted(
        averaged_results[key].keys(),
        key=lambda x: int("".join(filter(str.isdigit, x)))
    )

    rows = []
    for layer in layers:
        vals = averaged_results[key][layer]
        rows.append({
            "layer": layer,
            "prob_gap": vals["prob_gap"],
            "prob_contr": vals["prob_contr"],
            "prob_true": vals["prob_true"],
            "log_gap": vals["log_gap"],
            "log_contr": vals["log_contr"],
            "log_true": vals["log_true"],
        })

    save_dir = os.path.join(base_path, f"id_{id}_contr_id_{contrastive_id}")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"avg_gap_alpha_{alpha}.csv")
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"Saved CSV: {save_path}")


def steer_prompts_from_file_new(def_path: str, use_path: str, model, tokenizer, results, resid_type: str, dataset_specifier, dataset_specifier_fullname, part="FULL", alpha=50.0):

    # rly stupid way of doing this but ok
    if part == "FULL":
        data1 = read_fim_dataset(def_path)
        data2 = read_fim_dataset(use_path)
        data = data1 + data2
    elif part == "DEF":
        data = read_fim_dataset(def_path)
    elif part == "CALL":
        data = read_fim_dataset(use_path)
    ids = [0, 1, 2]
    
    averages = defaultdict(lambda: defaultdict(lambda: {
        "prob_gap": [],
        "prob_contr": [],
        "prob_true": [],
        "log_gap": [],
        "log_contr": [],
        "log_true": [],
    }))

    for id in ids:
        data_per_id = [d for d in data if int(d["identifier_type"]) == id]

        for prompt_dic in data_per_id:
            prompt_prefix = prompt_dic["prefix"]
            prompt_suffix = prompt_dic["suffix"]
            prompt = get_prompt(prompt_prefix, prompt_suffix)

            for contrastive_id in ids:
                if contrastive_id == id:
                    continue

                token = prompt_dic[str(id)]
                contrastive_token = prompt_dic[str(contrastive_id)]

                _, gap_differences = compare_steering_with_gap(
                    model=model,
                    tokenizer=tokenizer,
                    results=results,
                    prompt=prompt,
                    id=id,
                    contrastive_id=contrastive_id,
                    token=token,
                    contrastive_token=contrastive_token,
                    alpha=alpha,
                    resid_type=resid_type,
                    k=20,
                )

                key = (id, contrastive_id)

                for layer, vals in gap_differences.items():
                    for key_metric in vals:
                        averages[key][layer][key_metric].append(vals[key_metric])


    # Compute final averages 
    final_averages = {}

    for key, layer_dict in averages.items():
        final_averages[key] = {}

        for layer, vals in layer_dict.items():
            final_averages[key][layer] = {
                metric: np.mean(values)
                for metric, values in vals.items()
            }

    # Save plots + CSV
    for (id, contrastive_id) in final_averages.keys():
        plot_average_gap(
            final_averages,
            id,
            contrastive_id,
            alpha=alpha,
            use_logprob=True,
            base_path=f"figures/{dataset_specifier}_{alpha}",
            dataset_specifier_fullname=dataset_specifier_fullname
        )
        plot_average_gap(
            final_averages,
            id,
            contrastive_id,
            alpha=alpha,
            use_logprob=False,
            base_path=f"figures/{dataset_specifier}_{alpha}",
            dataset_specifier_fullname=dataset_specifier_fullname
        )
        save_average_to_csv(
            final_averages,
            id,
            contrastive_id,
            alpha=alpha,
            base_path=f"figures/{dataset_specifier}_{alpha}"
        )

    return final_averages



def steer_prompts_from_file_old(path: str, model, tokenizer, results, resid_type: str, dataset_specifier, dataset_specifier_fullname, alpha=50.0):
    data = read_steering_dataset(path)
    ids = [0, 1, 2]

    averages = defaultdict(lambda: defaultdict(lambda: {
        "prob_gap": [],
        "prob_contr": [],
        "prob_true": [],
        "log_gap": [],
        "log_contr": [],
        "log_true": [],
    }))

    for id in ids:
        data_per_id = [d for d in data if int(d["ID"]) == id]

        for prompt_dic in data_per_id:
            prompt_prefix = prompt_dic["prefix"]
            prompt_suffix = prompt_dic["suffix"]
            prompt = get_prompt(prompt_prefix, prompt_suffix)

            for contrastive_id in ids:
                if contrastive_id == id:
                    continue

                token = prompt_dic[str(id)]
                contrastive_token = prompt_dic[str(contrastive_id)]

                _, gap_differences = compare_steering_with_gap_non_contr(
                    model=model,
                    tokenizer=tokenizer,
                    results=results,
                    prompt=prompt,
                    id=id,
                    contrastive_id=contrastive_id,
                    token=token,
                    contrastive_token=contrastive_token,
                    alpha=alpha,
                    resid_type=resid_type, # HERE YOU CHOOSE THE LOCATION TO STEER IN
                    k=20,
                )

                key = (id, contrastive_id)

                for layer, vals in gap_differences.items():
                    for key_metric in vals:
                        averages[key][layer][key_metric].append(vals[key_metric])


    # Compute final averages 
    final_averages = {}

    for key, layer_dict in averages.items():
        final_averages[key] = {}

        for layer, vals in layer_dict.items():
            final_averages[key][layer] = {
                metric: np.mean(values)
                for metric, values in vals.items()
            }

    # Save plots + CSV
    for (id, contrastive_id) in final_averages.keys():
        plot_average_gap(
            final_averages,
            id,
            contrastive_id,
            alpha=alpha,
            use_logprob=True,
            base_path=f"figures/{dataset_specifier}_{alpha}",
            dataset_specifier_fullname=dataset_specifier_fullname
        )
        plot_average_gap(
            final_averages,
            id,
            contrastive_id,
            alpha=alpha,
            use_logprob=False,
            base_path=f"figures/{dataset_specifier}_{alpha}",
            dataset_specifier_fullname=dataset_specifier_fullname
        )
        save_average_to_csv(
            final_averages,
            id,
            contrastive_id,
            alpha=alpha,
            base_path=f"figures/{dataset_specifier}_{alpha}"
        )

    return final_averages


def main():

    # first three lines; contrastive letter-name dataset
    # next three lines; non-contrastive letter-name dataset
    # last three lines; contrastive realistic-name dataset
    # data_def = "training_data/def_FIM_data_final.txt"
    # data_call = "training_data/call_FIM_data_final.txt"
    # probe_save_dir = "probes_stored/probes_final" 
    data_def = "training_data/def_FIM_data_final.txt"
    data_call = "training_data/call_FIM_data_final.txt"
    # data_def = "training_data/def_FIM_data_nocont.txt"
    # data_call = "training_data/call_FIM_data_nocont.txt"
    # probe_save_dir = "probes_stored/probes_no_cont"
    # data_def = "training_data/def_FIM_data.txt"
    # data_call = "training_data/call_FIM_data.txt"
    # probe_save_dir = "probes_stored/probes_realistic"

    # data_def = "datasets/letters/mixed_definition.jsonl"
    # data_call = "datasets/letters/mixed_usage.jsonl"
    # probe_save_dir = "probes_stored/probes_new_data_set"

    # for the baseline we need a separate save directory
    #  probe_save_dir = "probes_stored/probes_final_baseline_fixed"

    # specify the name of the chosen dataset for saving the file and plot titles
    # dataset_specifier = "cont_baseline"
    # # dataset_specifier_fullname = "contrastive dataset baseline"
    # dataset_specifier = "letters_not_mixed"
    # dataset_specifier_fullname = "letters_not_mixed"
    # data_def = "training_data/def_FIM_data_final.txt"
    # data_call = "training_data/call_FIM_data_final.txt"
    # probe_save_dir = "probes_stored/probes_final" 

    # data_def = "training_data/def_FIM_data_nocont.txt"
    # data_call = "training_data/call_FIM_data_nocont.txt"
    # data_def = "training_data/def_FIM_data_final.txt"
    # data_call = "training_data/call_FIM_data_final.txt"
    #probe_save_dir = "probes_stored/probes_final" 
    #data_def = "training_data/def_FIM_data_nocont.txt"
    #data_call = "training_data/call_FIM_data_nocont.txt"
    # probe_save_dir = "probes_stored/probes_no_cont"
    # data_def = "training_data/def_FIM_data.txt"
    # data_call = "training_data/call_FIM_data.txt"ls
    # probe_save_dir = "probes_stored/probes_realistic"


    # new probe data and save dir for the contrastive dataset with only correct predicted examples
    # data_def = "training_data/def_FIM_data_final_only_correct.txt"
    # data_call = "training_data/call_FIM_data_final_only_correct.txt"
    # # probe_save_dir = "probes_stored/probes_final_only_correct"

    # identifier_mode = "letters"
    # # identifier_mode = "common"
    # # identifier_mode = "tokenizer"
    # # part = "FULL"

    # data_def = f"datasets/final/{identifier_mode}/mixed_definition_only_correct.jsonl"
    # data_call = f"datasets/final/{identifier_mode}/mixed_usage_only_correct.jsonl"

    # # for the baseline we need a separate save directory
    # # probe_save_dir = "probes_stored/probes_final_baseline"
    # # probe_save_dir = f"probes_stored/{identifier_mode}"

    # # specify the name of the chosen dataset for saving the file and plot titles
    dataset_specifier = "2_nocont_baseline_resid_post_w_initial_embed"
    dataset_specifier_fullname = "non contrastive dataset baseline residual post with initial embed"
    # # # dataset_specifier = "cont_only_correct"
    # # # dataset_specifier_fullname = "contrastive dataset only correct"
    # # # dataset_specifier = "cont_baseline"
    # # # dataset_specifier_fullname = "contrastive dataset baseline"
    # # dataset_specifier = f"{identifier_mode}"
    # # dataset_specifier_fullname = f"{identifier_mode}"

    # probe_save_dir = f"probes_stored/probes_{identifier_mode}_only_correct_baseline"
    # dataset_specifier = "letter_mixed_only_correct_baseline"
    # dataset_specifier_fullname = "letter mixed only correct baseline"

    model, tokenizer = load_model() # baseline vibessss
    model = randomize_model_weights(model, skip_embeddings=True) # use this line exacfor the baseline!!
    device = "cuda"
    n_layers = model.cfg.n_layers
    resid_type = "resid_post" # NOTE: HERE YOU CHOOSE THE LOCATION TO PROBE IN # want to try mlp.hook_post, did mlp_out
    identifier_mode = "old_data"

    # for resid_type in ["attn_out", "post"]:
    probe_save_dir = f"probes_stored/probes_final2_{resid_type}"

# for identifier_mode in ['letters', 'common', 'tokenizer']:
    # data_def = f"datasets/final/{identifier_mode}/mixed_definition.jsonl"
    # data_call = f"datasets/final/{identifier_mode}/mixed_usage.jsonl"

    # # for the baseline we need a separate save directory
    # # probe_save_dir = "probes_stored/probes_final_baseline"
    # probe_save_dir = f"probes_stored/{identifier_mode}_{resid_type}"
    # dataset_specifier = f"{identifier_mode}_{resid_type}"
    # dataset_specifier_fullname = f"{identifier_mode}_{resid_type}"

    dataset_specifier = f"{identifier_mode}_{resid_type}"
    dataset_specifier_fullname = f"{identifier_mode}_{resid_type}"


    prompts, labels = load_dataset(data_def, data_call)

    extractor = ResidualActivationExtractor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=8,
    )


    results = probe_all_layers(
        extractor=extractor,
        prompts=prompts,
        labels=labels,
        n_layers=n_layers,
        resid_type=resid_type,
        save_dir=probe_save_dir
        
    )

#     # print best layer
#     best_layer = max(results, key=lambda k: results[k]["test_acc"])
#     print("Best layer:", best_layer)
#     print("Test accuracy:", results[best_layer]["test_acc"])

#     # save accuracies
    save_accuracies_to_csv(results, dataset_specifier)

    alphas = [100.0]

    # plot the probe accuracies
    save_dir = "figures/probe_accuracy_old_data"
    filename = f"linear_probe_accuracy_per_layer_{dataset_specifier}_{resid_type}.png"
    plot_probe_accuracies(results, save_dir=save_dir, filename=filename, dataset_specifier=dataset_specifier_fullname)

    steering_path = "training_data/steering_data_300_final.txt"
    alpha = 100.0
    dataset_specifier = f"steering_away_{dataset_specifier}"
    steer_prompts_from_file_old(steering_path, model, tokenizer, results, resid_type, dataset_specifier, dataset_specifier_fullname, alpha=alpha)

    # # ---------- cross - steering -----------------------
    # steering_path_def = f"datasets/final/{identifier_mode}/steering_definition.jsonl"
    # steering_path_use = f"datasets/final/{identifier_mode}/steering_usage.jsonl"
    # steering_path = None

    # for mode in ["letters", "common", "tokenizer"]:
            
    #     steering_path_def = f"datasets/final/{mode}/steering_definition.jsonl"
    #     steering_path_use = f"datasets/final/{mode}/steering_usage.jsonl"

    #     dataset_specifier = f"{identifier_mode}_probe_{mode}_steering_{resid_type}"
    #     dataset_specifier_fullname = dataset_specifier

    #     alphas = [100.0]
    #     for alpha in alphas:
    #         if steering_path is None:
    #             steer_prompts_from_file_old(steering_path_def, steering_path_use, model, tokenizer, results, resid_type=resid_type, dataset_specifier=dataset_specifier, dataset_specifier_fullname=dataset_specifier_fullname, alpha=alpha)
    #         else:
    #             steer_prompts_from_file_old(steering_path, model, tokenizer, results, resid_type=resid_type, dataset_specifier=dataset_specifier, dataset_specifier_fullname=dataset_specifier_fullname, alpha=alpha)



    # model = randomize_model_weights(model) # use this line for the baseline only!!!!!!!

    # evaluate base accuracy for the mixed letters datafile and save the datasets with only correct examples
    # evaluate_first_token_accuracy_jsonl(model, tokenizer,"datasets/final/letters/mixed_definition.jsonl" , "datasets/final/letters/mixed_definition_only_correct.jsonl")
    # evaluate_first_token_accuracy_jsonl(model, tokenizer, "datasets/final/letters/mixed_usage.jsonl", "datasets/final/letters/mixed_usage_only_correct.jsonl")


    # #--- NEW: Evaluate Base Accuracy and save the datasets with only the correct examples---
    # print("--- Testing First Token Accuracy (Definitions) ---")
    # evaluate_first_token_accuracy(model, tokenizer, data_def, "training_data/def_FIM_data_final_only_correct.txt")

    # print("--- Testing First Token Accuracy (Calls) ---")
    # evaluate_first_token_accuracy(model, tokenizer, data_call, "training_data/call_FIM_data_final_only_correct.txt")

    # data_def = "training_data/def_FIM_data_nocont.txt"
    # data_call = "training_data/call_FIM_data_nocont.txt"

    # print("--- Testing First Token Accuracy (Definitions) ---")
    # evaluate_first_token_accuracy(model, tokenizer, data_def, "training_data/def_FIM_data_nocont_only_correct.txt")

    # print("--- Testing First Token Accuracy (Calls) ---")
    # evaluate_first_token_accuracy(model, tokenizer, data_call, "training_data/call_FIM_data_nocont_only_correct.txt")

    # data_def = "datasets/letters/mixed_definition_only_correct.jsonl"
    # data_call = "datasets/letters/mixed_call_only_correct.jsonl"
    # probe_save_dir = "probes_stored/probes_letter_mixed_only_correct"



    # # # model = randomize_model_weights(model) # use this line for the baseline!!
    prompts, labels = load_dataset(data_def, data_call)

    extractor = ResidualActivationExtractor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=8,
    )


    results = probe_all_layers(
        extractor=extractor,
        prompts=prompts,
        labels=labels,
        n_layers=n_layers,
        resid_type=resid_type,
        save_dir=probe_save_dir
    )


    # print best layer
    best_layer = max(results, key=lambda k: results[k]["test_acc"])
    print("Best layer:", best_layer)
    print("Test accuracy:", results[best_layer]["test_acc"])

    # save accuracies
    save_accuracies_to_csv(results, dataset_specifier)
    
    # plot the probe accuracies
    # save_dir = "figures/probe_accuracy"
    # filename = f"linear_probe_accuracy_per_layer_{dataset_specifier}.png"
    # plot_probe_accuracies(results, save_dir=save_dir, filename=filename, dataset_specifier=dataset_specifier_fullname)
    
    # # print("All results:", results)
    # steering_path = "training_data/steering_data_new.txt"
    # steer_prompts_from_file_new(steering_path, model, tokenizer, results)






    # steering:

#     prompt = get_prompt(prompt_prefix, prompt_suffix)

#     alpha = 10.0

#     df = compare_steering(
#     model=model,
#     tokenizer=tokenizer,
#     results=results,
#     prompt=prompt,
#     id=0,
#     contrastive_id=1,
#     alpha=alpha,
#     resid_type="mlp_out",
#     k=20,
# )

#     print(prompt)
#     print("alpha: ", alpha)
    # pd.set_option('display.max_columns', None)
#     print(df) 

    # # print best layer
    # best_layer = max(results, key=lambda k: results[k]["test_acc"])
    # print("Best layer:", best_layer)
    # print("Test accuracy:", results[best_layer]["test_acc"])

    # print("All results:", results)

    # use the first path for the realistic name dataset and the second path for both letter-name datasets
    # steering_path = "training_data/steering_data_300_realistic.txt"
    # steering_path = "training_data/steering_data_300_final.txt"

    # IF USING NEW DATA SET STEERING_PATH TO NONE!!!!!!!!!!!!!!!!!!
    # steering_path = None
    # steering_path_def = f"datasets/final/{identifier_mode}/steering_definition.jsonl"
    # steering_path_use = f"datasets/final/{identifier_mode}/steering_usage.jsonl"


    # for mode in ["letters", "common", "tokenizer"]:
            
    #     steering_path_def = f"datasets/final/{mode}/steering_definition.jsonl"
    #     steering_path_use = f"datasets/final/{mode}/steering_usage.jsonl"

    #     dataset_specifier = f"{identifier_mode}_probe_{mode}_steering"
    #     dataset_specifier_fullname = dataset_specifier

    #     alphas = [100.0]
    #     for alpha in alphas:
    #         if steering_path is not None:
    #             steer_prompts_from_file_old(steering_path, model, tokenizer, results, resid_type, dataset_specifier, dataset_specifier_fullname, alpha=alpha)
    #         else:
    #             steer_prompts_from_file_new(steering_path_def, steering_path_use, model, tokenizer, results, resid_type, dataset_specifier, dataset_specifier_fullname, alpha=alpha)

if __name__ == "__main__":
    main()

    
