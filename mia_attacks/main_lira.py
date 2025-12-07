# src/mia_lira.py

import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from .lira.shadow import *
from .lira.lira import *
from .lira.model import *  # only needed if you still want CNN here for shadow models etc.


def _collect_all_data(loader: DataLoader):
    """Turn a DataLoader into (images, labels) tensors."""
    imgs, labs = [], []
    for x, y in loader:
        imgs.append(x)
        labs.append(y)
    return torch.cat(imgs, dim=0), torch.cat(labs, dim=0)

def run_lira_attack(
    target_model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str | torch.device,
    perc: float = 0.0,
    perc_test: float = 0.20,
    measurement_number: int = 10,
    num_shadow_models: int = 50,
    lr_shadow_model: float = 1e-3,
    epochs_shadow_model: int = 20,
    results_dir: str = "results_lira",
    save_artifacts: bool = True,
    shadow_model_fn=None, 
):
    """
    Run a LIRA membership inference attack on a given `target_model`.

    Parameters
    ----------
    target_model : nn.Module
        The trained model you want to attack.
    train_loader : DataLoader
        Loader for the (global or victim) training data.
    test_loader : DataLoader
        Loader for the held-out test data (non-members).
    device : str or torch.device
        "cuda" or "cpu".
    perc : float
        Fraction of the training dataset that the attacker knows (shadow training side).
    perc_test : float
        Fraction of the test dataset that the attacker knows as "shadow non-members".
    measurement_number : int
        Number of samples from train and test used for measurement each.
    num_shadow_models : int
        Number of shadow models used by LIRA.
    lr_shadow_model : float
        Learning rate for shadow model training.
    epochs_shadow_model : int
        Number of epochs for each shadow model.
    results_dir : str
        Directory to save ROC plots and hyperparameter logs (if save_artifacts=True).
    save_artifacts : bool
        If True, save ROC figure and a small txt log of hyperparameters.

    Returns
    -------
    result : dict
        {
            "auc": float,
            "tpr": np.ndarray,
            "fpr": np.ndarray,
            "scores": np.ndarray,
            "measurement_ref": np.ndarray,
            "train_accuracy": float,
            "test_accuracy": float,
        }
    """

    if isinstance(device, str):
        device = torch.device(device)

    target_model.to(device)

    # ---- 1) Accuracy sanity check on the target model ----
    train_acc = calculate_accuracy(target_model, train_loader, device)
    test_acc = calculate_accuracy(target_model, test_loader, device)
    print(f"[LIRA] Target training accuracy: {train_acc:.2f}%")
    print(f"[LIRA] Target test accuracy:     {test_acc:.2f}%")

    # ---- 2) Collect all train/test data from loaders ----
    train_images, train_labels = _collect_all_data(train_loader)
    test_images, test_labels = _collect_all_data(test_loader)

    n_train = len(train_images)
    n_test = len(test_images)

    # Num samples attacker has access to
    num_samples_train = int(perc * n_train)
    num_samples_test = int(perc_test * n_test)

    print("----------------------------------")
    print("[LIRA] Attacker's knowledge:")
    print(
        f"[LIRA] Training Dataset Info: {num_samples_train}/{n_train} = "
        f"{(num_samples_train / max(1, n_train)) * 100:.2f}%"
    )
    print(
        f"[LIRA] Testing Dataset Info:  {num_samples_test}/{n_test} = "
        f"{(num_samples_test / max(1, n_test)) * 100:.2f}%"
    )
    print("----------------------------------")

    # ---- 3) Build attacker / measurement splits (reproducible via saved indices) ----
    # Training side
    idx_path = "original_indices_lira_train"

    if not os.path.exists(idx_path):
        # First-time run → create new permutation
        original_indices = torch.randperm(n_train)
        torch.save(original_indices, idx_path)
    else:
        # Load existing permutation
        original_indices = torch.load(idx_path)

        # If dataset size changed OR indices are out of range → regenerate
        if len(original_indices) != n_train or original_indices.max().item() >= n_train:
            original_indices = torch.randperm(n_train)
            torch.save(original_indices, idx_path)

    # samples the attacker knows of training data
    attacker_train_indices = original_indices[:num_samples_train]
    # Samples used for measuring attack success
    meas_train_indices = original_indices[num_samples_train : num_samples_train + measurement_number]

    attacker_train_images = train_images[attacker_train_indices]
    attacker_train_labels = train_labels[attacker_train_indices]
    measurement_train_images = train_images[meas_train_indices]
    measurement_train_labels = train_labels[meas_train_indices]

    # Test side
    idx_path_test = "original_indices_lira_test"

    if not os.path.exists(idx_path_test):
        original_indices_test = torch.randperm(n_test)
        torch.save(original_indices_test, idx_path_test)
    else:
        original_indices_test = torch.load(idx_path_test)

        if len(original_indices_test) != n_test or original_indices_test.max().item() >= n_test:
            original_indices_test = torch.randperm(n_test)
            torch.save(original_indices_test, idx_path_test)

    # samples attacker knows not part of data
    attacker_test_indices = original_indices_test[:num_samples_test]
    # Samples used for measuring attack success
    meas_test_indices = original_indices_test[num_samples_test : num_samples_test + measurement_number]

    attacker_test_images = test_images[attacker_test_indices]
    attacker_test_labels = test_labels[attacker_test_indices]
    measurement_test_images = test_images[meas_test_indices]
    measurement_test_labels = test_labels[meas_test_indices]

    # Shadow training data: attacker-known train + attacker-known test
    shadow_images = torch.cat([attacker_train_images, attacker_test_images], dim=0)
    shadow_labels = torch.cat([attacker_train_labels, attacker_test_labels], dim=0)

    # Measurement set: used to evaluate attack
    measurement_images = torch.cat([measurement_train_images, measurement_test_images], dim=0)
    measurement_labels = torch.cat([measurement_train_labels, measurement_test_labels], dim=0)
    measurement_ref = np.array(
        [0] * len(measurement_train_images) + [1] * len(measurement_test_images)
    )  # 0 = member, 1 = non-member

    print(f"[LIRA] Measurement sample size: {len(measurement_images)}")

    # ---- 4) Shadow zone: estimate in/out loss distributions via shadow models ----
    in_mean, in_std, out_mean, out_std = estimate_loss_distributions(
        measurement_images,
        measurement_labels,
        shadow_images,
        shadow_labels,
        num_shadow_models=num_shadow_models,
        epochs=epochs_shadow_model,
        lr=lr_shadow_model,
        shadow_model_fn=shadow_model_fn
    )

    # ---- 5) Attack zone: run LIRA scoring on the target model ----
    scores = run_over_MIA(
        target_model,
        measurement_images,
        measurement_labels,
        in_mean,
        in_std,
        out_mean,
        out_std,
    )

    # ---- 6) Compute ROC / AUC ----
    fpr, tpr, _ = roc_curve(measurement_ref, scores)
    auc_value = auc(fpr, tpr)
    print("[LIRA] --------------")
    print(f"[LIRA] AUC: {auc_value}  |")
    print("[LIRA] --------------")

    # ---- 7) Optional: save artifacts ----
    if save_artifacts:
        os.makedirs(results_dir, exist_ok=True)

        # ROC plot
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {auc_value:0.2f})")
        plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic - LIRA")
        plt.legend(loc="lower right")
        roc_path = os.path.join(results_dir, "ROC_LIRA_Attack.png")
        plt.savefig(roc_path)
        plt.close()

        # Hyperparameters log
        hyperparams = {
            "perc": perc,
            "perc_test": perc_test,
            "measurement_number": measurement_number,
            "num_shadow_models": num_shadow_models,
            "lr_shadow_model": lr_shadow_model,
            "epochs_shadow_model": epochs_shadow_model,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "device": str(device),
        }
        txt_path = os.path.join(results_dir, "LIRA_hyperparams.txt")
        with open(txt_path, "w") as f:
            for k, v in hyperparams.items():
                f.write(f"{k}: {v}\n")

    return {
        "auc": auc_value,
        "tpr": tpr,
        "fpr": fpr,
        "scores": scores,
        "measurement_ref": measurement_ref,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }
