# mia_attacks.py

import os
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# from .baseline.shadow import *
# from .baseline.model import *  # only needed if you still want CNN etc., but not strictly required here


def _select_attack_impl(choice: str):
    """Dynamically import the proper baseline attack implementation."""
    if choice == 'loss':
        from .baseline.baseline_loss import MIA, calculate_accuracy
        attack_epochs = 30
        attack_lr = 1e-2
        attack_hidden_size = 8
    elif choice == 'conf':
        from .baseline.baseline_conf import MIA, calculate_accuracy
        attack_epochs = 50
        attack_lr = 1e-2
        attack_hidden_size = 8
    elif choice == 'prob':
        from .baseline.baseline_prob import MIA, calculate_accuracy
        attack_epochs = 30
        attack_lr = 1e-3
        attack_hidden_size = 128
    else:
        raise ValueError(f"Invalid MIA choice: {choice}")

    return MIA, calculate_accuracy, attack_epochs, attack_lr, attack_hidden_size


def _collect_all_data_from_loader(loader):
    """Turn a DataLoader into two tensors (images, labels)."""
    images_list = []
    labels_list = []

    for images, labels in loader:
        images_list.append(images)
        labels_list.append(labels)

    images = torch.cat(images_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return images, labels


def run_mia_attack(
    target_model,
    train_loader,
    test_loader,
    device: str | torch.device,
    choice: str = "loss",
    measurement_number: int = 10,
    results_dir: str = "results",
    save_artifacts: bool = True,
):
    """
    Run a membership inference attack against `target_model`.

    Parameters
    ----------
    target_model : torch.nn.Module
        The trained model you want to attack.
    train_loader : DataLoader
        DataLoader for (training) members.
    test_loader : DataLoader
        DataLoader for non-members.
    device : str or torch.device
        Device on which the model lives ("cuda" or "cpu").
    choice : {"loss", "conf", "prob"}
        Which baseline attack variant to use.
    measurement_number : int
        Number of target samples from each of train and test to measure.
    results_dir : str
        Where to save ROC plots and hyperparameters (if save_artifacts is True).
    save_artifacts : bool
        If True, saves ROC figure and hyperparameters text file.

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

    # 1) Choose attack implementation + hyperparams
    MIA, calculate_accuracy, attack_epochs, attack_lr, attack_hidden_size = _select_attack_impl(choice)

    # 2) Collect full train/test tensors from loaders
    train_images, train_labels = _collect_all_data_from_loader(train_loader)
    test_images, test_labels = _collect_all_data_from_loader(test_loader)

    # 3) Accuracy sanity check (optional)
    train_acc = calculate_accuracy(target_model, train_loader, device)
    test_acc = calculate_accuracy(target_model, test_loader, device)
    print(f"[MIA] Target training accuracy: {train_acc:.2f}%")
    print(f"[MIA] Target test accuracy:     {test_acc:.2f}%")

    # 4) Build measurement sets 
    #    number of samples attacker already knows, keep 0 for now
    num_samples_train = int(0.0 * len(train_images))  # 0 
    num_samples_test = int(0.0 * len(test_images)) # 0

    # Training side (save to disk for reproducibility, but regenerate if size changed)
    idx_path_train = "results_mia/indices/original_indices"
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(idx_path_train), exist_ok=True)

    if not os.path.exists(idx_path_train):
        original_indices = torch.randperm(len(train_images))
        torch.save(original_indices, idx_path_train)
    else:
        original_indices = torch.load(idx_path_train)
        # If dataset size changed or indices are invalid, regenerate
        if len(original_indices) != len(train_images) or original_indices.max().item() >= len(train_images):
            original_indices = torch.randperm(len(train_images))
            torch.save(original_indices, idx_path_train)

    indices_train = original_indices[:num_samples_train] # Empty because num_samples_train = 0
    # Since num_samples_train = 0 is effectively take first n samples of training data.
    anti_indices_train = original_indices[num_samples_train:num_samples_train + measurement_number]

    attacker_train_images = train_images[indices_train] # Empty because 0
    attacker_train_labels = train_labels[indices_train] # Empty because 0
    
    # Samples where MIA attack will be performed on.
    measurement_train_images = train_images[anti_indices_train]
    measurement_train_labels = train_labels[anti_indices_train]

    idx_path_test = "results_mia/indices/original_indices_test"

    if not os.path.exists(idx_path_test):
        original_indices_test = torch.randperm(len(test_images))
        torch.save(original_indices_test, idx_path_test)
    else:
        original_indices_test = torch.load(idx_path_test)
        if len(original_indices_test) != len(test_images) or original_indices_test.max().item() >= len(test_images):
            original_indices_test = torch.randperm(len(test_images))
            torch.save(original_indices_test, idx_path_test)

    indices_test = original_indices_test[:num_samples_test]
    anti_indices_test = original_indices_test[num_samples_test:num_samples_test + measurement_number]

    attacker_test_images = test_images[indices_test]
    attacker_test_labels = test_labels[indices_test]
    measurement_test_images = test_images[anti_indices_test]
    measurement_test_labels = test_labels[anti_indices_test]

    # Combine
    # Both empty and not used but would be used to train a shadow model
    shadow_images = torch.cat([attacker_train_images, attacker_test_images])
    shadow_labels = torch.cat([attacker_train_labels, attacker_test_labels])

    # Combine train and test to get a mix of members and non members
    measurement_images = torch.cat([measurement_train_images, measurement_test_images])
    measurement_labels = torch.cat([measurement_train_labels, measurement_test_labels])
    
    # Array to track membership (1 = member, 0 = non-member)
    measurement_ref = np.array([1] * len(measurement_train_images) +
                               [0] * len(measurement_test_images))

    print(f"[MIA] Measurement sample size: {len(measurement_images)}")

    # 5) Run the actual attack
    scores = MIA(
        target_model,
        train_loader,
        test_loader,
        measurement_images,
        measurement_labels,
        attack_hidden_size,
        attack_epochs,
        attack_lr,
        device,
    )
    scores_members = scores[measurement_ref == 0]      # by your current convention
    scores_nonmembers = scores[measurement_ref == 1]

    print("mean score for members:    ", scores_members.mean())
    print("mean score for nonmembers: ", scores_nonmembers.mean())


    # 6) ROC & AUC
    fpr, tpr, _ = roc_curve(measurement_ref, scores, pos_label=1)
    auc_value = auc(fpr, tpr)
    print("[MIA] -------------------------")
    print(f"[MIA] AUC: {auc_value}")
    print("[MIA] -------------------------")

    # 7) Optionally save artifacts
    if save_artifacts:
        os.makedirs(results_dir, exist_ok=True)

        # ROC plot
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {auc_value:0.2f})')
        plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - Attack: {choice}')
        plt.legend(loc="lower right")
        roc_path = os.path.join(results_dir, f"ROC_Baseline_Attack_{choice}.png")
        plt.savefig(roc_path)
        plt.close()

        # Hyperparams log
        hyperparams = {
            "measurement_number": measurement_number,
            "attack_epochs": attack_epochs,
            "attack_lr": attack_lr,
            "attack_hidden_size": attack_hidden_size,
            "device": str(device),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        }
        txt_path = os.path.join(results_dir, f"Baseline_Attack_{choice}.txt")
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
