
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import os
import wandb
from scipy.special import softmax
from torch.utils.data import TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true, y_pred):
    wa = balanced_accuracy_score(y_true, y_pred)
    ua = accuracy_score(y_true, y_pred)
    wf1 = f1_score(y_true, y_pred, average="weighted")
    uf1 = f1_score(y_true, y_pred, average="macro")
    return wa, ua, wf1, uf1

def plot_and_save_roc(labels, probs, num_classes, save_path):
    labels_bin = label_binarize(labels, classes=list(range(num_classes)))
    plt.figure(figsize=(8, 6))

    if num_classes == 4:
        class_names = ["Angry", "Happy", "Sad", "Neutral"]
    elif num_classes == 5:
        class_names = ["Angry", "Happy", "Sad", "Neutral", "Surprise"]
    elif num_classes == 7:
        class_names = ["Neutral", "Joy", "Anger", "Sadness", "Disgust", "Fear", "Surprise"]
    else:
        class_names = [f"Class {i}" for i in range(num_classes)]

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve per Class")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ROC curve saved to {save_path}")
from torch.utils.data import DataLoader

def train_and_evaluate(model, train_dataset, valid_dataset, test_dataset,
                       optimizer, scheduler, criterion_fn,
                       alpha, epochs=150, save_path="best_model.pt", seed=None,
                       batch_size=64):

    num_classes = getattr(model, "num_classes", None)
    if num_classes is None:
        all_labels = torch.cat([train_dataset.tensors[2],
                                valid_dataset.tensors[2],
                                test_dataset.tensors[2]])
        num_classes = len(torch.unique(all_labels))

    run_name = os.path.basename(save_path).replace(".pt", "")

    wandb.init(
        project="CMCL-EmotionRecognition",
        name=f"{run_name}_alpha{alpha}_seed{seed}" if seed is not None else f"{run_name}_alpha{alpha}",
        config={
            "epochs": epochs,
            "lr": optimizer.param_groups[0]['lr'],
            "weight_decay": optimizer.param_groups[0]['weight_decay'],
            "scheduler": "ReduceLROnPlateau",
            "alpha": alpha,
            "seed": seed,
            "batch_size": batch_size
        }
    )

    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    best_val_wa = 0

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        total_train_loss = 0

        for text_x, audio_x, y in train_loader:
            text_x, audio_x, y = text_x.to(device), audio_x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(text_x, audio_x, return_all=True)
            loss = criterion_fn(outputs, y, None)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * y.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        val_losses, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for text_x, audio_x, y in val_loader:
                text_x, audio_x, y = text_x.to(device), audio_x.to(device), y.to(device)
                val_outputs = model(text_x, audio_x, return_all=True)
                val_loss = criterion_fn(val_outputs, y, None)
                val_losses.append(val_loss.item() * y.size(0))
                val_preds.extend(torch.argmax(val_outputs["logits"], dim=1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        avg_val_loss = np.sum(val_losses) / len(val_loader.dataset)
        wa, ua, wf1, uf1 = compute_metrics(val_labels, val_preds)

        scheduler.step(avg_val_loss)

        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_WA": wa,
            "val_UA": ua,
            "val_WF1": wf1,
            "val_UF1": uf1,
            "epoch": epoch + 1
        })

        if wa > best_val_wa:
            best_val_wa = wa
            torch.save(model.state_dict(), save_path)
            print(f"\nSaved best model at epoch {epoch + 1} with WA = {wa:.4f}")
            wandb.run.summary["best_val_wa"] = wa

        print(
            f"[Epoch {epoch + 1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"WA: {wa:.4f}, UA: {ua:.4f}, WF1: {wf1:.4f}, UF1: {uf1:.4f}"
        )

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))

    model.eval()
    test_preds, test_labels, test_logits_list = [], [], []
    with torch.no_grad():
        for text_x, audio_x, y in test_loader:
            text_x, audio_x, y = text_x.to(device), audio_x.to(device), y.to(device)
            test_outputs = model(text_x, audio_x, return_all=True)
            test_logits = test_outputs["logits"].cpu().numpy()
            test_preds.extend(np.argmax(test_logits, axis=1))
            test_labels.extend(y.cpu().numpy())
            test_logits_list.append(test_logits)

    test_logits_all = np.vstack(test_logits_list)
    test_probs = softmax(test_logits_all, axis=1)

    test_wa, test_ua, test_wf1, test_uf1 = compute_metrics(test_labels, test_preds)

    wandb.log({
        "test_WA": test_wa,
        "test_UA": test_ua,
        "test_WF1": test_wf1,
        "test_UF1": test_uf1
    })

    wandb.run.summary.update({
        "final_test_WA": test_wa,
        "final_test_UA": test_ua,
        "final_test_WF1": test_wf1,
        "final_test_UF1": test_uf1
    })

    print("\nFinal Test Metrics:")
    print(f"WA: {test_wa:.4f}, UA: {test_ua:.4f}, WF1: {test_wf1:.4f}, UF1: {test_uf1:.4f}")

    roc_save_path = save_path.replace(".pt", "_roc_curve.png")
    plot_and_save_roc(np.array(test_labels), test_probs, num_classes, roc_save_path)

    wandb.finish()

    return {
        "test_WA": test_wa,
        "test_UA": test_ua,
        "test_WF1": test_wf1,
        "test_UF1": test_uf1
    }
import torch
import time
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count

def get_model_stats(model, sample_input, device="cuda"):
    """
    Calculate FLOPs, parameters, and inference time.
    """
    model.eval().to(device)

    # FLOPs and parameters
    flops = FlopCountAnalysis(model, sample_input).total()
    params = sum(p.numel() for p in model.parameters())

    # THOP alternative (sometimes needed)
    # flops, params = profile(model, inputs=(sample_input,), verbose=False)

    # Inference time (avg over 50 runs)
    n_runs = 50
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(*sample_input) if isinstance(sample_input, tuple) else model(sample_input)
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / n_runs

    return {
        "FLOPs": flops,
        "Parameters": params,
        "Inference time (s)": avg_time
    }
