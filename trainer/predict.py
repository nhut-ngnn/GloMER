import os
import sys
import time
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    accuracy_score,
    f1_score
)
from sklearn.manifold import TSNE
from scipy.special import softmax
from thop import profile

import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.utils import set_seed
from src.architecture.GloMER import GloMER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data[0], tuple):
        raise ValueError(f"Loaded data is raw. Please provide feature-extracted .pkl: {pkl_path}")
    text = torch.stack([torch.tensor(item['text_embed']) for item in data])
    audio = torch.stack([torch.tensor(item['audio_embed']) for item in data])
    labels = torch.tensor([item['label'] for item in data])
    return Data(text_x=text, audio_x=audio, y=labels)


def compute_metrics(y_true, y_pred):
    wa = balanced_accuracy_score(y_true, y_pred)
    ua = accuracy_score(y_true, y_pred)
    wf1 = f1_score(y_true, y_pred, average="weighted")
    uf1 = f1_score(y_true, y_pred, average="macro")
    return wa, ua, wf1, uf1

def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_percent, annot=True, fmt=".2f", cmap="Blues", cbar=True,
        xticklabels=label_names, yticklabels=label_names,
        annot_kws={"size": 14, "weight": "bold"} 
    )
    plt.xlabel("Predicted Label", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=14, fontweight="bold")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", dpi=500, format="pdf")
    plt.close()
    print(f"Saved confusion matrix to {save_path}.pdf")



def plot_tsne(features, labels, label_names, save_path):
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(7, 6))
    for i, label_name in enumerate(label_names):
        idx = labels == i
        plt.scatter(
            reduced[idx, 0], reduced[idx, 1],
            label=label_name, s=10, alpha=0.7
        )
    plt.legend(fontsize=10)
    plt.xticks(fontsize=12, fontweight="bold")
    plt.yticks(fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", dpi=500, format="pdf")
    plt.close()
    print(f"Saved t-SNE plot to {save_path}.pdf")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict GloMER model on IEMOCAP, and ESD")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing test.pkl")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pt")
    parser.add_argument("--dataset", type=str, required=True, choices=["IEMOCAP", "ESD"], help="Dataset")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of emotion classes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="logs", help="Directory to save outputs")
    parser.add_argument("--modality", type=str, choices=["both", "text", "audio"], default="both", help="Modality to predict with")
    return parser.parse_args()


def get_filenames(data_dir, dataset, num_classes, split="test"):
    if dataset == "IEMOCAP":
        assert num_classes == 4, "IEMOCAP supports only 4 classes."
        prefix = f"IEMOCAP_{num_classes}class_BERT_WAV2VEC"
    elif dataset == "ESD":
        assert num_classes == 5, "ESD uses 5 classes."
        prefix = "ESD_BERT_WAV2VEC" 
    else:
        raise ValueError("Dataset must be either 'IEMOCAP', or 'ESD'.")
    return os.path.join(data_dir, f"{prefix}_{split}.pkl")


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.dataset == "IEMOCAP":
        label_names = ["Angry", "Happy", "Sad", "Neutral"]
    elif args.dataset == "ESD":
        label_names = ["Angry", "Happy", "Sad", "Neutral", "Surprise"]

    print("\nLoading test data...")
    test_pkl = get_filenames(args.data_dir, args.dataset, args.num_classes, split="test")
    test_data = load_data(test_pkl).to(device)

    print("Loading model...")
    model = GloMER(
        text_input_dim=768,
        audio_input_dim=768,
        fusion_dim=512,
        projection_dim=256,
        num_heads=8,
        dropout=0.2,
        linear_layer_dims=[512, 128],
        num_classes=args.num_classes
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.model_path))[0]
    suffix = f"_{args.modality}only" if args.modality != "both" else ""
    report_path = os.path.join(args.save_dir, f"{base_name}_classification_report{suffix}.txt")
    cm_path = os.path.join(args.save_dir, f"{base_name}_confusion_matrix{suffix}")
    tsne_path = os.path.join(args.save_dir, f"{base_name}_tsne{suffix}")

    print("\nCalculating model parameters and FLOPs...")
    dummy_text = torch.randn(1, 768).to(device)
    dummy_audio = torch.randn(1, 768).to(device)
    flops, params = profile(model, inputs=(dummy_text, dummy_audio), verbose=False)
    print(f"Model Parameters: {params/1e6:.3f} M")
    print(f"Model FLOPs: {flops/1e9:.3f} GFLOPs")

    repeats = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(dummy_text, dummy_audio)
    end_time = time.time()
    avg_time = (end_time - start_time) / repeats * 1000
    print(f"Average inference time per sample: {avg_time:.2f} ms")

    print(f"\nRunning prediction with modality = {args.modality} ...")
    with torch.no_grad():
        if args.modality == "text":
            audio_dummy = torch.zeros_like(test_data.audio_x)
            outputs = model(test_data.text_x, audio_dummy, return_all=True)
        elif args.modality == "audio":
            text_dummy = torch.zeros_like(test_data.text_x)
            outputs = model(text_dummy, test_data.audio_x, return_all=True)
        else:  
            outputs = model(test_data.text_x, test_data.audio_x, return_all=True)

        logits = outputs["logits"].cpu().numpy()
        probs = softmax(logits, axis=1)
        preds = np.argmax(probs, axis=1)
        labels = test_data.y.cpu().numpy()

    wa, ua, wf1, uf1 = compute_metrics(labels, preds)
    print(f"\nResults:\nWA: {wa:.4f}, UA: {ua:.4f}, WF1: {wf1:.4f}, UF1: {uf1:.4f}")

    report = classification_report(labels, preds, target_names=label_names, digits=4)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved classification report to {report_path}")

    plot_confusion_matrix(labels, preds, label_names, cm_path)

    plot_tsne(probs, labels, label_names, tsne_path)

    print("\nPrediction and evaluation complete.")


if __name__ == "__main__":
    main()
