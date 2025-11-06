import os
import sys
import torch
import pickle
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.utils import set_seed, train_and_evaluate
from src.architecture.GloMER import GloMER
from src.architecture.Losses import NTXentLoss, DiversityContrastiveLoss, ConsistencyContrastiveLoss
from torch.utils.data import TensorDataset

def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    text = torch.stack([item['text_embed'] for item in data])
    audio = torch.stack([item['audio_embed'] for item in data])
    labels = torch.tensor([item['label'] for item in data])
    return TensorDataset(text, audio, labels)


def combined_loss(outputs, labels, ntxent, dcl, ccl, ce_loss, alpha=0.3):
    ce = ce_loss(outputs["logits"], labels)
    ntx = ntxent(outputs["text_proj"], outputs["audio_proj"])
    div = dcl(outputs["text_pool"], outputs["audio_pool"])
    con = ccl(outputs["text_pool"], outputs["audio_pool"])
    return ce + ntx + alpha * div + alpha * con

def get_filenames(data_dir, dataset, num_classes):
    if dataset == "IEMOCAP":
        assert num_classes == 4, "IEMOCAP now only supports 4 classes."
        prefix = "IEMOCAP_4class_BERT_WAV2VEC"
    elif dataset == "ESD":
        assert num_classes == 5, "ESD uses 5 classes."
        prefix = "ESD_BERT_WAV2VEC"
    else:
        raise ValueError("Dataset must be either 'IEMOCAP', or 'ESD'.")

    return {
        "train": os.path.join(data_dir, f"{prefix}_train.pkl"),
        "val": os.path.join(data_dir, f"{prefix}_val.pkl"),
        "test": os.path.join(data_dir, f"{prefix}_test.pkl"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train GloMER on IEMOCAP, or ESD with 5 seeds")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["IEMOCAP", "ESD"])
    parser.add_argument("--num_classes", type=int, required=True, choices=[4, 5, 7],
                        help="4 for IEMOCAP, 5 for ESD")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and evaluation")

    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [42, 52, 103, 128, 923]

    wa_list, ua_list, wf1_list, uf1_list = [], [], [], []

    for seed in seeds:
        set_seed(seed)
        print(f"\n=== Running with seed {seed} ===")

        filenames = get_filenames(args.data_dir, args.dataset, args.num_classes)
        train_data = load_data(filenames["train"])
        val_data = load_data(filenames["val"])
        test_data = load_data(filenames["test"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        ntxent = NTXentLoss(temperature=0.05)
        dcl = DiversityContrastiveLoss()
        ccl = ConsistencyContrastiveLoss()
        ce_loss_fn = torch.nn.CrossEntropyLoss()

        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, verbose=True)

        loss_fn = lambda out, y, _: combined_loss(
            out, y, ntxent, dcl, ccl, ce_loss_fn,
            alpha=args.alpha
        )

        save_path = f"saved_model/{args.dataset}_{args.num_classes}class_GloMER_seed{seed}.pt"

        metrics = train_and_evaluate(
            model, train_data, val_data, test_data,
            optimizer, scheduler,
            loss_fn,
            alpha=args.alpha,
            epochs=args.epochs,
            save_path=save_path,
            seed=seed,
            batch_size=args.batch_size  
        )

        wa_list.append(metrics["test_WA"])
        ua_list.append(metrics["test_UA"])
        wf1_list.append(metrics["test_WF1"])
        uf1_list.append(metrics["test_UF1"])

        print(f"Seed {seed} - WA: {metrics['test_WA']:.4f}, UA: {metrics['test_UA']:.4f}, "
              f"WF1: {metrics['test_WF1']:.4f}, UF1: {metrics['test_UF1']:.4f}")

    print("\n=== Average Results over 5 seeds ===")
    print(f"Avg WA:  {np.mean(wa_list):.4f}, {np.std(wa_list, ddof=1):.4f}")
    print(f"Avg UA:  {np.mean(ua_list):.4f}, {np.std(ua_list, ddof=1):.4f}")
    print(f"Avg WF1: {np.mean(wf1_list):.4f}, {np.std(wf1_list, ddof=1):.4f}")
    print(f"Avg UF1: {np.mean(uf1_list):.4f}, {np.std(uf1_list, ddof=1):.4f}")

    results_df = pd.DataFrame({
        "Metric": ["WA", "UA", "WF1", "UF1"],
        "Mean": [np.mean(wa_list), np.mean(ua_list), np.mean(wf1_list), np.mean(uf1_list)],
        "Std": [np.std(wa_list, ddof=1), np.std(ua_list, ddof=1), np.std(wf1_list, ddof=1), np.std(uf1_list, ddof=1)]
    })

    os.makedirs("results", exist_ok=True)
    results_df.to_csv(f"results/{args.dataset}_{args.num_classes}class_GloMER_5seeds.csv", index=False)
    print(f"Results saved to results/{args.dataset}_{args.num_classes}class_GloMER_5seeds.csv")

if __name__ == "__main__":
    main()
