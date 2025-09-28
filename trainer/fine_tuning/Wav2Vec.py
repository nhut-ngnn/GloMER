import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import warnings
import sys

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)

from src.fine_tuning.Wav2Vec import AudioEmbeddingModel, NTXentLoss, AudioDataset
from src.fine_tuning.config import WAV_CONFIG

def train_wav2vec_finetune():
    # set_seed(WAV_CONFIG["seed"])

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    dataset = AudioDataset(WAV_CONFIG["metadata_path"], processor, WAV_CONFIG["segment_length"])
    dataloader = DataLoader(dataset, batch_size=WAV_CONFIG["batch_size"], shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioEmbeddingModel(
        embedding_dim=WAV_CONFIG["embedding_dim"],
        projection_dim=WAV_CONFIG["projection_dim"]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=WAV_CONFIG["learning_rate"])
    criterion = NTXentLoss(WAV_CONFIG["temperature"])

    best_loss = float("inf")
    patience = WAV_CONFIG.get("early_stopping_patience", 5)  # Add to config if not already there
    patience_counter = 0

    os.makedirs(os.path.dirname(WAV_CONFIG["save_path"]), exist_ok=True)

    for epoch in range(WAV_CONFIG["num_epochs"]):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{WAV_CONFIG['num_epochs']}"):
            x1 = batch["input_values1"].to(device)
            x2 = batch["input_values2"].to(device)

            _, z1 = model(x1)
            _, z2 = model(x2)

            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict()}, WAV_CONFIG["save_path"])
            print(f"Saved best model to {WAV_CONFIG['save_path']}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Wav2Vec fine-tuning complete.")

if __name__ == "__main__":
    train_wav2vec_finetune()
