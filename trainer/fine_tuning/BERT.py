import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW
from tqdm import tqdm
import warnings
import sys

warnings.filterwarnings("ignore")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)

from src.fine_tuning.BERT import BERTEmbeddingModel, NTXentLoss, TextDataset
from src.fine_tuning.config import BERT_CONFIG


def train_bert_finetune():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dataset = TextDataset(BERT_CONFIG["pkl_path"], tokenizer, max_length=BERT_CONFIG["max_length"])
    dataloader = DataLoader(dataset, batch_size=BERT_CONFIG["batch_size"], shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BERTEmbeddingModel(BERT_CONFIG["embedding_dim"], BERT_CONFIG["projection_dim"]).to(device)
    optimizer = AdamW(model.parameters(), lr=BERT_CONFIG["lr"])
    criterion = NTXentLoss(BERT_CONFIG["temperature"])

    best_loss = float('inf')
    patience = BERT_CONFIG.get("early_stopping_patience", 5)
    patience_counter = 0

    os.makedirs(os.path.dirname(BERT_CONFIG["save_path"]), exist_ok=True)

    for epoch in range(BERT_CONFIG["epochs"]):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{BERT_CONFIG['epochs']}"):
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)

            _, z1 = model(input_ids1, attention_mask1)
            _, z2 = model(input_ids2, attention_mask2)

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
            torch.save({'model_state_dict': model.state_dict()}, BERT_CONFIG["save_path"])
            print(f"Saved best model to {BERT_CONFIG['save_path']}")
        else:
            patience_counter += 1
            print(f"No improvement in loss. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("BERT fine-tuning completed.")


if __name__ == "__main__":
    train_bert_finetune()
