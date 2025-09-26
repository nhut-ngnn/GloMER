import os
import sys
import torch
import pickle
import torchaudio
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_extract.config import (
    PKL_DIR, OUTPUT_DIR, device,
    TOKENIZER, AUDIO_PROCESSOR,
    TEXT_MODEL, AUDIO_MODEL
)

def extract_text_features(text, tokenizer, model, device):
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            pooled, _ = model(encoded["input_ids"], encoded["attention_mask"])
        return pooled.squeeze().cpu()
    except Exception as e:
        print(f"[TEXT] Error: {e}")
        return None

def extract_audio_features(audio_path, processor, model, device):
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resample(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            pooled, _ = model(input_values)
        return pooled.squeeze().cpu()
    except Exception as e:
        print(f"[AUDIO] Error in {audio_path}: {e}")
        return None

def process_single_sample(audio_path, text, label):
    text_embed = extract_text_features(text, TOKENIZER, TEXT_MODEL, device)
    audio_embed = extract_audio_features(audio_path, AUDIO_PROCESSOR, AUDIO_MODEL, device)

    if text_embed is not None and audio_embed is not None:
        return {
            'text_embed': text_embed,
            'audio_embed': audio_embed,
            'label': torch.tensor(label) if isinstance(label, (int, float)) else label,
            'sample_id': f"{os.path.basename(audio_path)}"
        }
    else:
        return None

def process_dataset(pkl_path, output_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    processed_samples = []
    print(f"Processing {len(data)} samples from {pkl_path}")

    for idx, (audio_path, text, label) in tqdm(enumerate(data), total=len(data)):
        sample = process_single_sample(audio_path, text, label)
        if sample is not None:
            processed_samples.append(sample)
        else:
            print(f"[SKIP] Failed to process: {audio_path}")

    with open(output_path, "wb") as f:
        pickle.dump(processed_samples, f)

    print(f"Saved processed data to: {output_path}")
    print(f"Total processed samples: {len(processed_samples)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["IEMOCAP", "ESD"], help="Dataset to process")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Starting feature extraction...")
    print(f"Using device: {device}")

    if args.dataset == "IEMOCAP":
        pkl_prefix = "IEMOCAP_4class"
    elif args.dataset == "ESD":
        pkl_prefix = "ESD"

    datasets = [
        ("train", f"{pkl_prefix}_preprocessed/train.pkl", f"{pkl_prefix}_BERT_WAV2VEC_train.pkl"),
        ("val",   f"{pkl_prefix}_preprocessed/val.pkl",   f"{pkl_prefix}_BERT_WAV2VEC_val.pkl"),
        ("test",  f"{pkl_prefix}_preprocessed/test.pkl",  f"{pkl_prefix}_BERT_WAV2VEC_test.pkl")
    ]

    for split_name, pkl_file, output_file in datasets:
        print(f"\n{'='*50}")
        print(f"Processing {split_name} split: {pkl_file}")
        print(f"{'='*50}")
        process_dataset(
            os.path.join(PKL_DIR, pkl_file),
            os.path.join(OUTPUT_DIR, output_file)
        )

if __name__ == "__main__":
    main()
