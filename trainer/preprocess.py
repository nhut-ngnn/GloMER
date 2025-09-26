import argparse
import glob
import logging
import os
import pickle
import random
import soundfile as sf
import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

LABEL_MAP_4 = {
    "ang": 0,
    "hap": 1,
    "sad": 2,
    "neu": 3,
    "exc": 1 
}

LABEL_MAP_MELD_7 = {
    "neutral": 0,
    "surprise": 1,
    "fear": 2,
    "sadness": 3,
    "joy": 4,
    "disgust": 5,
    "anger": 6
}

LABEL_MAP_ESD = {
    "Angry": 0,
    "Happy": 1,
    "Neutral": 2,
    "Sad": 3,
    "Surprise": 4
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def preprocess_iemocap(args):
    session_ids = list(range(1, 6))
    ignore_length = args.ignore_length
    seed = args.seed
    data_root = args.data_root

    label_map = LABEL_MAP_4
    valid_emotions = {"ang", "hap", "sad", "neu", "exc"}

    samples = []
    labels = []

    for sess_id in tqdm.tqdm(session_ids, desc="Processing IEMOCAP"):
        sess_path = os.path.join(data_root, f"Session{sess_id}")
        audio_root = os.path.join(sess_path, "sentences/wav")
        text_root = os.path.join(sess_path, "dialog/transcriptions")
        label_root = os.path.join(sess_path, "dialog/EmoEvaluation")
        label_files = glob.glob(os.path.join(label_root, "*.txt"))

        for label_file in label_files:
            base_name = os.path.basename(label_file)
            transcript_file = os.path.join(text_root, base_name)

            with open(transcript_file, "r") as f:
                transcript_lines = {
                    line.split(":")[0]: line.split(":")[1].strip()
                    for line in f.readlines()
                }

            with open(label_file, "r") as f:
                for line in f:
                    if not line.startswith("["):
                        continue
                    data = line[1:].split()
                    start_time = float(data[0])
                    end_time = float(data[2][:-1])
                    utt_id = data[3]
                    emotion = data[4]

                    if emotion not in valid_emotions:
                        continue

                    folder = utt_id[:-5]
                    wav_name = utt_id + ".wav"
                    wav_path = os.path.join(audio_root, folder, wav_name)

                    try:
                        wav_data, _ = sf.read(wav_path, dtype="int16")
                    except Exception:
                        logging.warning(f"Cannot read {wav_path}")
                        continue

                    if len(wav_data) < ignore_length:
                        logging.warning(f"Ignored short sample: {wav_path}")
                        continue

                    text_key = f"{utt_id} [{start_time:08.4f}-{end_time:08.4f}]"
                    text = transcript_lines.get(text_key)

                    if text is None:
                        text_key_alt1 = f"{utt_id} [{start_time:08.4f}-{end_time + 0.0001:08.4f}]"
                        text_key_alt2 = f"{utt_id} [{start_time + 0.0001:08.4f}-{end_time:08.4f}]"
                        text = transcript_lines.get(text_key_alt1) or transcript_lines.get(text_key_alt2)

                    if text is None:
                        logging.warning(f"Transcript not found: {text_key}")
                        continue

                    label = label_map.get(emotion)
                    if label is None:
                        continue

                    samples.append((wav_path, text, label))
                    labels.append(label)

    data = list(zip(samples, labels))
    random.Random(seed).shuffle(data)
    samples, labels = zip(*data)

    train, test_samples, train_labels, _ = train_test_split(
        samples, labels, test_size=0.1, random_state=seed
    )
    train_samples, val_samples, _, _ = train_test_split(
        train, train_labels, test_size=0.1, random_state=seed
    )

    output_dir = "IEMOCAP_preprocessed/IEMOCAP_4class_preprocessed"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(output_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(output_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    logging.info(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    logging.info(f"Saved preprocessed data to {output_dir}")

def preprocess_meld(args):
    data_root = args.data_root
    ignore_length = args.ignore_length
    seed = args.seed

    splits = ["train", "dev", "test"]
    samples = []

    for split in splits:
        csv_path = os.path.join(data_root, f"{split}_sent_emo.csv")
        df = pd.read_csv(csv_path)

        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=f"Processing MELD {split}"):
            wav_path = os.path.join(data_root, "WAV", f"{row['Utterance_ID']}.wav")
            text = row["Utterance"]
            emotion = row["Emotion"].lower()

            if emotion not in LABEL_MAP_MELD_7:
                continue

            try:
                wav_data, _ = sf.read(wav_path, dtype="int16")
            except Exception:
                logging.warning(f"Cannot read {wav_path}")
                continue

            if len(wav_data) < ignore_length:
                logging.warning(f"Ignored short sample: {wav_path}")
                continue

            label = LABEL_MAP_MELD_7[emotion]
            samples.append((split, (wav_path, text, label)))

    random.Random(seed).shuffle(samples)

    train_samples = [s[1] for s in samples if s[0] == "train"]
    val_samples = [s[1] for s in samples if s[0] == "dev"]
    test_samples = [s[1] for s in samples if s[0] == "test"]

    output_dir = "MELD_preprocessed/MELD_7class_preprocessed"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(output_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(output_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    logging.info(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    logging.info(f"Saved preprocessed data to {output_dir}")
def preprocess_esd(args):
    data_root = args.data_root
    ignore_length = args.ignore_length
    seed = args.seed

    # English speakers are 0011 ~ 0020
    speaker_dirs = [
        os.path.join(data_root, spk)
        for spk in os.listdir(data_root)
        if spk.isdigit() and 11 <= int(spk) <= 20
    ]
    samples = []

    for spk_dir in tqdm.tqdm(speaker_dirs, desc="Processing ESD English"):
        for emo in os.listdir(spk_dir):
            emo_path = os.path.join(spk_dir, emo)
            if not os.path.isdir(emo_path):
                continue

            if emo not in LABEL_MAP_ESD:
                continue

            label = LABEL_MAP_ESD[emo]
            wav_files = glob.glob(os.path.join(emo_path, "*.wav"))

            for wav_path in wav_files:
                try:
                    wav_data, _ = sf.read(wav_path, dtype="int16")
                except Exception:
                    logging.warning(f"Cannot read {wav_path}")
                    continue

                if len(wav_data) < ignore_length:
                    logging.warning(f"Ignored short sample: {wav_path}")
                    continue

                text = ""  
                samples.append((wav_path, text, label))

    random.Random(seed).shuffle(samples)

    labels = [s[2] for s in samples]
    train, test_samples = train_test_split(samples, test_size=0.2, random_state=seed, stratify=labels)
    val_samples, test_samples = train_test_split(test_samples, test_size=0.5, random_state=seed,
                                                stratify=[s[2] for s in test_samples])

    output_dir = "ESD_preprocessed/ESD_5class_preprocessed"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.pkl"), "wb") as f:
        pickle.dump(train, f)
    with open(os.path.join(output_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(output_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    logging.info(f"Train: {len(train)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    logging.info(f"Saved preprocessed data to {output_dir}")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["iemocap", "meld", "esd"], required=True)
    parser.add_argument("--data_root", type=str, required=True, help="Root path to dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore_length", type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    if args.dataset == "iemocap":
        preprocess_iemocap(args)
    elif args.dataset == "meld":
        preprocess_meld(args)
    elif args.dataset == "esd":
        preprocess_esd(args)