#!/bin/bash
# preprocess.sh
# Run preprocessing for IEMOCAP, MELD, and ESD datasets

set -e  # stop if error occurs

# Paths to datasets (update these to your actual dataset locations)
IEMOCAP_ROOT="/path/to/IEMOCAP"
ESD_ROOT="/path/to/ESD"

# Random seed and ignore length (you can tune if needed)
SEED=42
IGNORE_LENGTH=16000  # ~1 second if 16kHz audio

echo "===== Preprocessing IEMOCAP ====="
python trainer/preprocess.py \
  --dataset iemocap \
  --data_root "$IEMOCAP_ROOT" \
  --seed $SEED \
  --ignore_length $IGNORE_LENGTH

echo "===== Preprocessing ESD ====="
python trainer/preprocess.py \
  --dataset esd \
  --data_root "$ESD_ROOT" \
  --seed $SEED \
  --ignore_length $IGNORE_LENGTH

echo "All preprocessing finished."
