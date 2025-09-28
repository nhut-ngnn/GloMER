#!/bin/bash
# run_prediction.sh

set -e  # stop on error

# ===== User Config =====
DATA_DIR="data/processed"
SAVE_DIR="logs"
MODEL_PATH="saved_model/IEMOCAP_4class_GloMER_seed42.pt"

# ===== Run for IEMOCAP =====
echo "=== Predicting on IEMOCAP ==="
python trainer/GloMER_predict.py \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL_PATH" \
    --dataset IEMOCAP \
    --num_classes 4 \
    --save_dir "$SAVE_DIR" \
    --modality both

# ===== Run for ESD =====
echo "=== Predicting on ESD ==="
python trainer/GloMER_predict.py \
    --data_dir "$DATA_DIR" \
    --model_path "saved_model/ESD_5class_GloMER_seed42.pt" \
    --dataset ESD \
    --num_classes 5 \
    --save_dir "$SAVE_DIR" \
    --modality both

echo "All predictions finished"
