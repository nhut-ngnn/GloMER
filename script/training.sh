#!/bin/bash
# run_experiments.sh

# Exit if any command fails
set -e

# Make sure to run preprocess.sh and feature_extract.sh before this script
echo "=== Training GloMER ==="
python trainer/train.py \
    --data_dir "data/processed" \
    --dataset IEMOCAP \
    --num_classes 4 \
    --epochs 100 \
    --alpha 0.3 \
    --batch_size 128
