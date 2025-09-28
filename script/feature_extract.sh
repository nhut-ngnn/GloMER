#!/bin/bash
# feature_extract.sh
# Run feature extraction for IEMOCAP and ESD datasets

set -e  # stop if error occurs

echo "===== Extracting features for IEMOCAP ====="
python trainer/feature_extraction.py --dataset IEMOCAP

echo "===== Extracting features for ESD ====="
python trainer/feature_extraction.py --dataset ESD

echo "All feature extraction finished"
