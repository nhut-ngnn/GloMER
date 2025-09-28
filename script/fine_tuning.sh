#!/bin/bash
# fine_tune.sh
# Run fine-tuning for BERT and Wav2Vec2

set -e  # stop if error occurs

echo "===== Fine-tuning BERT ====="
python trainer/fine_tuning/BERT.py

echo "===== Fine-tuning Wav2Vec2 ====="
python trainer/fine_tuning/Wav2Vec2.py

echo "All fine-tuning finished."
