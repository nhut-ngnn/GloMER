
BERT_CONFIG = {
    "pkl_path": "ESD_preprocessed/ESD_5class_preprocessed/train.pkl",
    "save_path": "fine_tuning/ESD/models/best_bert_embeddings.pt",
    "epochs": 10,
    "batch_size": 16,
    "lr": 2e-5,
    "max_length": 128,
    "embedding_dim": 768,
    "projection_dim": 512,
    "temperature": 0.07,
}

WAV_CONFIG = {
    "metadata_path": "ESD_preprocessed/ESD_5class_preprocessed/train.pkl",
    "save_path": "fine_tuning/ESD/models/best_wav2vec_embeddings.pt",
    "segment_length": 16000,
    "embedding_dim": 768,
    "projection_dim": 512,
    "batch_size": 16,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "temperature": 0.07,
    "seed": 42,
}