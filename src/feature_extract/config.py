import torch
from transformers import BertTokenizer, Wav2Vec2Processor
from .model_encode import BERTEmbeddingModel, AudioEmbeddingModel

PKL_DIR = "IEMOCAP_preprocessed"
OUTPUT_DIR = "features_output"
TEXT_CKPT_PATH = "Fine_tuning/ESD/models/best_bert_embeddings.pt"
AUDIO_CKPT_PATH = "fine_tuning/ESD/models/best_wav2vec_embeddings.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
AUDIO_PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

TEXT_MODEL = BERTEmbeddingModel().to(device)
AUDIO_MODEL = AudioEmbeddingModel().to(device)

TEXT_MODEL.load_state_dict(torch.load(TEXT_CKPT_PATH)["model_state_dict"])
AUDIO_MODEL.load_state_dict(torch.load(AUDIO_CKPT_PATH)["model_state_dict"])

TEXT_MODEL.eval()
AUDIO_MODEL.eval()
