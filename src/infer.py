import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import MultiTaskModel
from dataset import InferenceDataset

MODEL_NAME = 'bert-base-multilingual-cased'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 128
BATCH_SIZE = 8

# Carga dataset entrenamiento solo para obtener etiquetas consistentes
train_data = pd.read_csv('data/Rest-Mex_2025_train.csv')
MAGICAL_TOWNS = sorted(train_data['Town'].astype(str).str.strip().dropna().unique().tolist())
CATEGORY_LABELS = sorted(train_data['Type'].astype(str).str.strip().dropna().unique().tolist())

idx_to_ranking = {i: i + 1 for i in range(5)}
idx_to_town = {i: name for i, name in enumerate(MAGICAL_TOWNS)}
idx_to_category = {i: name for i, name in enumerate(CATEGORY_LABELS)}

def main():
    # Carga tokenizer guardado
    tokenizer = BertTokenizer.from_pretrained('./trained_tokenizer')

    # Inicializa modelo y carga pesos entrenados
    model = MultiTaskModel(MODEL_NAME, num_rankings=5, num_towns=len(MAGICAL_TOWNS), num_categories=len(CATEGORY_LABELS))
    model.load_state_dict(torch.load('trained_model.pt', map_location=DEVICE))
    model
