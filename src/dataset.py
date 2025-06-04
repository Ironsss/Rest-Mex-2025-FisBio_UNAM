import torch
from torch.utils.data import Dataset

# Dataset para entrenamiento y validación, incluye etiquetas multitarea
class RestMexDataset(Dataset):
    def __init__(self, texts, rankings, towns, categories, tokenizer, max_len):
        # Guardar listas de textos y etiquetas, tokenizer y longitud máxima
        self.texts = texts
        self.rankings = rankings
        self.towns = towns
        self.categories = categories
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        # Devuelve el número de muestras en el dataset
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokeniza y codifica el texto en índice idx
        encoding = self.tokenizer(
            self.texts[idx],            # Texto a tokenizar
            padding='max_length',       # Rellenar para longitud fija
            truncation=True,            # Truncar si texto es muy largo
            max_length=self.max_len,    # Longitud máxima permitida
            return_tensors='pt'         # Retorna tensores PyTorch
        )
        # Retorna diccionario con tensores de entrada y etiquetas
        return {
            'input_ids': encoding['input_ids'].squeeze(0),          # IDs tokens, sin dimensión batch
            'attention_mask': encoding['attention_mask'].squeeze(0),# Máscara atención, sin dimensión batch
            'ranking': torch.tensor(self.rankings[idx], dtype=torch.long),   # Etiqueta ranking (sentimiento)
            'town': torch.tensor(self.towns[idx], dtype=torch.long),         # Etiqueta pueblo mágico
            'category': torch.tensor(self.categories[idx], dtype=torch.long) # Etiqueta tipo destino
        }

# Dataset para inferencia: solo textos, sin etiquetas
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts            # Lista de textos a inferir
        self.tokenizer = tokenizer    # Tokenizer para codificación
        self.max_len = max_len        # Longitud máxima permitida

    def __len__(self):
        # Número de textos para inferir
        return len(self.texts)

    def __getitem__(self, idx):
        # Codifica texto para inferencia
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        # Retorna tensores de entrada para el modelo
        return {
            'input_ids': encoding['input_ids'].squeeze(0),          # IDs tokens
            'attention_mask': encoding['attention_mask'].squeeze(0) # Máscara atención
        }
