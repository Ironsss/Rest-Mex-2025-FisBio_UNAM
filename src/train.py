import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch import nn
from model import MultiTaskModel
from dataset import RestMexDataset

# Configuraciones generales
MODEL_NAME = 'bert-base-multilingual-cased'       # Modelo base BERT multilingüe
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU si disponible
BATCH_SIZE = 16
MAX_LEN = 128
EPOCHS = 3
LEARNING_RATE = 2e-5

print(f"Running on device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

# Carga y limpieza de datos
data = pd.read_csv('data/Rest-Mex_2025_train.csv')
data['Town'] = data['Town'].astype(str).str.strip()           # Limpia espacios en Town
data['Type'] = data['Type'].astype(str).str.strip()           # Limpia espacios en Type
data['Title'] = data['Title'].fillna('').astype(str).str.strip() # Limpia título
data['Review'] = data['Review'].fillna('').astype(str).str.strip() # Limpia reseña
data['Text'] = data['Title'] + ' ' + data['Review']           # Une título + reseña

# Obtiene etiquetas únicas ordenadas
MAGICAL_TOWNS = sorted(data['Town'].dropna().unique().tolist())
CATEGORY_LABELS = sorted(data['Type'].dropna().unique().tolist())

# Mapas para etiquetas: índice a etiqueta y viceversa
idx_to_ranking = {i: i + 1 for i in range(5)}    # Polaridad 1-5
idx_to_town = {i: name for i, name in enumerate(MAGICAL_TOWNS)}
idx_to_category = {i: name for i, name in enumerate(CATEGORY_LABELS)}

ranking_to_idx = {v: k for k, v in idx_to_ranking.items()}
town_to_idx = {v: k for k, v in idx_to_town.items()}
category_to_idx = {v: k for k, v in idx_to_category.items()}

# Convertir textos y etiquetas a listas y codificar etiquetas numéricas
texts = data['Text'].tolist()
rankings = [ranking_to_idx[int(r)] for r in data['Polarity']]
towns = [town_to_idx[t] for t in data['Town']]
categories = [category_to_idx[c] for c in data['Type']]

# Divide en train y test
train_texts, test_texts, train_rankings, test_rankings, train_towns, test_towns, train_categories, test_categories = train_test_split(
    texts, rankings, towns, categories, test_size=0.2, random_state=42
)

# Carga tokenizer preentrenado
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Crea datasets y dataloaders
train_dataset = RestMexDataset(train_texts, train_rankings, train_towns, train_categories, tokenizer, MAX_LEN)
test_dataset = RestMexDataset(test_texts, test_rankings, test_towns, test_categories, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

# Inicializa modelo multitarea y envía a dispositivo
model = MultiTaskModel(MODEL_NAME, num_rankings=5, num_towns=len(MAGICAL_TOWNS), num_categories=len(CATEGORY_LABELS))
model.to(DEVICE)

# Define optimizador y función de pérdida
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()  # Para entrenamiento con mixed precision

# Loop principal de entrenamiento
for epoch in range(EPOCHS):
    model.train()                  # Modo entrenamiento
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Training")
    for batch in pbar:
        # Lleva tensores a GPU o CPU
        input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
        attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
        ranking_labels = batch['ranking'].to(DEVICE, non_blocking=True)
        town_labels = batch['town'].to(DEVICE, non_blocking=True)
        category_labels = batch['category'].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()      # Limpia gradientes previos
        with torch.cuda.amp.autocast():  # Cálculo con mixed precision
            # Forward pass multitarea
            ranking_logits, town_logits, category_logits = model(input_ids, attention_mask)
            # Calcula pérdidas para cada tarea
            ranking_loss = criterion(ranking_logits, ranking_labels)
            town_loss = criterion(town_logits, town_labels)
            category_loss = criterion(category_logits, category_labels)
            loss = ranking_loss + town_loss + category_loss   # Pérdida total

        scaler.scale(loss).backward()  # Backpropagation
        scaler.step(optimizer)          # Actualiza pesos
        scaler.update()                 # Actualiza escalador

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)  # Promedio pérdida epoch
    print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

    # Guardar modelo entrenado en disco
    torch.save(model.state_dict(), 'trained_model.pt')
    print("Model saved as 'trained_model.pt'")

# Guardar tokenizer para uso futuro
tokenizer.save_pretrained('./trained_tokenizer')
print("Tokenizer saved to './trained_tokenizer'")
