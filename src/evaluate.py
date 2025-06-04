import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import f1_score
from tqdm import tqdm
from model import MultiTaskModel
from dataset import RestMexDataset
from sklearn.model_selection import train_test_split

MODEL_NAME = 'bert-base-multilingual-cased'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
MAX_LEN = 128

# Carga datos y limpieza idéntica a train.py para obtener etiquetas y texto
data = pd.read_csv('data/Rest-Mex_2025_train.csv')
data['Town'] = data['Town'].astype(str).str.strip()
data['Type'] = data['Type'].astype(str).str.strip()
data['Title'] = data['Title'].fillna('').astype(str).str.strip()
data['Review'] = data['Review'].fillna('').astype(str).str.strip()
data['Text'] = data['Title'] + ' ' + data['Review']

# Obtiene etiquetas únicas ordenadas
MAGICAL_TOWNS = sorted(data['Town'].dropna().unique().tolist())
CATEGORY_LABELS = sorted(data['Type'].dropna().unique().tolist())

idx_to_ranking = {i: i + 1 for i in range(5)}
idx_to_town = {i: name for i, name in enumerate(MAGICAL_TOWNS)}
idx_to_category = {i: name for i, name in enumerate(CATEGORY_LABELS)}

ranking_to_idx = {v: k for k, v in idx_to_ranking.items()}
town_to_idx = {v: k for k, v in idx_to_town.items()}
category_to_idx = {v: k for k, v in idx_to_category.items()}

texts = data['Text'].tolist()
rankings = [ranking_to_idx[int(r)] for r in data['Polarity']]
towns = [town_to_idx[t] for t in data['Town']]
categories = [category_to_idx[c] for c in data['Type']]

# Separar conjunto de validación (20%)
_, test_texts, _, test_rankings, _, test_towns, _, test_categories = train_test_split(
    texts, rankings, towns, categories, test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained('./trained_tokenizer')
test_dataset = RestMexDataset(test_texts, test_rankings, test_towns, test_categories, tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

# Carga modelo entrenado
model = MultiTaskModel(MODEL_NAME, num_rankings=5, num_towns=len(MAGICAL_TOWNS), num_categories=len(CATEGORY_LABELS))
model.load_state_dict(torch.load('trained_model.pt', map_location=DEVICE))
model.to(DEVICE)
model.eval()

ranking_preds_all, ranking_labels_all = [], []
town_preds_all, town_labels_all = [], []
category_preds_all, category_labels_all = [], []
results = []
line_counter = 0

with torch.inference_mode():
    pbar = tqdm(test_loader, desc="Evaluating")
    for batch in pbar:
        # Llevar datos a dispositivo
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        ranking_labels = batch['ranking'].cpu().tolist()
        town_labels = batch['town'].cpu().tolist()
        category_labels = batch['category'].cpu().tolist()

        # Forward pass
        ranking_logits, town_logits, category_logits = model(input_ids, attention_mask)

        # Predicciones
        ranking_preds = torch.argmax(ranking_logits, dim=1).cpu().tolist()
        town_preds = torch.argmax(town_logits, dim=1).cpu().tolist()
        category_preds = torch.argmax(category_logits, dim=1).cpu().tolist()

        # Guardar etiquetas y predicciones
        ranking_preds_all.extend(ranking_preds)
        ranking_labels_all.extend(ranking_labels)
        town_preds_all.extend(town_preds)
        town_labels_all.extend(town_labels)
        category_preds_all.extend(category_preds)
        category_labels_all.extend(category_labels)

        # Preparar líneas de resultado para archivo
        for i in range(len(ranking_preds)):
            ranking = idx_to_ranking[ranking_preds[i]]
            town = idx_to_town[town_preds[i]]
            category = idx_to_category[category_preds[i]]
            line = f'rest-mex\t{line_counter}\t{ranking}\t{town}\t{category}\n'
            results.append(line)
            line_counter += 1

# Calcular métricas macro F1 para cada tarea
resp_k = f1_score(ranking_labels_all, ranking_preds_all, average='macro')
rest_k = f1_score(category_labels_all, category_preds_all, average='macro')
resmt_k = f1_score(town_labels_all, town_preds_all, average='macro')

# Cálculo final ponderado
sentiment_k = (2 * resp_k + rest_k + 3 * resmt_k) / 6

# Mostrar métricas
print(f"F1 Polarity (Resp_k): {resp_k:.4f}")
print(f"F1 Type (Rest_k): {rest_k:.4f}")
print(f"F1 Magical Town (ResMT_k): {resmt_k:.4f}")
print(f"Final Sentiment Score: {sentiment_k:.4f}")

# Guardar resultados en archivo de texto
with open('autor_run1.txt', 'w', encoding='utf-8') as f:
    f.writelines(results)

print("Results saved in 'autor_run1.txt'")
