import torch.nn as nn
from transformers import BertModel

# Modelo multitarea basado en BERT para tres salidas simultáneas
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_rankings, num_towns, num_categories):
        super(MultiTaskModel, self).__init__()
        # Carga modelo BERT preentrenado
        self.bert = BertModel.from_pretrained(model_name)
        # Dropout para regularización
        self.dropout = nn.Dropout(0.3)
        # Obtiene tamaño del vector oculto de BERT
        hidden_size = self.bert.config.hidden_size
        # Cabezas lineales para cada tarea
        self.ranking_head = nn.Linear(hidden_size, num_rankings)     # Clasifica polaridad
        self.town_head = nn.Linear(hidden_size, num_towns)           # Clasifica pueblo mágico
        self.category_head = nn.Linear(hidden_size, num_categories)  # Clasifica tipo destino

    def forward(self, input_ids, attention_mask):
        # Forward pass por BERT con entrada tokenizada y máscara
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extrae vector pooled output (representación [CLS])
        pooled_output = self.dropout(outputs.pooler_output)
        # Predicciones para cada tarea
        ranking_logits = self.ranking_head(pooled_output)
        town_logits = self.town_head(pooled_output)
        category_logits = self.category_head(pooled_output)
        # Retorna logits para las tres salidas
        return ranking_logits, town_logits, category_logits
