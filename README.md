# Rest-Mex 2025 - Equipo FisBio UNAM

Este repositorio contiene la solución oficial desarrollada por el equipo FisBio UNAM para la competencia Rest-Mex 2025, organizada en el marco del congreso IberLEF 2025. La solución aborda un problema de clasificación multitarea en procesamiento de lenguaje natural (PLN), aplicado a reseñas turísticas en español sobre los Pueblos Mágicos de México.


**Autores:**  
David Alexis García-Espinosa (Autor principal, ORCID: 0000-0001-6141-407X)  
Luis Eduardo Flores Luna (Coautor)  
Andrés Moreno Sánchez (Coautor)

**Afiliación:**  
Universidad Nacional Autónoma de México, Facultad de Ciencias, Ciudad de México, México

**Contacto:**  
davale@ciencias.unam.mx

---

## Descripción

Este repositorio contiene la solución desarrollada para la competencia **Rest-Mex 2025**, realizada en el marco de la materia *Temas Selectos en Biomatemáticas: Introducción a la Ciencia de Datos Aplicada a Escenarios Médico-Biológicos*.

La tarea consiste en un problema de clasificación multitarea de reseñas turísticas en español sobre los "Pueblos Mágicos" de México, que contempla simultáneamente:  
- Predicción de la polaridad del sentimiento en una escala de 5 niveles.  
- Clasificación del tipo de destino: atractivo turístico, hotel o restaurante.  
- Identificación del Pueblo Mágico entre 40 posibles localidades.

El modelo base es un BERT multilingüe finamente ajustado para esta tarea multitarea.

---

## Contenido

- `src/train.py`: Script para entrenar el modelo multitarea con el conjunto de entrenamiento.  
- `src/evaluate.py`: Script para evaluar el modelo con un conjunto de validación.  
- `src/infer.py`: Script para realizar inferencias con datos de prueba sin etiquetas.  
- `src/model.py`: Definición del modelo multitarea basado en BERT.  
- `src/dataset.py`: Definición de datasets personalizados para entrenamiento e inferencia.  


---

## Requisitos

Las dependencias principales son:

```bash
pip install -r requirements.txt

...

---

## Resultados oficiales

Los resultados oficiales de la competencia Rest-Mex 2025 están disponibles en:  
[https://sites.google.com/cimat.mx/rest-mex-2025/results?authuser=0](https://sites.google.com/cimat.mx/rest-mex-2025/results?authuser=0)

...

