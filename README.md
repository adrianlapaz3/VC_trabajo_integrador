# Clasificación de enfermedades en hojas de plantas con inteligencia artificial

Este repositorio contiene el código y los experimentos para un proyecto de clasificación de imágenes, cuyo objetivo es identificar 38 clases diferentes (enfermedades y estados saludables) en hojas de 14 especies de plantas utilizando redes neuronales convolucionales (CNNs).

## Set de datos

Este proyecto utiliza el dataset **"PlantVillage"**, que contiene más de 54,000 imágenes de hojas de plantas. El dataset está disponible públicamente en Kaggle.
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data

## Estructura del proyecto
El proyecto está organizado para mantener un flujo de trabajo limpio y reproducible:
* **/data:** Contiene los DataFrames procesados (`.csv`). La data cruda (imágenes) se mantiene fuera del repositorio y debe ser descargada por separado.
* **/notebooks:** Cuadernos de Jupyter que documentan el proceso de forma secuencial.
* **/src:** Código fuente reutilizable (ej. funciones para construir arquitecturas de modelos).
* **/outputs:** Almacena los artefactos de cada experimento (modelos guardados, logs, gráficos). Ignorado por Git.

---

## Metodología del proyecto
El flujo de trabajo se divide en las siguientes etapas, cada una documentada en su respectivo notebook:

### 1. Análisis exploratorio y división de datos (`1_exploracion_y_split.ipynb`)
En esta primera fase, se realiza un análisis inicial del dataset para entender la distribución de clases. Posteriormente, el conjunto de datos se divide de forma estratificada en:
* **Entrenamiento:** 70%
* **Validación:** 20%
* **Testeo:** 10%

El resultado de este proceso es el archivo `data/processed/dataframe_splitted.csv`, que sirve como base para las siguientes etapas.

### 2. Estrategias de Aumento de Datos (`2_aumento_datos.ipynb`)
Se definen y preparan las estrategias de pre-procesamiento que serán comparadas. El objetivo es determinar el impacto de dos factores clave:

* **Balanceo de clases:** Se compara el efecto de entrenar con el dataset original (con su desbalance natural) contra una versión balanceada mediante **sobremuestreo (oversampling)**. El resultado de este proceso es el archivo `data/processed/df_train_balanced.csv`.
* **Técnicas de aumentación:** Se definen dos pipelines con `Albumentations`:
    1.  **Aumentación geométrica:** Solo aplica transformaciones de forma (rotación, escala, zoom, etc.).
    2.  **Aumentación geométrica + color:** Añade transformaciones fotométricas que alteran el color, como brillo, contraste, matiz y saturación (HSV).

### 3. Entrenamiento de modelos baseline (`3_baseline.ipynb`)
Esta es la fase de experimentación principal para establecer un punto de partida sólido.
* **Arquitectura:** Se utiliza **InceptionV3** pre-entrenada en ImageNet.
* **Estrategia de entrenamiento:** Se aplica la técnica de **Feature Extraction**, manteniendo congelada la base del modelo y entrenando únicamente un clasificador personalizado añadido al final.
* **Selección del mejor modelo base:** Se ejecutan **4 experimentos** cruzando las estrategias de datos (balanceado vs. no balanceado) y de aumentación (con color vs. sin color) para seleccionar la combinación con el mejor rendimiento en el conjunto de validación.

### 4. Fine-Tuning del modelo base seleccionado (`4_fine_tuning.ipynb`)
Una vez seleccionada la mejor estrategia del baseline, se procede a la fase de ajuste fino para maximizar la precisión:
* Se carga el mejor modelo guardado de la etapa anterior.
* Se "descongelan" las capas superiores del modelo `InceptionV3`.
* Se ejecutan y comparan dos experimentos de re-entrenamiento:
    * **Estrategia de Fine-Tuning 1:** se descongelan y re-entrenan los **últimos 2 bloques convolucionales** del modelo.
    * **Estrategia de Fine-Tuning 2:** se descongelan y re-entrenan los **últimos 4 bloques convolucionales** del modelo.


### 5. Evaluación final sobre datos de test (`5_performance.ipynb`)
El rendimiento final y definitivo del mejor modelo (después del fine-tuning) se mide sobre el **conjunto de testeo (10% de los datos)**. Este conjunto no ha sido utilizado en ninguna etapa anterior, lo que garantiza una métrica imparcial de la capacidad de generalización del modelo. Se analizan métricas como `Accuracy`, `Precision`, `Recall`, `F1-score` y la **matriz de confusión**.
