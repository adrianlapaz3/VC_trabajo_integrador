# src/data_utils.py

import os
import pandas as pd
import logging
from PIL import Image
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split


def process_image_directory(root_directory: str, folder_separator: str = '___') -> pd.DataFrame:
    """
    Recorre un directorio raíz, extrae rutas de imágenes y metadatos de subcarpetas.
    """
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_data = []
    
    if not os.path.isdir(root_directory):
        logging.error(f"El directorio raíz especificado no existe: {root_directory}")
        return pd.DataFrame()

    logging.info(f"Comenzando el procesamiento del directorio: {root_directory}")

    for subdirectory_name in os.listdir(root_directory):
        subdirectory_path = os.path.join(root_directory, subdirectory_name)
        if not os.path.isdir(subdirectory_path):
            continue

        if folder_separator not in subdirectory_name:
            logging.warning(f"'{subdirectory_name}' no contiene el separador '{folder_separator}'. Saltando.")
            continue

        try:
            group_name, tag_name = subdirectory_name.split(folder_separator, 1)
        except ValueError:
            logging.warning(f"No se pudo dividir '{subdirectory_name}'. Saltando directorio.")
            continue

        for filename in os.listdir(subdirectory_path):
            file_path = os.path.join(subdirectory_path, filename)
            if not os.path.isfile(file_path):
                continue

            _, file_extension = os.path.splitext(filename)
            if file_extension.lower() in IMAGE_EXTENSIONS:
                image_data.append({
                    'relative_path': os.path.join(subdirectory_name, filename),
                    'class': subdirectory_name,
                    'group': group_name,
                    'tag': tag_name
                })

    if not image_data:
        logging.warning("No se encontraron imágenes válidas.")

    return pd.DataFrame(image_data)

def load_image(df: pd.DataFrame, index: int, root_dir: str):
    """Carga una imagen PIL desde una fila específica de un DataFrame."""
    # Primero, necesitamos la ruta completa. Asumimos que el dataframe ya tiene
    # una columna 'relative_path' creada por process_image_directory.
    # Si no la tiene, necesitaríamos construirla.
    # Vamos a asumir que 'process_image_directory' guarda la ruta relativa.
    
    try:
        row = df.iloc[index]
        # NOTA: Asegúrate que tu process_image_directory guarde la ruta relativa de esta forma
        full_path = os.path.join(root_dir, row['relative_path'])
        img = Image.open(full_path)
        return img
    except FileNotFoundError:
        print(f"Archivo no encontrado: {full_path}")
        return None
    except Exception as e:
        print(f"Error al cargar la imagen en el índice {index}: {e}")
        return None
    

def calculate_average_histograms_by_category(df: pd.DataFrame, root_dir: str, category_col: str, bins: int = 256, max_images_per_category: int = 50):
    """
    Calcula el histograma RGB promedio para cada categoría única en una columna especificada.

    Args:
        df (pd.DataFrame): DataFrame con la información de las imágenes.
        root_dir (str): Directorio raíz donde están las imágenes.
        category_col (str): Nombre de la columna por la cual agrupar (ej. 'class', 'label_binaria').
        bins (int): Número de bins para el histograma.
        max_images_per_category (int): Límite de imágenes a procesar por categoría para agilizar el cálculo. 
                                     Si es None, se procesan todas.

    Returns:
        pd.DataFrame: Un DataFrame con los histogramas promedio [category, bin, r, g, b].
    """
    
    # Función anidada para cargar imágenes de forma segura
    def load_image(full_path):
        try:
            return Image.open(full_path).convert("RGB")
        except Exception:
            return None

    # Agrupamos por la columna de categoría especificada
    grouped = df.groupby(category_col)
    histograms_data = defaultdict(lambda: {'r': [], 'g': [], 'b': [], 'count': 0})

    print(f"Calculando histogramas promedio por '{category_col}'...")
    
    for category_name, group_df in grouped:
        image_count = 0
        # Usamos la columna 'full_path' que ya habías creado
        for path in group_df["full_path"]:
            if max_images_per_category and image_count >= max_images_per_category:
                break
            
            img = load_image(path)
            if img:
                arr = np.array(img)
                hr, _ = np.histogram(arr[:, :, 0], bins=bins, range=(0, 256))
                hg, _ = np.histogram(arr[:, :, 1], bins=bins, range=(0, 256))
                hb, _ = np.histogram(arr[:, :, 2], bins=bins, range=(0, 256))
                
                histograms_data[category_name]['r'].append(hr)
                histograms_data[category_name]['g'].append(hg)
                histograms_data[category_name]['b'].append(hb)
                image_count += 1
        
        histograms_data[category_name]['count'] = image_count
        print(f"  - Categoría '{category_name}': {image_count} imágenes procesadas.")

    # Convertimos los resultados a un DataFrame tidy
    result_list = []
    for category, hists in histograms_data.items():
        if hists['count'] > 0:
            avg_r = np.mean(hists['r'], axis=0)
            avg_g = np.mean(hists['g'], axis=0)
            avg_b = np.mean(hists['b'], axis=0)
            for i in range(bins):
                result_list.append({
                    "category": category,
                    "bin": i,
                    "r": avg_r[i],
                    "g": avg_g[i],
                    "b": avg_b[i],
                })
    
    return pd.DataFrame(result_list)


def split_data(df: pd.DataFrame,
               target_column: str = 'class',
               test_size: float = 0.2,
               validation_size: float = 0.0,
               random_state: int = 42,
               split_column_name: str = 'split') -> pd.DataFrame:
    """
    Añade una columna al DataFrame indicando la división (train/valid/test) usando estratificación.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        target_column (str): Columna a usar para la estratificación.
        test_size (float): Proporción para el conjunto de test.
        validation_size (float): Proporción para el conjunto de validación.
        random_state (int): Semilla para reproducibilidad.
        split_column_name (str): Nombre de la columna de salida con 'train', 'valid', 'test'.

    Returns:
        pd.DataFrame: DataFrame con la nueva columna de división.
    """
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no se encuentra en el DataFrame.")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size debe ser un float > 0 y < 1.")
    if not (0.0 <= validation_size < 1.0):
        raise ValueError("validation_size debe ser un float >= 0 y < 1.")
    if test_size + validation_size >= 1.0:
        raise ValueError("La suma de test_size y validation_size debe ser menor que 1.0.")

    df[split_column_name] = 'unassigned'
    labels = df[target_column]
    indices = df.index

    # Primera división: separar el conjunto de TEST del resto (TRAIN + VALID)
    remaining_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    df.loc[test_indices, split_column_name] = 'test'

    # Si se necesita un conjunto de validación
    if validation_size > 0:
        # Calcular el tamaño relativo de validación sobre el conjunto restante
        relative_val_size = validation_size / (1.0 - test_size)
        remaining_labels = df.loc[remaining_indices, target_column]

        # Segunda división: separar TRAIN y VALID del conjunto restante
        train_indices, validation_indices = train_test_split(
            remaining_indices,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=remaining_labels
        )
        df.loc[train_indices, split_column_name] = 'train'
        df.loc[validation_indices, split_column_name] = 'valid'
    else:
        # Si no hay validación, todo lo restante es entrenamiento
        df.loc[remaining_indices, split_column_name] = 'train'

    # Imprimir resumen de la división
    train_prop = (df[split_column_name] == 'train').mean()
    valid_prop = (df[split_column_name] == 'valid').mean()
    test_prop = (df[split_column_name] == 'test').mean()
    print(f"División completada y estratificada por '{target_column}':")
    print(f"  - Train:      {train_prop:.1%}")
    if valid_prop > 0:
        print(f"  - Validation: {valid_prop:.1%}")
    print(f"  - Test:       {test_prop:.1%}")
    
    return df
