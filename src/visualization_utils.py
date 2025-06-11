# src/visualization_utils.py

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
import random 
import math 
import cv2 
import albumentations as A 
from PIL import Image 

from src.data_utils import load_image 

def plot_image_grid(df, root_dir, n_rows=5, n_cols=5, figsize=(20, 20)):
    """
    Muestra una grilla de imágenes aleatorias desde el DataFrame.

    Args:
        df (pd.DataFrame): DataFrame que contiene la información de las imágenes.
        root_dir (str): La ruta raíz donde se encuentran las carpetas de clases.
        n_rows (int): Número de filas en la grilla.
        n_cols (int): Número de columnas en la grilla.
        figsize (tuple): Tamaño de la figura de matplotlib.
    """
    total_images = n_rows * n_cols
    
    # Nos aseguramos de no pedir más imágenes de las que hay
    num_samples = min(total_images, len(df))
    random_indices = random.sample(range(len(df)), num_samples)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() # Aplanamos el array de ejes para iterar fácilmente

    for i, index in enumerate(random_indices):
        # Cargamos la imagen usando nuestra función de data_utils
        img = load_image(df, index=index, root_dir=root_dir)
        
        if img:
            ax = axes[i]
            ax.imshow(img)
            # Usamos la columna 'class' para el título
            ax.set_title(df.iloc[index]['class'], fontsize=15)
            ax.axis('off')

    # Ocultamos los ejes que no se usen
    for j in range(num_samples, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_full_hierarchical_distribution(df: pd.DataFrame, normalize_y_axis: bool = False):
    """
    Crea una visualización de tres paneles con la distribución jerárquica completa
    y añade una leyenda de colores en los dos paneles inferiores.
    Permite normalizar el eje Y para mostrar porcentajes.
    """
    print("Generando gráfico de distribución jerárquica de 3 niveles...")

    # --- 1. Preparación de Colores y Datos de Nivel 1 (Grupo) ---
    group_counts = df['group'].value_counts(normalize=normalize_y_axis).sort_index()
    palette = plt.get_cmap('tab20')(np.linspace(0, 1, len(group_counts)))
    color_map = {group: color for group, color in zip(group_counts.index, palette)}

    # --- 2. Preparación de Datos de Nivel 2 (Tag) ---
    tag_counts = df['tag'].value_counts(normalize=normalize_y_axis)
    tag_to_group_map = df[['tag', 'group']].drop_duplicates().set_index('tag')['group'].to_dict()
    sorted_tags = sorted(tag_counts.index, key=lambda tag: (tag_to_group_map.get(tag, ''), tag))
    tag_counts_sorted = tag_counts.reindex(sorted_tags)
    tag_colors = [color_map.get(tag_to_group_map.get(tag)) for tag in tag_counts_sorted.index]

    # --- 3. Preparación de Datos de Nivel 3 (Clase) ---
    class_counts_sorted = df['class'].value_counts(normalize=normalize_y_axis).sort_index()
    class_colors = [color_map.get(class_name.split('___')[0]) for class_name in class_counts_sorted.index]
    legend_patches = [mpatches.Patch(color=color, label=group)
                      for group, color in color_map.items()]

    # --- 4. Creación de la Visualización ---
    fig, axes = plt.subplots(3, 1, figsize=(20, 32), gridspec_kw={'height_ratios': [1, 2, 3]})
    fig.suptitle('Análisis de Distribución Jerárquica del Dataset', fontsize=22, y=1.0)

    y_label = 'Proporción de Imágenes (%)' if normalize_y_axis else 'Número de Imágenes'

    # Panel Superior: Distribución por Grupo (Especie)
    sns.barplot(x=group_counts.index, y=group_counts.values * (100 if normalize_y_axis else 1), ax=axes[0], palette=color_map.values())
    axes[0].set_title('Nivel 1: Distribución por Especie de Planta (group)', fontsize=16)
    axes[0].set_ylabel(y_label, fontsize=12)
    axes[0].tick_params(axis='x', rotation=30, labelsize=12)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), ha='right')

    # Panel Medio: Distribución por Tag (Enfermedad General)
    sns.barplot(x=tag_counts_sorted.index, y=tag_counts_sorted.values * (100 if normalize_y_axis else 1), ax=axes[1], palette=tag_colors)
    axes[1].set_title('Nivel 2: Distribución por Enfermedad/Estado (tag)', fontsize=16)
    axes[1].set_ylabel(y_label, fontsize=14)
    axes[1].tick_params(axis='x', rotation=45, labelsize=12)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), ha='right')
    axes[1].legend(handles=legend_patches, title='Especies (Grupo)', loc='best', fontsize='medium')

    # Panel Inferior: Distribución por Clase (Especie + Enfermedad)
    sns.barplot(x=class_counts_sorted.index, y=class_counts_sorted.values * (100 if normalize_y_axis else 1), ax=axes[2], palette=class_colors)
    axes[2].set_title('Nivel 3: Distribución por Clase Específica', fontsize=16)
    axes[2].set_ylabel(y_label, fontsize=14)
    axes[2].tick_params(axis='x', rotation=90, labelsize=12)
    axes[2].legend(handles=legend_patches, title='Especies (Grupo)', loc='best', fontsize='medium')
    
    # Ajustamos el layout para dar espacio a la leyenda
    fig.tight_layout(rect=[0, 0, 0.9, 1]) # El 0.9 en 'right' deja espacio a la derecha
    plt.show()
    
def pie_graph(df: pd.DataFrame,
              crop_col: str,
              class_col: str,
              ncols: int = 3):
    """
    Analiza un DataFrame y genera una grilla de gráficos de torta, uno por cada
    cultivo, mostrando la distribución de sus clases.
    Versión con gráficos más grandes y leyenda en la parte inferior.

    Args:
        df (pd.DataFrame): DataFrame con los datos.
        crop_col (str): Nombre de la columna que identifica al cultivo.
        class_col (str): Nombre de la columna que identifica la clase.
        ncols (int): Número de columnas para la grilla.
    """
    print(f"Generando distribución de '{class_col}' por '{crop_col}'...")

    unique_crops = sorted(df[crop_col].unique())
    num_crops = len(unique_crops)
    nrows = (num_crops + ncols - 1) // ncols
    
    # --- CAMBIO 1: Gráficos más grandes ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 8)) 
    fig.suptitle('Distribución de Clases por Cultivo', fontsize=20, weight='bold')
    axes = axes.flatten()

    for i, crop_name in enumerate(unique_crops):
        ax = axes[i]
        crop_df = df[df[crop_col] == crop_name]
        class_distribution = crop_df[class_col].value_counts()
        legend_labels = [str(label).split('___')[-1].replace('_', ' ') for label in class_distribution.index]

        wedges, _ = ax.pie(
            class_distribution,
            startangle=90,
            colors=plt.cm.Paired.colors
        )

        ax.set_title(str(crop_name).replace("_", " ").title(), fontsize=16)
        ax.axis('equal')

        # --- CAMBIO 2: Leyenda en la parte inferior ---
        ax.legend(wedges, legend_labels,
                  title="Clases",
                  loc="best",
                  bbox_to_anchor=(1, 0, 0.5, 1), 
                  fontsize=12)

    for j in range(num_crops, len(axes)):
        axes[j].axis('off')

    # Ajustamos el layout para evitar solapamientos
    plt.tight_layout() 
    plt.show()

def plot_spectral_signatures(hist_df: pd.DataFrame, title_prefix: str = "Firma Espectral"):
    """
    Grafica las firmas espectrales (histogramas RGB promedio) desde un DataFrame pre-calculado.

    Args:
        hist_df (pd.DataFrame): DataFrame con las columnas ['category', 'bin', 'r', 'g', 'b'].
        title_prefix (str): Prefijo para el título de cada gráfico.
    """
    
    categories = hist_df["category"].unique()
    
    for category in categories:
        plt.figure(figsize=(12, 6))
        
        # Filtramos los datos para la categoría actual
        category_data = hist_df[hist_df["category"] == category]
        
        # Graficamos cada canal de color
        plt.plot(category_data["bin"], category_data["r"], color='red', label='Rojo', alpha=0.8)
        plt.plot(category_data["bin"], category_data["g"], color='green', label='Verde', alpha=0.8)
        plt.plot(category_data["bin"], category_data["b"], color='blue', label='Azul', alpha=0.8)
        
        plt.title(f"{title_prefix} - {category}", fontsize=16)
        plt.xlabel("Intensidad de Píxel (0-255)")
        plt.ylabel("Frecuencia Promedio Normalizada")
        plt.legend()
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()

def plot_distribution(df: pd.DataFrame, column: str, hue_column: str = None, title: str = None, normalize: bool = False, rotation: int = 45):
    """
    Plots the distribution of a specified column, optionally with a hue and normalization.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to plot the distribution for.
        hue_column (str, optional): An optional column to use for hue. Defaults to None.
        title (str, optional): The title of the plot. Defaults to "Distribution of {column}".
        normalize (bool): If True, normalize the counts to percentages. Defaults to False.
        rotation (int): Rotation of x-axis labels. Defaults to 45.
    """
    plt.figure(figsize=(18, 8))
    
    if normalize:
        # Calculate value counts normalized
        counts = df.groupby(hue_column)[column].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
        sns.barplot(data=counts, x=column, y='percentage', hue=hue_column, palette='viridis')
        plt.ylabel('Proporción de Imágenes (%)')
    else:
        sns.countplot(data=df, x=column, hue=hue_column, palette='viridis')
        plt.ylabel('Número de Imágenes')

    plt.xticks(rotation=rotation)
    plt.title(title if title else f'Distribution of {column}' + (f' by {hue_column}' if hue_column else ''))
    plt.xlabel(column)
    plt.tight_layout()
    plt.show()

def plot_all_augmentations(dataframe: pd.DataFrame, image_root_dir: str, 
                           transform_conservative_pipeline: A.Compose, 
                           transform_intensive_pipeline: A.Compose, 
                           load_image_fn: callable, # <-- Recibe la función de carga de imágenes
                           apply_transform_fn: callable, # <-- Recibe la función de aplicación de transformación
                           num_samples: int = 5):
    """
    Visualiza la imagen original, la versión con aumento conservador y la versión con aumento intensivo
    para un conjunto de muestras seleccionadas aleatoriamente.
    
    Args:
        dataframe (pd.DataFrame): DataFrame que contiene las rutas relativas de las imágenes.
        image_root_dir (str): La ruta raíz donde se encuentran las imágenes.
        transform_conservative_pipeline (albumentations.Compose): Pipeline de aumentación conservadora.
        transform_intensive_pipeline (albumentations.Compose): Pipeline de aumentación intensiva.
        load_image_fn (callable): Función para cargar una imagen (ej. load_image_np).
        apply_transform_fn (callable): Función para aplicar una transformación (ej. apply_transformation_direct).
        num_samples (int): Número de muestras a visualizar.
    """
    print(f"\n--- Visualizando Original, Aumentación Conservadora e Intensiva para {num_samples} muestras: ---")
    random_indices = random.sample(range(len(dataframe)), num_samples)

    for i, index in enumerate(random_indices):
        try:
            relative_path = dataframe.loc[index, 'relative_path']
            original_filepath = os.path.join(image_root_dir, relative_path)
            
            # Usar las funciones pasadas como argumento
            original_image_array = load_image_fn(original_filepath)
            
            conservative_augmented_image = apply_transform_fn(original_image_array, transform_conservative_pipeline)
            intensive_augmented_image = apply_transform_fn(original_image_array, transform_intensive_pipeline)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(original_image_array)
            axes[0].set_title('Original')
            axes[0].axis('off')

            axes[1].imshow(conservative_augmented_image)
            axes[1].set_title('Aumentación Conservadora')
            axes[1].axis('off')

            axes[2].imshow(intensive_augmented_image)
            axes[2].set_title('Aumentación Intensiva')
            axes[2].axis('off')

            plt.suptitle(f"Muestra #{i+1} (Índice: {index})")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        except FileNotFoundError:
            print(f"Error: No se encontró la imagen en la ruta: {original_filepath}")
        except Exception as e:
            print(f"Error al cargar/transformar la imagen en el índice {index}: {e}")
