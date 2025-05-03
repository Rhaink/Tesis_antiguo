# Tesis/alineamiento/src/data_splitter.py (o añadir a data_loader.py)

import numpy as np
from sklearn.model_selection import train_test_split
import logging
import os # Necesario si guardamos desde aquí

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_data(num_samples, test_size=0.2, validation_size=0.15, stratify_labels=None, random_state=42):
    """
    Divide los índices en conjuntos de entrenamiento, validación y prueba.

    Args:
        num_samples (int): Número total de muestras (e.g., 800).
        test_size (float): Proporción para el conjunto de prueba.
        validation_size (float): Proporción para el conjunto de validación (del total original).
        stratify_labels (array-like, optional): Etiquetas para estratificación (e.g., category_id).
        random_state (int): Semilla para reproducibilidad.

    Returns:
        dict: Diccionario con {'train': train_indices, 'validation': val_indices, 'test': test_indices}.
              val_indices será None si validation_size es 0.
    """
    indices = np.arange(num_samples)
    
    if stratify_labels is not None and len(stratify_labels) != num_samples:
        logging.error("La longitud de stratify_labels debe coincidir con num_samples.")
        stratify_labels = None # Ignorar estratificación si hay error

    # Primera división: separar prueba
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        stratify=stratify_labels if stratify_labels is not None else None,
        random_state=random_state
    )
    
    val_indices = None
    train_indices = train_val_indices # Por defecto si no hay validación

    # Segunda división: separar validación del resto (si es necesario)
    if validation_size > 0:
        # Calcular proporción de validación respecto al conjunto train_val
        relative_val_size = validation_size / (1.0 - test_size)
        
        # Obtener etiquetas correspondientes a train_val_indices si se estratifica
        stratify_train_val = None
        if stratify_labels is not None:
            try:
                # Asegurarse que stratify_labels sea indexable por train_val_indices
                stratify_labels_array = np.array(stratify_labels)
                stratify_train_val = stratify_labels_array[train_val_indices]
            except IndexError:
                 logging.error("Error al indexar stratify_labels para la segunda división. Se procederá sin estratificación.")
                 stratify_train_val = None
            except Exception as e:
                 logging.error(f"Error inesperado preparando estratificación para la segunda división: {e}. Se procederá sin estratificación.")
                 stratify_train_val = None

        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=relative_val_size,
            stratify=stratify_train_val,
            random_state=random_state 
        )

    logging.info(f"División de datos: {len(train_indices)} entrenamiento, "
                 f"{len(val_indices) if val_indices is not None else 0} validación, "
                 f"{len(test_indices)} prueba.")

    return {
        'train': train_indices,
        'validation': val_indices,
        'test': test_indices
    }

def save_indices(indices_dict, output_dir):
     """Guarda los diccionarios de índices en archivos de texto."""
     os.makedirs(output_dir, exist_ok=True)
     for split_name, indices in indices_dict.items():
         if indices is not None:
             filepath = os.path.join(output_dir, f"{split_name}_indices.txt")
             try:
                 np.savetxt(filepath, indices, fmt='%d')
                 logging.info(f"Índices de {split_name} guardados en {filepath}")
             except Exception as e:
                 logging.error(f"No se pudo guardar {filepath}: {e}")