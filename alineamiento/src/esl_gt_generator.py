# Tesis/alineamiento/src/esl_gt_generator.py

import os
import numpy as np
import pandas as pd
import logging

# Importar cargador de datos
try:
    from data_loader import load_all_data
except ImportError:
    print("Error: Asegúrate de que data_loader.py esté accesible.")
    exit(1)

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración de Rutas ---
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ALINEAMIENTO_DIR = os.path.dirname(CURRENT_SCRIPT_DIR) # Tesis/alineamiento
    BASE_DIR = os.path.dirname(ALINEAMIENTO_DIR) # Directorio 'Tesis'
    RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
except NameError:
    # Para ejecución interactiva
    ALINEAMIENTO_DIR = '.'
    BASE_DIR = os.path.dirname(ALINEAMIENTO_DIR)
    RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
    logger.info("Ejecutando en modo interactivo, rutas ajustadas.")

TRAIN_INDICES_PATH = os.path.join(RESULTS_DIR, 'train_indices.txt')
OUTPUT_GT_POSE_PATH = os.path.join(RESULTS_DIR, 'esl_ground_truth_pose_train.npz') # Guardaremos aquí

# --- Constantes ---
NUM_LANDMARKS = 15
NUM_DIMS = 2
# ------------------

def calculate_aabb_params(landmarks):
    """
    Calcula los parámetros de la caja delimitadora alineada a ejes (AABB)
    y la pose (T, S, theta=0) a partir de un conjunto de landmarks.

    Args:
        landmarks (np.ndarray): Array de landmarks (k, d), e.g., (15, 2).

    Returns:
        dict: Diccionario con 'l1', 'l2', 'l3', 'l4', 'theta', 'Tx', 'Ty', 'S_width', 'S_height'.
              Retorna None si los landmarks son inválidos.
    """
    if landmarks is None or landmarks.shape != (NUM_LANDMARKS, NUM_DIMS):
        logger.warning(f"Landmarks inválidos proporcionados. Shape: {landmarks.shape if landmarks is not None else 'None'}")
        return None
    if np.isnan(landmarks).any() or np.isinf(landmarks).any():
        logger.warning("Landmarks contienen NaN/Inf.")
        return None

    try:
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]

        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)

        # Líneas (representadas por sus coordenadas)
        l1_gt = x_min
        l3_gt = x_max
        l2_gt = y_min
        l4_gt = y_max

        # Orientación (fija para AABB)
        theta_gt = 0.0

        # Centro
        Tx_gt = (x_min + x_max) / 2.0
        Ty_gt = (y_min + y_max) / 2.0

        # Escala (ancho y alto)
        S_width_gt = x_max - x_min
        S_height_gt = y_max - y_min
        
        # Validar que la escala no sea cero o negativa (puntos colineales/coincidentes)
        if S_width_gt < 1e-6 or S_height_gt < 1e-6:
             logger.warning("Ancho o alto de la caja delimitadora casi cero. Puede indicar landmarks degenerados.")
             # Podríamos devolver None o continuar con valores pequeños/cero

        return {
            'l1': l1_gt, 'l2': l2_gt, 'l3': l3_gt, 'l4': l4_gt,
            'theta': theta_gt,
            'Tx': Tx_gt, 'Ty': Ty_gt,
            'S_width': S_width_gt, 'S_height': S_height_gt
        }

    except Exception as e:
        logger.error(f"Error calculando parámetros AABB: {e}")
        return None

def main():
    logger.info("--- Iniciando Generación de Parámetros de Localización Ground Truth para ESL (Entrenamiento) ---")
    
    # 1. Cargar datos generales (landmarks 64x64)
    logger.info(f"Cargando datos desde directorio base: {BASE_DIR}")
    index_map, landmarks_array_orig, _ = load_all_data(BASE_DIR) # No necesitamos paths aquí
    if index_map is None or landmarks_array_orig is None:
        logger.error("Fallo al cargar datos iniciales. Abortando.")
        return
        
    # Validar dimensiones de landmarks
    if landmarks_array_orig.ndim != 3 or landmarks_array_orig.shape[1:] != (NUM_LANDMARKS, NUM_DIMS):
         logger.error(f"Dimensiones inesperadas para landmarks_array_orig: {landmarks_array_orig.shape}")
         return
    num_total_samples = landmarks_array_orig.shape[0]
    logger.info(f"Landmarks originales (64x64) cargados para {num_total_samples} muestras.")

    # 2. Cargar índices de entrenamiento
    if not os.path.exists(TRAIN_INDICES_PATH):
        logger.error(f"Archivo de índices de entrenamiento no encontrado: {TRAIN_INDICES_PATH}")
        logger.error("Ejecute 'prepare_splits.py' primero.")
        return
    try:
        train_indices = np.loadtxt(TRAIN_INDICES_PATH, dtype=int)
        if train_indices.ndim == 0: train_indices = np.array([train_indices.item()])
        
        # Validar índices contra el número total de muestras
        valid_train_indices = [idx for idx in train_indices if 0 <= idx < num_total_samples]
        if len(valid_train_indices) < len(train_indices):
            logger.warning(f"Descartados {len(train_indices) - len(valid_train_indices)} índices de entrenamiento fuera de rango [0, {num_total_samples-1}].")
        if not valid_train_indices:
            logger.error("No quedan índices de entrenamiento válidos.")
            return
        train_indices = np.array(valid_train_indices)
        logger.info(f"Se procesarán {len(train_indices)} muestras de entrenamiento.")

    except Exception as e:
        logger.error(f"Error cargando o validando índices de entrenamiento: {e}")
        return

    # 3. Calcular y almacenar parámetros GT para cada muestra de entrenamiento
    gt_params_list = []
    processed_indices = [] # Guardar los índices para los que se calcularon parámetros

    for idx in train_indices:
        landmarks_64 = landmarks_array_orig[idx]
        params = calculate_aabb_params(landmarks_64)
        if params is not None:
            gt_params_list.append(params)
            processed_indices.append(idx) # Guardar el índice original
        else:
             logger.warning(f"No se pudieron calcular los parámetros GT para el índice {idx}. Se omitirá.")

    if not gt_params_list:
        logger.error("No se pudieron calcular parámetros GT para ninguna muestra de entrenamiento.")
        return

    # Convertir lista de diccionarios a un diccionario de arrays NumPy para guardar
    # Esto es más eficiente para np.savez
    gt_params_dict = {}
    # Obtener las claves del primer diccionario (asumiendo que todas son iguales)
    keys = gt_params_list[0].keys()
    for key in keys:
        gt_params_dict[key] = np.array([d[key] for d in gt_params_list])
        
    # Añadir los índices procesados al diccionario para referencia
    gt_params_dict['index'] = np.array(processed_indices)

    logger.info(f"Parámetros GT calculados para {len(processed_indices)} muestras.")

    # 4. Guardar diccionario en archivo .npz
    try:
        np.savez(OUTPUT_GT_POSE_PATH, **gt_params_dict)
        logger.info(f"Parámetros Ground Truth para ESL (entrenamiento) guardados en: {OUTPUT_GT_POSE_PATH}")
    except Exception as e:
        logger.error(f"Error al guardar el archivo NPZ en {OUTPUT_GT_POSE_PATH}: {e}")

    logger.info("--- Generación de Parámetros de Localización Ground Truth Finalizada ---")

if __name__ == "__main__":
    main()