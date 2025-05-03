# Tesis/alineamiento/src/esl_patch_extractor.py

import os
import numpy as np
import cv2
import logging
import random

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
    ALINEAMIENTO_DIR = os.path.dirname(CURRENT_SCRIPT_DIR)
    BASE_DIR = os.path.dirname(ALINEAMIENTO_DIR)
    RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
except NameError:
    ALINEAMIENTO_DIR = '.'
    BASE_DIR = os.path.dirname(ALINEAMIENTO_DIR)
    RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
    logger.info("Ejecutando en modo interactivo, rutas ajustadas.")

TRAIN_INDICES_PATH = os.path.join(RESULTS_DIR, 'train_indices.txt')
GT_POSE_PATH = os.path.join(RESULTS_DIR, 'esl_ground_truth_pose_train.npz')
OUTPUT_PATCHES_PATH = os.path.join(RESULTS_DIR, 'esl_line_patches_train.npz') # Guardaremos aquí

# --- Parámetros de Extracción ---
PATCH_SIZE = 64            # Tamaño final del parche para la red
NUM_NEG_PER_POS = 3        # Número de negativos por cada positivo
NEGATIVE_MIN_DIST_FACTOR = 0.1 # Factor de distancia mínima para negativos (10% del ancho/alto)
IMAGE_BORDER_MODE = cv2.BORDER_CONSTANT # Cómo manejar bordes al extraer parches
IMAGE_BORDER_VALUE = 0                  # Valor para el borde constante
# --------------------------------

def extract_patch(image, center_x, center_y, patch_size):
    """
    Extrae un parche cuadrado de la imagen centrado en (center_x, center_y),
    lo reescala a (patch_size, patch_size) y maneja los bordes.

    Args:
        image (np.ndarray): Imagen de entrada (escala de grises).
        center_x (float): Coordenada X del centro del parche.
        center_y (float): Coordenada Y del centro del parche.
        patch_size (int): Tamaño deseado del parche final (ancho y alto).

    Returns:
        np.ndarray or None: El parche extraído y reescalado (patch_size, patch_size),
                            o None si hay un error.
    """
    img_h, img_w = image.shape[:2]
    half_size = patch_size / 2.0

    # Calcular coordenadas de la región a extraer en la imagen original
    # (Estas pueden estar fuera de los límites)
    x1 = int(round(center_x - half_size))
    y1 = int(round(center_y - half_size))
    x2 = x1 + patch_size # Coordenada final + 1
    y2 = y1 + patch_size # Coordenada final + 1

    # Calcular cuánto rellenar (padding) si el parche se sale
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - img_w)
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - img_h)

    # Extraer la parte del parche que SÍ está dentro de la imagen
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(img_w, x2)
    crop_y2 = min(img_h, y2)

    try:
        # Extraer la porción válida de la imagen
        if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
             # Si la región válida es de tamaño cero, devolver parche negro
             patch_cropped = np.full((patch_size, patch_size), IMAGE_BORDER_VALUE, dtype=image.dtype)
             logger.debug(f"Parche completamente fuera de la imagen en ({center_x:.1f}, {center_y:.1f}). Devolviendo parche negro.")
             return patch_cropped
        else:
             patch_cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

             # Añadir padding si es necesario para obtener tamaño patch_size x patch_size
             if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                  patch_padded = cv2.copyMakeBorder(patch_cropped, pad_top, pad_bottom, pad_left, pad_right,
                                                    IMAGE_BORDER_MODE, value=IMAGE_BORDER_VALUE)
             else:
                  patch_padded = patch_cropped

             # Verificar tamaño final antes de reescalar (debería ser patch_size x patch_size)
             # Nota: Debido a redondeos, podría ser patch_size +/- 1. Reescalar lo arregla.
             if patch_padded.shape[0] != patch_size or patch_padded.shape[1] != patch_size:
                  # Reescalar al tamaño exacto (esto maneja pequeños errores de redondeo)
                  final_patch = cv2.resize(patch_padded, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
             else:
                  final_patch = patch_padded

             # Normalizar parche a [0, 1] (o podrías hacer Z-score aquí también)
             final_patch = final_patch.astype(np.float32)
             min_val, max_val = np.min(final_patch), np.max(final_patch)
             if max_val > min_val:
                 final_patch = (final_patch - min_val) / (max_val - min_val)
             else: # Evitar división por cero si el parche es constante
                 final_patch = np.zeros_like(final_patch) # O el valor constante normalizado

             return final_patch

    except Exception as e:
        logger.error(f"Error extrayendo parche en ({center_x:.1f}, {center_y:.1f}): {e}", exc_info=True)
        return None # Indicar error

def main():
    logger.info(f"--- Iniciando Extracción de Parches para Clasificadores de Líneas ESL (Entrenamiento) ---")
    logger.info(f"Tamaño del parche final: {PATCH_SIZE}x{PATCH_SIZE}")
    logger.info(f"Negativos por positivo: {NUM_NEG_PER_POS}")
    logger.info(f"Factor distancia mínima negativo: {NEGATIVE_MIN_DIST_FACTOR}")

    # 1. Cargar datos generales y GT pose
    logger.info(f"Cargando datos generales desde directorio base: {BASE_DIR}")
    index_map, _, image_paths_dict = load_all_data(BASE_DIR) # Solo necesitamos paths e index_map
    if index_map is None or image_paths_dict is None:
        logger.error("Fallo al cargar datos iniciales. Abortando.")
        return

    logger.info(f"Cargando parámetros GT desde: {GT_POSE_PATH}")
    if not os.path.exists(GT_POSE_PATH):
        logger.error(f"Archivo GT no encontrado: {GT_POSE_PATH}. Ejecute esl_gt_generator.py primero.")
        return
    try:
        gt_data = np.load(GT_POSE_PATH)
        # Crear un mapeo de índice original a su posición en los arrays GT
        gt_indices = gt_data['index']
        index_to_gt_pos = {idx: pos for pos, idx in enumerate(gt_indices)}
        num_gt_samples = len(gt_indices)
        logger.info(f"Parámetros GT cargados para {num_gt_samples} muestras.")
    except Exception as e:
        logger.error(f"Error cargando archivo GT: {e}")
        return

    # 2. Inicializar listas para parches y etiquetas
    data_l1 = {'patches': [], 'labels': []}
    data_l2 = {'patches': [], 'labels': []}
    data_l3 = {'patches': [], 'labels': []}
    data_l4 = {'patches': [], 'labels': []}
    all_data_lists = [data_l1, data_l2, data_l3, data_l4]

    processed_count = 0
    error_count = 0

    # 3. Iterar sobre las muestras con GT disponible
    for original_idx in gt_indices:
        if original_idx not in image_paths_dict:
            logger.warning(f"No se encontró ruta de imagen para índice {original_idx}. Saltando.")
            error_count += 1
            continue

        image_path = image_paths_dict[original_idx]
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning(f"No se pudo cargar la imagen: {image_path}. Saltando.")
                error_count += 1
                continue
            H, W = image.shape
            
            # Obtener posición de este índice en los arrays GT
            gt_pos = index_to_gt_pos[original_idx]

            # Obtener coordenadas GT (64x64) y escala (64x64)
            l1_gt = gt_data['l1'][gt_pos]
            l2_gt = gt_data['l2'][gt_pos]
            l3_gt = gt_data['l3'][gt_pos]
            l4_gt = gt_data['l4'][gt_pos]
            S_width_gt = gt_data['S_width'][gt_pos]
            S_height_gt = gt_data['S_height'][gt_pos]

            # Escalar a dimensiones de la imagen
            scale_x = W / 64.0
            scale_y = H / 64.0
            L1 = l1_gt * scale_x
            L2 = l2_gt * scale_y
            L3 = l3_gt * scale_x
            L4 = l4_gt * scale_y
            S_W_img = S_width_gt * scale_x
            S_H_img = S_height_gt * scale_y

            # Definir centros y distancias mínimas para negativos
            center_y_vert = (L2 + L4) / 2.0 # Centro Y para líneas verticales L1, L3
            center_x_horz = (L1 + L3) / 2.0 # Centro X para líneas horizontales L2, L4
            min_dist_x = NEGATIVE_MIN_DIST_FACTOR * S_W_img
            min_dist_y = NEGATIVE_MIN_DIST_FACTOR * S_H_img

            # --- Procesar Línea 1 (Vertical Izquierda) ---
            # Positivo
            patch_pos = extract_patch(image, L1, center_y_vert, PATCH_SIZE)
            if patch_pos is not None:
                data_l1['patches'].append(patch_pos)
                data_l1['labels'].append(1)
                # Negativos
                for _ in range(NUM_NEG_PER_POS):
                    cx_neg = random.uniform(0, W)
                    # Asegurar que esté suficientemente lejos
                    while abs(cx_neg - L1) <= min_dist_x:
                        cx_neg = random.uniform(0, W)
                    patch_neg = extract_patch(image, cx_neg, center_y_vert, PATCH_SIZE)
                    if patch_neg is not None:
                        data_l1['patches'].append(patch_neg)
                        data_l1['labels'].append(0)
            else: error_count += 1

            # --- Procesar Línea 2 (Horizontal Superior) ---
            patch_pos = extract_patch(image, center_x_horz, L2, PATCH_SIZE)
            if patch_pos is not None:
                data_l2['patches'].append(patch_pos)
                data_l2['labels'].append(1)
                # Negativos
                for _ in range(NUM_NEG_PER_POS):
                    cy_neg = random.uniform(0, H)
                    while abs(cy_neg - L2) <= min_dist_y:
                         cy_neg = random.uniform(0, H)
                    patch_neg = extract_patch(image, center_x_horz, cy_neg, PATCH_SIZE)
                    if patch_neg is not None:
                         data_l2['patches'].append(patch_neg)
                         data_l2['labels'].append(0)
            else: error_count += 1

            # --- Procesar Línea 3 (Vertical Derecha) ---
            patch_pos = extract_patch(image, L3, center_y_vert, PATCH_SIZE)
            if patch_pos is not None:
                data_l3['patches'].append(patch_pos)
                data_l3['labels'].append(1)
                # Negativos
                for _ in range(NUM_NEG_PER_POS):
                    cx_neg = random.uniform(0, W)
                    while abs(cx_neg - L3) <= min_dist_x:
                        cx_neg = random.uniform(0, W)
                    patch_neg = extract_patch(image, cx_neg, center_y_vert, PATCH_SIZE)
                    if patch_neg is not None:
                        data_l3['patches'].append(patch_neg)
                        data_l3['labels'].append(0)
            else: error_count += 1
            
            # --- Procesar Línea 4 (Horizontal Inferior) ---
            patch_pos = extract_patch(image, center_x_horz, L4, PATCH_SIZE)
            if patch_pos is not None:
                data_l4['patches'].append(patch_pos)
                data_l4['labels'].append(1)
                # Negativos
                for _ in range(NUM_NEG_PER_POS):
                    cy_neg = random.uniform(0, H)
                    while abs(cy_neg - L4) <= min_dist_y:
                         cy_neg = random.uniform(0, H)
                    patch_neg = extract_patch(image, center_x_horz, cy_neg, PATCH_SIZE)
                    if patch_neg is not None:
                         data_l4['patches'].append(patch_neg)
                         data_l4['labels'].append(0)
            else: error_count += 1

            processed_count += 1
            if processed_count % 50 == 0:
                logger.info(f"Procesadas {processed_count}/{num_gt_samples} imágenes...")

        except Exception as e:
            logger.error(f"Error procesando índice {original_idx} ({image_path}): {e}", exc_info=True)
            error_count += 1

    logger.info(f"Procesamiento de imágenes finalizado. {processed_count} procesadas, {error_count} errores/omisiones.")

    # 4. Convertir listas a arrays y guardar
    output_data = {}
    total_patches = 0
    try:
        for i, data_list in enumerate(all_data_lists):
            line_num = i + 1
            patches_array = np.array(data_list['patches'], dtype=np.float32)
            # Añadir dimensión de canal (para CNNs que esperan H, W, C)
            if patches_array.ndim == 3: # Si son (N, H, W)
                 patches_array = patches_array[..., np.newaxis] # Convertir a (N, H, W, 1)
            labels_array = np.array(data_list['labels'], dtype=np.int32)

            if patches_array.shape[0] != labels_array.shape[0]:
                 logger.error(f"Discrepancia en número de parches y etiquetas para L{line_num}!")
                 continue # No guardar si hay error

            logger.info(f"Línea {line_num}: {patches_array.shape[0]} parches, shape={patches_array.shape}")
            output_data[f'patches_l{line_num}'] = patches_array
            output_data[f'labels_l{line_num}'] = labels_array
            total_patches += patches_array.shape[0]

        if output_data:
            logger.info(f"Total de parches generados: {total_patches}")
            logger.info(f"Guardando datos de parches en: {OUTPUT_PATCHES_PATH}")
            np.savez_compressed(OUTPUT_PATCHES_PATH, **output_data) # Usar compresión
            logger.info("Datos guardados exitosamente.")
        else:
            logger.error("No se generaron datos válidos para guardar.")

    except Exception as e:
        logger.error(f"Error convirtiendo listas a arrays o guardando archivo: {e}", exc_info=True)

    logger.info(f"--- Extracción de Parches Finalizada ---")


if __name__ == "__main__":
    main()