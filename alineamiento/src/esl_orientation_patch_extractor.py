# Tesis/alineamiento/src/esl_orientation_patch_extractor.py

import os
import numpy as np
import cv2
import logging
import math

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
OUTPUT_ORIENTATION_PATCHES_PATH = os.path.join(RESULTS_DIR, 'esl_orientation_patches_train.npz')

# --- Parámetros ---
PATCH_SIZE = 64 # Tamaño final del parche para la red (igual que líneas)
ROTATION_ANGLES_DEG = np.arange(-30, 31, 5) # Ángulos en grados: -30, -25, ..., 25, 30
ANGLE_THRESHOLD_DEG = 5.0 # Umbral para etiqueta positiva
BASE_PATCH_PADDING_FACTOR = 1.1 # Factor para agrandar parche base antes de rotar
IMAGE_BORDER_MODE = cv2.BORDER_CONSTANT
IMAGE_BORDER_VALUE = 0
# ------------------

def extract_patch_base(image, center_x, center_y, base_size):
    """
    Extrae un parche cuadrado de la imagen centrado en (center_x, center_y)
    con tamaño base_size x base_size, manejando bordes. NO reescala.

    Args:
        image (np.ndarray): Imagen de entrada (escala de grises).
        center_x (float): Coordenada X del centro del parche.
        center_y (float): Coordenada Y del centro del parche.
        base_size (int): Tamaño del parche cuadrado a extraer.

    Returns:
        np.ndarray or None: El parche extraído (base_size, base_size), o None.
    """
    img_h, img_w = image.shape[:2]
    half_size = base_size / 2.0

    x1 = int(round(center_x - half_size))
    y1 = int(round(center_y - half_size))
    x2 = x1 + base_size
    y2 = y1 + base_size

    pad_left = max(0, -x1)
    pad_right = max(0, x2 - img_w)
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - img_h)

    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(img_w, x2)
    crop_y2 = min(img_h, y2)

    try:
        if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
             patch_cropped = np.full((base_size, base_size), IMAGE_BORDER_VALUE, dtype=image.dtype)
             logger.debug(f"Parche base completamente fuera en ({center_x:.1f}, {center_y:.1f}).")
             return patch_cropped
        else:
             patch_cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
             if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                  patch_padded = cv2.copyMakeBorder(patch_cropped, pad_top, pad_bottom, pad_left, pad_right,
                                                    IMAGE_BORDER_MODE, value=IMAGE_BORDER_VALUE)
             else:
                  patch_padded = patch_cropped
             # Asegurar que tenga el tamaño base_size exacto (puede faltar por redondeo)
             if patch_padded.shape[0] != base_size or patch_padded.shape[1] != base_size:
                   patch_padded = cv2.resize(patch_padded, (base_size, base_size), interpolation=cv2.INTER_LINEAR)
             return patch_padded
    except Exception as e:
        logger.error(f"Error extrayendo parche base en ({center_x:.1f}, {center_y:.1f}): {e}")
        return None

def normalize_patch(patch):
    """Normaliza un parche a [0, 1]."""
    patch_float = patch.astype(np.float32)
    min_val, max_val = np.min(patch_float), np.max(patch_float)
    if max_val > min_val:
        normalized = (patch_float - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(patch_float)
    return normalized

def main():
    logger.info(f"--- Iniciando Extracción de Parches para Clasificador de Orientación ESL (Entrenamiento) ---")
    logger.info(f"Ángulos a probar: {ROTATION_ANGLES_DEG} grados")
    logger.info(f"Umbral positivo: +/- {ANGLE_THRESHOLD_DEG} grados")
    logger.info(f"Tamaño final del parche: {PATCH_SIZE}x{PATCH_SIZE}")

    # 1. Cargar datos generales y GT pose
    logger.info(f"Cargando datos generales desde directorio base: {BASE_DIR}")
    index_map, _, image_paths_dict = load_all_data(BASE_DIR)
    if index_map is None or image_paths_dict is None:
        logger.error("Fallo al cargar datos iniciales. Abortando.")
        return

    logger.info(f"Cargando parámetros GT desde: {GT_POSE_PATH}")
    if not os.path.exists(GT_POSE_PATH):
        logger.error(f"Archivo GT no encontrado: {GT_POSE_PATH}.")
        return
    try:
        gt_data = np.load(GT_POSE_PATH)
        gt_indices = gt_data['index']
        index_to_gt_pos = {idx: pos for pos, idx in enumerate(gt_indices)}
        num_gt_samples = len(gt_indices)
        logger.info(f"Parámetros GT cargados para {num_gt_samples} muestras.")
    except Exception as e:
        logger.error(f"Error cargando archivo GT: {e}")
        return

    # 2. Inicializar listas para parches y etiquetas
    patches_theta = []
    labels_theta = []
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

            gt_pos = index_to_gt_pos[original_idx]

            # Obtener T y S (coords 64x64)
            Tx_gt = gt_data['Tx'][gt_pos]
            Ty_gt = gt_data['Ty'][gt_pos]
            S_width_gt = gt_data['S_width'][gt_pos]
            S_height_gt = gt_data['S_height'][gt_pos]

            # Escalar T y S a dimensiones de la imagen
            scale_x = W / 64.0
            scale_y = H / 64.0
            Tx_img = Tx_gt * scale_x
            Ty_img = Ty_gt * scale_y
            S_W_img = S_width_gt * scale_x
            S_H_img = S_height_gt * scale_y

            # Calcular tamaño del parche base (un poco más grande que la diagonal)
            diagonal = math.sqrt(S_W_img**2 + S_H_img**2)
            base_patch_size = int(math.ceil(diagonal * BASE_PATCH_PADDING_FACTOR))
            # Asegurarse que el tamaño sea par si es necesario para el centro
            # if base_patch_size % 2 != 0: base_patch_size += 1
            if base_patch_size < PATCH_SIZE: # Asegurar tamaño mínimo
                 base_patch_size = PATCH_SIZE

            # Extraer parche base
            patch_base = extract_patch_base(image, Tx_img, Ty_img, base_patch_size)
            if patch_base is None:
                logger.warning(f"No se pudo extraer parche base para índice {original_idx}. Saltando imagen.")
                error_count += 1
                continue

            # Centro para la rotación
            center = (base_patch_size // 2, base_patch_size // 2)

            # Generar parches rotados
            for angle in ROTATION_ANGLES_DEG:
                # Calcular matriz de rotación
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                # Aplicar rotación
                rotated_base = cv2.warpAffine(patch_base, M, (base_patch_size, base_patch_size),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=IMAGE_BORDER_MODE,
                                              borderValue=IMAGE_BORDER_VALUE)

                # Reescalar al tamaño final
                final_patch = cv2.resize(rotated_base, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LINEAR)

                # Normalizar
                normalized_patch = normalize_patch(final_patch)

                # Determinar etiqueta
                label = 1 if abs(angle) <= ANGLE_THRESHOLD_DEG else 0

                patches_theta.append(normalized_patch)
                labels_theta.append(label)

            processed_count += 1
            if processed_count % 50 == 0:
                logger.info(f"Procesadas {processed_count}/{num_gt_samples} imágenes...")

        except Exception as e:
            logger.error(f"Error procesando índice {original_idx} ({image_path}): {e}", exc_info=True)
            error_count += 1

    logger.info(f"Procesamiento de imágenes finalizado. {processed_count} procesadas, {error_count} errores.")

    # 4. Convertir listas a arrays y guardar
    try:
        patches_array = np.array(patches_theta, dtype=np.float32)
        # Añadir dimensión de canal
        if patches_array.ndim == 3:
            patches_array = patches_array[..., np.newaxis]
        labels_array = np.array(labels_theta, dtype=np.int32)

        if patches_array.shape[0] != labels_array.shape[0]:
             raise ValueError("Discrepancia en número de parches y etiquetas para orientación!")

        logger.info(f"Total de parches de orientación generados: {patches_array.shape[0]}, shape={patches_array.shape}")
        logger.info(f"Distribución de etiquetas: {np.bincount(labels_array)}")

        output_data = {
            'patches_theta': patches_array,
            'labels_theta': labels_array
        }

        logger.info(f"Guardando datos de parches de orientación en: {OUTPUT_ORIENTATION_PATCHES_PATH}")
        np.savez_compressed(OUTPUT_ORIENTATION_PATCHES_PATH, **output_data)
        logger.info("Datos guardados exitosamente.")

    except Exception as e:
        logger.error(f"Error convirtiendo listas a arrays o guardando archivo: {e}", exc_info=True)

    logger.info(f"--- Extracción de Parches de Orientación Finalizada ---")

if __name__ == "__main__":
    main()