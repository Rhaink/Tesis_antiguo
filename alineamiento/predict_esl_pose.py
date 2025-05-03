# Tesis/alineamiento/predict_esl_pose.py
# CORREGIDO v2: Funciones de data_loader y extracción copiadas directamente

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import logging
import time
import math
import pandas as pd # Necesario para data_loader

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración de Rutas y Parámetros ---
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Asume que este script está en alineamiento/src/ o alineamiento/
    if os.path.basename(CURRENT_SCRIPT_DIR) == 'src':
        ALINEAMIENTO_DIR = os.path.dirname(CURRENT_SCRIPT_DIR)
    else: # Asume que está en alineamiento/
        ALINEAMIENTO_DIR = CURRENT_SCRIPT_DIR

    MODELS_DIR = os.path.join(ALINEAMIENTO_DIR, "models")
    BASE_DIR = os.path.dirname(ALINEAMIENTO_DIR) # Directorio 'Tesis'
    RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
    PLOTS_DIR = os.path.join(ALINEAMIENTO_DIR, "plots")
except NameError:
    ALINEAMIENTO_DIR = '.'
    MODELS_DIR = os.path.join(ALINEAMIENTO_DIR, "models")
    BASE_DIR = os.path.dirname(ALINEAMIENTO_DIR)
    if not BASE_DIR: BASE_DIR = '..' # Ajuste si alineamiento es '.'
    RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
    PLOTS_DIR = os.path.join(ALINEAMIENTO_DIR, "plots")
    logger.info("Ejecutando en modo interactivo, rutas ajustadas.")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True) # Asegurar que exista

MODEL_NAMES = {
    'l1': 'esl_classifier_l1_best.h5',
    'l2': 'esl_classifier_l2_best.h5',
    'l3': 'esl_classifier_l3_best.h5',
    'l4': 'esl_classifier_l4_best.h5',
    'theta': 'esl_classifier_theta_best.h5'
}

# Constantes copiadas/adaptadas
PATCH_SIZE = 64
ROTATION_ANGLES_DEG = np.arange(-30, 31, 5)
BASE_PATCH_PADDING_FACTOR = 1.1
IMAGE_BORDER_MODE = cv2.BORDER_CONSTANT
IMAGE_BORDER_VALUE = 0
SEARCH_RANGE_FACTOR = 0.8
SEARCH_STEP = 2
BATCH_SIZE = 64
NUM_LANDMARKS = 15 # Necesario para data_loader
NUM_DIMS = 2       # Necesario para data_loader
DEFAULT_INDICES_FILE = os.path.join(BASE_DIR, "indices", "indices_maestro_1.csv")
DEFAULT_COORDS_FILE = os.path.join(BASE_DIR, "coordenadas", "coordenadas_maestro_1.csv")
DEFAULT_DATASET_DIR = os.path.join(BASE_DIR, "COVID-19_Radiography_Dataset")
# -----------------------------------------

# === FUNCIONES COPIADAS DE data_loader.py ===

def load_index_map(index_file_path=DEFAULT_INDICES_FILE):
    # (Código de load_index_map de data_loader.py)
    logging.info(f"Cargando mapa de índices desde: {index_file_path}")
    try:
        index_map = pd.read_csv(index_file_path, header=None,
                                names=['new_index', 'category_id', 'original_id'],
                                dtype={0: int, 1: int, 2: object})
        index_map['original_id'] = pd.to_numeric(index_map['original_id'], errors='coerce')
        initial_count = len(index_map)
        index_map.dropna(subset=['original_id'], inplace=True)
        final_count = len(index_map)
        if final_count < initial_count:
            logging.warning(f"Se eliminaron {initial_count - final_count} filas del archivo de índices debido a IDs no numéricos.")
        index_map['original_id'] = index_map['original_id'].astype(int)
        if not (index_map['category_id'].isin([1, 2, 3]).all()):
             logging.warning("Se encontraron category_id fuera del rango esperado [1, 2, 3].")
        logging.info(f"Mapa de índices cargado exitosamente con {len(index_map)} entradas válidas.")
        return index_map.set_index('new_index')
    except FileNotFoundError:
        logging.error(f"Error CRÍTICO: No se encontró el archivo de índices '{index_file_path}'.")
        return None
    except Exception as e:
        logging.error(f"Error al leer o procesar el archivo de índices '{index_file_path}': {e}")
        return None

def load_landmarks(coords_file_path=DEFAULT_COORDS_FILE, num_landmarks=NUM_LANDMARKS):
    # (Código de load_landmarks de data_loader.py)
    logging.info(f"Cargando landmarks desde: {coords_file_path} (asumiendo {num_landmarks} landmarks, 64x64)")
    num_coord_cols = num_landmarks * 2
    expected_cols = 1 + num_coord_cols + 1
    try:
        coords_df = pd.read_csv(coords_file_path, header=None)
        if coords_df.shape[1] != expected_cols:
            logging.error(f"Número inesperado de columnas en {coords_file_path}. Esperado: {expected_cols}, Encontrado: {coords_df.shape[1]}")
            return None, None
        landmark_coords = coords_df.iloc[:, 1:1+num_coord_cols].values
        if not np.issubdtype(landmark_coords.dtype, np.number):
             logging.warning("Las coordenadas no parecen ser numéricas. Intentando convertir...")
             landmark_coords = landmark_coords.astype(float)
        num_images = landmark_coords.shape[0]
        landmarks_array = landmark_coords.reshape(num_images, num_landmarks, 2)
        labels = coords_df.iloc[:, -1].tolist()
        logging.info(f"Landmarks cargados para {num_images} imágenes. Shape: {landmarks_array.shape}")
        # if np.any(landmarks_array < 0) or np.any(landmarks_array >= 64): # Asumiendo 64x64
        #     logging.warning("Algunas coordenadas de landmarks están fuera del rango esperado [0, 64).")
        return landmarks_array, labels
    except FileNotFoundError:
        logging.error(f"Error CRÍTICO: No se encontró el archivo de coordenadas '{coords_file_path}'.")
        return None, None
    except Exception as e:
        logging.error(f"Error al leer o procesar el archivo de coordenadas '{coords_file_path}': {e}")
        return None, None

def get_image_paths(index_map, dataset_base_dir=DEFAULT_DATASET_DIR):
    # (Código de get_image_paths de data_loader.py)
    logging.info(f"Construyendo rutas de imágenes desde: {dataset_base_dir}")
    image_paths = {}
    missing_files = 0
    if index_map is None or not isinstance(index_map, pd.DataFrame):
        logging.error("Mapa de índices inválido proporcionado a get_image_paths.")
        return {}
    for new_idx in index_map.index:
        try:
            row = index_map.loc[new_idx]
            categoria = int(row['category_id'])
            img_id = int(row['original_id'])
            path = None
            sub_dir = ""
            filename = ""
            if categoria == 1: sub_dir = "COVID/images"; filename = f"COVID-{img_id}.png"
            elif categoria == 2: sub_dir = "Normal/images"; filename = f"Normal-{img_id}.png"
            elif categoria == 3: sub_dir = "Viral Pneumonia/images"; filename = f"Viral Pneumonia-{img_id}.png"
            else: logging.warning(f"Categoría desconocida '{categoria}' para idx {new_idx}."); continue
            path = os.path.join(dataset_base_dir, sub_dir, filename)
            if os.path.exists(path): image_paths[new_idx] = path
            else: logging.warning(f"Archivo no encontrado: {path}."); missing_files += 1
        except KeyError: logging.error(f"Error procesando idx {new_idx}. Faltan columnas?"); continue
        except Exception as e: logging.error(f"Error inesperado procesando idx {new_idx}: {e}"); continue
    logging.info(f"Se encontraron {len(image_paths)} rutas válidas.")
    if missing_files > 0: logging.warning(f"No se encontraron {missing_files} archivos.")
    return image_paths

def load_all_data(base_dir_load=BASE_DIR): # Permitir pasar directorio base
    # (Código de load_all_data de data_loader.py)
    # Usa rutas relativas a base_dir_load
    indices_file = os.path.join(base_dir_load, "indices", "indices_maestro_1.csv")
    coords_file = os.path.join(base_dir_load, "coordenadas", "coordenadas_maestro_1.csv")
    dataset_dir = os.path.join(base_dir_load, "COVID-19_Radiography_Dataset")

    index_map = load_index_map(indices_file)
    if index_map is None: return None, None, None
    landmarks_array, _ = load_landmarks(coords_file)
    if landmarks_array is None: return None, None, None
    if len(index_map) != landmarks_array.shape[0]:
        logging.warning(f"Discrepancia muestras: {len(index_map)} índices vs {landmarks_array.shape[0]} coords.")
    image_paths = get_image_paths(index_map, dataset_dir)
    if len(image_paths) < len(index_map):
         logging.warning(f"No todas las entradas del índice tienen imagen válida ({len(image_paths)} vs {len(index_map)}).")
    return index_map, landmarks_array, image_paths

# === FIN FUNCIONES COPIADAS DE data_loader.py ===


# === FUNCIONES COPIADAS DE scripts anteriores ===

def extract_patch(image, center_x, center_y, patch_size):
    """
    Extrae un parche cuadrado de la imagen centrado en (center_x, center_y),
    lo reescala a (patch_size, patch_size) y maneja los bordes.
    ¡ASEGÚRATE QUE LA IMAGEN DE ENTRADA YA ESTÉ NORMALIZADA SI ES NECESARIO!
    """
    img_h, img_w = image.shape[:2]
    half_size = patch_size / 2.0
    x1 = int(round(center_x - half_size))
    y1 = int(round(center_y - half_size))
    x2 = x1 + patch_size
    y2 = y1 + patch_size

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
             patch_cropped = np.full((patch_size, patch_size), float(IMAGE_BORDER_VALUE), dtype=np.float32)
             return patch_cropped
        else:
             # Asume que 'image' ya es float32 si fue pre-normalizada
             patch_cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
             if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                  patch_padded = cv2.copyMakeBorder(patch_cropped, pad_top, pad_bottom, pad_left, pad_right,
                                                    IMAGE_BORDER_MODE, value=float(IMAGE_BORDER_VALUE))
             else:
                  patch_padded = patch_cropped

             if patch_padded.shape[0] != patch_size or patch_padded.shape[1] != patch_size:
                  final_patch = cv2.resize(patch_padded, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
             else:
                  final_patch = patch_padded
             return final_patch.astype(np.float32)

    except Exception as e:
        logger.error(f"Error extrayendo parche en ({center_x:.1f}, {center_y:.1f}): {e}", exc_info=False)
        return None

def extract_patch_base(image, center_x, center_y, base_size):
    """ Extrae parche base sin reescalar ni normalizar. """
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
             return patch_cropped
        else:
             patch_cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
             if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                  patch_padded = cv2.copyMakeBorder(patch_cropped, pad_top, pad_bottom, pad_left, pad_right,
                                                    IMAGE_BORDER_MODE, value=IMAGE_BORDER_VALUE)
             else:
                  patch_padded = patch_cropped
             if patch_padded.shape[0] != base_size or patch_padded.shape[1] != base_size:
                   patch_padded = cv2.resize(patch_padded, (base_size, base_size), interpolation=cv2.INTER_LINEAR)
             return patch_padded
    except Exception as e:
        logger.error(f"Error extrayendo parche base en ({center_x:.1f}, {center_y:.1f}): {e}")
        return None

def normalize_patch(patch):
    """ Normaliza parche a [0, 1]. """
    patch_float = patch.astype(np.float32)
    min_val, max_val = np.min(patch_float), np.max(patch_float)
    if max_val > min_val:
        normalized = (patch_float - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(patch_float)
    return normalized

# === FIN FUNCIONES COPIADAS ===

# --- El resto del script (load_esl_models, predict_line_position, etc.) ---
# ... (El código de estas funciones permanece igual que en la versión anterior) ...
def load_esl_models(models_dir):
    """Carga los 5 modelos ESL entrenados."""
    models = {}
    logger.info("Cargando modelos ESL...")
    all_loaded = True
    for key, filename in MODEL_NAMES.items():
        model_path = os.path.join(models_dir, filename)
        if not os.path.exists(model_path):
            logger.error(f"Modelo no encontrado: {model_path}")
            all_loaded = False
            continue
        try:
            models[key] = keras.models.load_model(model_path, compile=False)
            logger.info(f"Modelo '{key}' cargado desde {model_path}")
        except Exception as e:
            logger.error(f"Error cargando modelo '{key}' desde {model_path}: {e}")
            all_loaded = False
    if not all_loaded:
        logger.error("Fallo al cargar uno o más modelos ESL.")
        return None
    logger.info("Todos los modelos ESL disponibles fueron cargados.")
    return models

def predict_line_position(image, line_key, model, search_range_factor=SEARCH_RANGE_FACTOR, step=SEARCH_STEP):
    """Busca la mejor posición para una línea usando ventana deslizante."""
    img_h, img_w = image.shape
    best_prob = -1.0
    best_pos = -1.0

    # --- Pre-normalizar imagen completa ---
    image_norm = image.astype(np.float32)
    min_img, max_img = np.min(image_norm), np.max(image_norm)
    if max_img > min_img:
         image_norm = (image_norm - min_img) / (max_img - min_img)
    else:
         image_norm = np.zeros_like(image_norm)
    # --- Fin Pre-normalización ---

    if line_key in ['l1', 'l3']:
        coord_max = img_w; search_dim = 'x'; center_coord_other_dim = img_h / 2.0
    elif line_key in ['l2', 'l4']:
        coord_max = img_h; search_dim = 'y'; center_coord_other_dim = img_w / 2.0
    else: raise ValueError("Clave de línea no válida.")

    search_min = int(coord_max * (1.0 - search_range_factor) / 2.0)
    search_max = int(coord_max * (1.0 + search_range_factor) / 2.0)
    candidate_positions = range(search_min, search_max, step)

    if not list(candidate_positions):
         logger.warning(f"Rango de búsqueda vacío para {line_key}. Rango: [{search_min}, {search_max}]")
         return -1.0

    patches_to_predict = []
    positions_tested = []

    for pos_candidate in candidate_positions:
        center_x = float(pos_candidate) if search_dim == 'x' else center_coord_other_dim
        center_y = float(pos_candidate) if search_dim == 'y' else center_coord_other_dim
        patch = extract_patch(image_norm, center_x, center_y, PATCH_SIZE) # Usa imagen normalizada
        if patch is not None:
            patches_to_predict.append(patch)
            positions_tested.append(pos_candidate)

    if not patches_to_predict:
         logger.error(f"No se pudieron extraer parches válidos para {line_key}.")
         return -1.0

    patches_array = np.array(patches_to_predict, dtype=np.float32)
    if patches_array.ndim == 3: patches_array = patches_array[..., np.newaxis]

    try:
        if patches_array.shape[0] > 0:
            probabilities = model.predict(patches_array, batch_size=BATCH_SIZE, verbose=0).flatten()
            if len(probabilities) > 0:
                 best_idx = np.argmax(probabilities)
                 best_pos = float(positions_tested[best_idx])
                 best_prob = probabilities[best_idx]
            else: best_pos = -1.0
        else: best_pos = -1.0
    except Exception as e:
         logger.error(f"Error durante la predicción para {line_key}: {e}")
         best_pos = -1.0

    return best_pos


def predict_orientation(image, T_pred, S_pred, model_theta):
    """Predice la orientación usando el modelo theta."""
    Tx_img, Ty_img = T_pred
    Sw_img, Sh_img = S_pred

    if Sw_img <= 0 or Sh_img <= 0:
         logger.warning(f"Escala inválida S=({Sw_img:.1f}, {Sh_img:.1f}) para predecir orientación.")
         return 0.0

    diagonal = math.sqrt(Sw_img**2 + Sh_img**2)
    base_patch_size = int(math.ceil(diagonal * BASE_PATCH_PADDING_FACTOR))
    if base_patch_size < PATCH_SIZE: base_patch_size = PATCH_SIZE

    patch_base = extract_patch_base(image, Tx_img, Ty_img, base_patch_size)
    if patch_base is None:
        logger.error("Fallo al extraer parche base para predicción de orientación.")
        return 0.0

    center = (base_patch_size // 2, base_patch_size // 2)
    patches_to_predict = []
    angles_tested = []

    for angle in ROTATION_ANGLES_DEG:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_base = cv2.warpAffine(patch_base, M, (base_patch_size, base_patch_size),
                                      flags=cv2.INTER_LINEAR, borderMode=IMAGE_BORDER_MODE, borderValue=IMAGE_BORDER_VALUE)
        final_patch = cv2.resize(rotated_base, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LINEAR)
        normalized_patch = normalize_patch(final_patch)
        patches_to_predict.append(normalized_patch)
        angles_tested.append(angle)

    patches_array = np.array(patches_to_predict, dtype=np.float32)
    if patches_array.ndim == 3: patches_array = patches_array[..., np.newaxis]

    try:
        if patches_array.shape[0] > 0:
             probabilities = model_theta.predict(patches_array, batch_size=BATCH_SIZE, verbose=0).flatten()
             if len(probabilities) > 0:
                  best_idx = np.argmax(probabilities)
                  best_angle = angles_tested[best_idx]
                  return float(best_angle)
             else: return 0.0
        else: return 0.0

    except Exception as e:
         logger.error(f"Error durante la predicción de orientación: {e}")
         return 0.0

def predict_esl_pose(image_path, models):
    """Predice la pose T, S, theta usando los modelos ESL cargados."""
    logger.info(f"Procesando imagen: {image_path}")
    start_time = time.time()
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None: return None
        img_h, img_w = image.shape
        logger.info(f"Imagen cargada ({img_w}x{img_h}).")

        logger.info("Prediciendo posiciones de líneas...")
        L1_pred = predict_line_position(image, 'l1', models['l1'])
        L2_pred = predict_line_position(image, 'l2', models['l2'])
        L3_pred = predict_line_position(image, 'l3', models['l3'])
        L4_pred = predict_line_position(image, 'l4', models['l4'])

        if any(pos < 0 for pos in [L1_pred, L2_pred, L3_pred, L4_pred]): return None

        if L1_pred >= L3_pred or L2_pred >= L4_pred:
             logger.warning(f"Predicciones inconsistentes: L1={L1_pred:.1f}, L3={L3_pred:.1f}, L2={L2_pred:.1f}, L4={L4_pred:.1f}")
             if L1_pred >= L3_pred: L1_pred, L3_pred = L3_pred, L1_pred
             if L2_pred >= L4_pred: L2_pred, L4_pred = L4_pred, L2_pred
             if L3_pred - L1_pred < 1: L3_pred = L1_pred + 1
             if L4_pred - L2_pred < 1: L4_pred = L2_pred + 1
             logger.warning(f"--> Ajustadas a: L1={L1_pred:.1f}, L3={L3_pred:.1f}, L2={L2_pred:.1f}, L4={L4_pred:.1f}")

        logger.info(f"Líneas predichas: L1={L1_pred:.1f}, L2={L2_pred:.1f}, L3={L3_pred:.1f}, L4={L4_pred:.1f}")

        Tx_pred = (L1_pred + L3_pred) / 2.0
        Ty_pred = (L2_pred + L4_pred) / 2.0
        Sw_pred = L3_pred - L1_pred
        Sh_pred = L4_pred - L2_pred
        T_pred = (Tx_pred, Ty_pred)
        S_pred = (Sw_pred, Sh_pred)
        logger.info(f"Pose intermedia: T=({Tx_pred:.1f}, {Ty_pred:.1f}), S=({Sw_pred:.1f} x {Sh_pred:.1f})")

        logger.info("Prediciendo orientación...")
        theta_pred_deg = predict_orientation(image, T_pred, S_pred, models['theta'])
        logger.info(f"Orientación predicha: {theta_pred_deg:.1f} grados")

        end_time = time.time()
        logger.info(f"Predicción ESL completada en {end_time - start_time:.2f} segundos.")
        return {'T': T_pred, 'S': S_pred, 'theta_deg': theta_pred_deg}

    except Exception as e:
        logger.error(f"Error durante la predicción ESL para {image_path}: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    logger.info("====== Iniciando Prueba de Inferencia ESL ======")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"{len(gpus)} GPUs configuradas.")
        except RuntimeError as e: logger.error(f"Error configurando GPU: {e}")
    else: logger.info("No se detectaron GPUs. Usando CPU.")

    esl_models = load_esl_models(MODELS_DIR)

    if esl_models:
        try:
            # Ahora load_all_data está definida en este script
            index_map, _, image_paths = load_all_data(BASE_DIR)
            if not image_paths: raise ValueError("No se pudieron cargar las rutas de las imágenes.")

            test_indices_path = os.path.join(RESULTS_DIR, 'test_indices.txt')
            if not os.path.exists(test_indices_path): raise FileNotFoundError("test_indices.txt no encontrado")

            test_indices = np.loadtxt(test_indices_path, dtype=int)
            if test_indices.ndim == 0: test_indices = np.array([test_indices.item()])

            valid_test_indices = [idx for idx in test_indices if idx in image_paths]
            if not valid_test_indices: raise ValueError("No hay imágenes válidas en el conjunto de prueba.")

            target_test_idx = 790
            if target_test_idx in valid_test_indices: test_idx = target_test_idx
            else: test_idx = np.random.choice(valid_test_indices); logger.warning(f"Índice {target_test_idx} no válido. Usando aleatorio: {test_idx}")

            test_image_path = image_paths[test_idx]
            logger.info(f"Usando imagen de prueba: {test_image_path} (Índice: {test_idx})")

            predicted_pose = predict_esl_pose(test_image_path, esl_models)

            if predicted_pose:
                logger.info("\n--- Pose Predicha ---")
                logger.info(f"Centro T: ({predicted_pose['T'][0]:.2f}, {predicted_pose['T'][1]:.2f})")
                logger.info(f"Escala S: ({predicted_pose['S'][0]:.2f} x {predicted_pose['S'][1]:.2f})")
                logger.info(f"Orientación Theta: {predicted_pose['theta_deg']:.2f} grados")

                try:
                    image_disp = cv2.imread(test_image_path)
                    tx, ty = predicted_pose['T']
                    sw, sh = predicted_pose['S']
                    angle = predicted_pose['theta_deg']
                    center = (int(tx), int(ty)); size = (int(sw), int(sh))
                    rect_points = cv2.boxPoints(((center[0], center[1]), (size[0], size[1]), 0))
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated_rect_points = cv2.transform(np.array([rect_points]), M)[0]
                    rotated_rect_points_int = np.intp(rotated_rect_points)
                    cv2.drawContours(image_disp, [rotated_rect_points_int], 0, (0, 255, 0), 2)
                    cv2.circle(image_disp, center, 5, (0, 0, 255), -1)
                    vis_filename = f"esl_prediction_idx{test_idx}.png"
                    vis_path = os.path.join(PLOTS_DIR, vis_filename) # Usar PLOTS_DIR
                    cv2.imwrite(vis_path, image_disp)
                    logger.info(f"Visualización guardada en: {vis_path}")
                except Exception as e: logger.error(f"Error al generar visualización: {e}")
            else: logger.error("La predicción de pose ESL falló.")

        except Exception as e: logger.error(f"Error en el bloque principal de prueba: {e}", exc_info=True)
    else: logger.error("No se pudieron cargar los modelos ESL. Abortando prueba.")

    logger.info("====== Prueba de Inferencia ESL Finalizada ======")