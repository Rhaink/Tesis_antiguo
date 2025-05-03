# Tesis/alineamiento/src/mashdl_patch_extractor.py
# MODIFICADO para usar 144 puntos para extracción de parches

import os
import sys
import numpy as np
import cv2
import logging
logger = logging.getLogger(__name__)
import time
import tensorflow as tf
from tensorflow import keras

# --- Ajuste de Rutas para Importación ---
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ALINEAMIENTO_DIR = os.path.dirname(CURRENT_SCRIPT_DIR)
    if ALINEAMIENTO_DIR not in sys.path:
        sys.path.append(ALINEAMIENTO_DIR)
    BASE_PROJECT_DIR = os.path.dirname(ALINEAMIENTO_DIR)
    if BASE_PROJECT_DIR not in sys.path:
         sys.path.append(BASE_PROJECT_DIR)
    SRC_DIR_PATH = os.path.join(ALINEAMIENTO_DIR, "src")
    if SRC_DIR_PATH not in sys.path:
        sys.path.append(SRC_DIR_PATH)
except NameError:
    pass # Ajustar manualmente si es necesario en interactivo

# --- Importaciones de tu código ---
try:
    from src.data_loader import load_all_data
    from src.ssm_builder import devectorize_shape
    # Importar funciones necesarias (asegúrate que predict_esl_pose esté accesible)
    # Si estas funciones ya están definidas en este script, no necesitas importarlas
    try:
        from predict_esl_pose import load_esl_models, predict_esl_pose, extract_patch, normalize_patch, extract_patch_base
    except ImportError:
         logger.warning("No se pudo importar desde predict_esl_pose.py. Asegúrate de que esté en sys.path o define las funciones aquí.")
         # Define aquí las funciones extract_patch, etc., si no se pueden importar
         def extract_patch(image, center_x, center_y, patch_size):
             # ... (tu código de extract_patch) ...
             pass
         def normalize_patch(patch):
             # ... (tu código de normalize_patch) ...
             pass
         def extract_patch_base(image, center_x, center_y, base_size):
              # ... (tu código de extract_patch_base) ...
              pass
         def load_esl_models(models_dir):
             # ... (tu código de load_esl_models) ...
             pass
         def predict_esl_pose(image_path, models):
              # ... (tu código de predict_esl_pose) ...
              pass

    # Si necesitas ssm_fitter.calculate_normal aquí (no parece ser el caso ahora)
    # from src.ssm_fitter import calculate_normal
except ImportError as e:
    print(f"Error importando módulos DESPUÉS de ajustar sys.path: {e}")
    print(f"Verifica que los archivos .py existan en las ubicaciones esperadas (src/ y alineamiento/).")
    print(f"sys.path incluye ahora: {sys.path}")
    exit(1)

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración de Rutas ---
try:
    # CURRENT_SCRIPT_DIR y ALINEAMIENTO_DIR ya están definidos arriba
    BASE_DIR = os.path.dirname(ALINEAMIENTO_DIR) # Directorio 'Tesis'
    RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
    MODELS_DIR = os.path.join(ALINEAMIENTO_DIR, "models")
    PLOTS_DIR = os.path.join(ALINEAMIENTO_DIR, "plots")
except NameError:
    ALINEAMIENTO_DIR = '.'
    BASE_DIR = os.path.dirname(ALINEAMIENTO_DIR);
    if not BASE_DIR: BASE_DIR = '..'
    RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
    MODELS_DIR = os.path.join(ALINEAMIENTO_DIR, "models")
    PLOTS_DIR = os.path.join(ALINEAMIENTO_DIR, "plots")
    logger.info("Ejecutando en modo interactivo, rutas ajustadas.")

# --- Rutas Archivos de Entrada ---
TRAIN_INDICES_PATH = os.path.join(RESULTS_DIR, 'train_indices.txt')
B_GT_PATH = os.path.join(RESULTS_DIR, 'mashdl_ground_truth_b_train.npz')
PCA_MEAN_VECTOR_PATH = os.path.join(RESULTS_DIR, 'pca_mean_vector.npy')
PCA_COMPONENTS_PATH = os.path.join(RESULTS_DIR, 'pca_components.npy')
PCA_STD_DEVS_PATH = os.path.join(RESULTS_DIR, 'pca_std_devs.npy')
# MEAN_SHAPE_PATH = os.path.join(RESULTS_DIR, 'mean_shape.npy') # No se usa directamente aquí

# --- Ruta Archivo de Salida ---
OUTPUT_MASHSDL_DATA_PATH = os.path.join(RESULTS_DIR, 'mashdl_training_hypotheses_144lmk.npz') # Nuevo nombre

# --- Parámetros Clave ---
Q_PATCH_SIZE = 21            # Tamaño del parche local (ej. 21x21)
# NUM_LANDMARKS = 15         # <-- YA NO SE USA para determinar el número de parches
NUM_TOTAL_SHAPE_POINTS = 144 # <-- NUEVO: Número de puntos de la forma completa a usar
INPUT_DIM = NUM_TOTAL_SHAPE_POINTS * (Q_PATCH_SIZE ** 2) # <-- NUEVO CÁLCULO: 144 * 441 = 63504
NUM_MODES = 4                # Modos SSM a procesar
NUM_SSM_LANDMARKS = 15       # Landmarks originales del SSM (puede ser útil para info)
NUM_DIMS = 2                 # Dimensiones (2D)
NUM_NEG_PER_POS = 3          # Hipótesis negativas por positiva
B_BINS = 7                   # Número de clases/bins para b_k
B_CLAMP_STD = 3.0            # Rango de b_k en desviaciones estándar
IMAGE_BORDER_MODE = cv2.BORDER_CONSTANT
IMAGE_BORDER_VALUE = 0

# ===========================================================
# Funciones de Utilidad SSM (Placeholder - Usa las importadas o las tuyas)
# ===========================================================
def generate_shape_instance(mean_vector, P, b):
    """Genera una instancia de forma (k x d) a partir de los parámetros b."""
    # Esta función debe devolver la forma completa (ej. 144, 2)
    num_modes_in_P = P.shape[0]
    k_times_d = mean_vector.shape[0]
    # Inferir k y d (AHORA k debería ser 144)
    d = NUM_DIMS
    k = k_times_d // d # Debería resultar en 144 si los archivos SSM son correctos

    # Verificar si k es 144, si no, advertir
    if k != NUM_TOTAL_SHAPE_POINTS:
        logger.warning(f"_generate_shape_instance: k inferido ({k}) no coincide con NUM_TOTAL_SHAPE_POINTS ({NUM_TOTAL_SHAPE_POINTS}). Usando k={k}.")
        # Podrías forzar k=144 aquí si estás seguro, pero es mejor arreglar los archivos SSM
        # k = NUM_TOTAL_SHAPE_POINTS
        # k_times_d = k * d

    if b is None:
        b = np.zeros(num_modes_in_P)
    elif b.shape != (num_modes_in_P,):
         # logger.warning(f"_generate_shape_instance: Shape de b inválido {b.shape}. Esperado ({num_modes_in_P},). Usando b=0.")
         b = np.zeros(num_modes_in_P)

    try:
        if P.shape[0] != len(b):
             raise ValueError(f"Incompatibilidad entre P.shape[0]={P.shape[0]} y len(b)={len(b)}")
        shape_vector = mean_vector + P.T @ b
        # Añadir log para verificar k inferido
        # logger.debug(f"Generando forma con k={k}, d={d}")
        return devectorize_shape(shape_vector, k, d) # Usa k inferido
    except ValueError as e:
        logger.error(f"Error en matmul de _generate_shape_instance (P.T @ b): {e}. Usando mean_shape.")
        return devectorize_shape(mean_vector, k, d)
    except Exception as e:
         logger.error(f"Error inesperado en _generate_shape_instance: {e}")
         return devectorize_shape(mean_vector, k, d)

def apply_similarity_transform(shape_model, s_x, s_y, theta_rad, tx, ty):
    """Aplica transformación de similitud ANISOTRÓPICA a una forma modelo."""
    if shape_model is None: return None
    # Validar que shape_model tenga shape (N, 2)
    if shape_model.ndim != 2 or shape_model.shape[1] != 2:
         logger.error(f"apply_similarity_transform: shape_model inválido. Shape: {shape_model.shape}")
         return None

    # 1. Escalar (anisotrópicamente)
    scaled_shape = shape_model.copy()
    scaled_shape[:, 0] *= s_x
    scaled_shape[:, 1] *= s_y
    # 2. Rotar
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                  [np.sin(theta_rad),  np.cos(theta_rad)]])
    rotated_shape = scaled_shape @ R.T # Matriz (N, 2) x (2, 2) -> (N, 2)
    # 3. Trasladar
    rotated_shape[:, 0] += tx
    rotated_shape[:, 1] += ty
    return rotated_shape

# ===========================================================
# Función Principal (Modificada)
# ===========================================================
def generate_mashdl_hypotheses():
    """Función principal para cargar datos y generar hipótesis MaShDL
       usando TODOS los puntos de la forma reconstruida."""
    logger.info(f"--- Iniciando Generación de Hipótesis MaShDL (Usando {NUM_TOTAL_SHAPE_POINTS} Puntos/Parches) ---")
    logger.info(f"Dimensión de entrada resultante: {INPUT_DIM}")
    start_total_time = time.time()

    # --- 1. Cargar Datos Necesarios ---
    logger.info("Cargando datos de entrada...")
    try:
        # Índices de entrenamiento
        if not os.path.exists(TRAIN_INDICES_PATH): raise FileNotFoundError(TRAIN_INDICES_PATH)
        train_indices_all = np.loadtxt(TRAIN_INDICES_PATH, dtype=int)
        if train_indices_all.ndim == 0: train_indices_all = np.array([train_indices_all.item()])
        logger.info(f"Cargados {len(train_indices_all)} índices de entrenamiento raw.")

        # Datos b Ground Truth
        if not os.path.exists(B_GT_PATH): raise FileNotFoundError(B_GT_PATH)
        b_gt_data = np.load(B_GT_PATH)
        b_gt_all = b_gt_data['b_gt']
        b_gt_indices = b_gt_data['index']
        if b_gt_all.shape[1] != NUM_MODES:
             raise ValueError(f"Número de modos en b_gt ({b_gt_all.shape[1]}) no coincide con NUM_MODES ({NUM_MODES})")
        logger.info(f"Cargados {b_gt_all.shape[0]} vectores b_gt con {b_gt_all.shape[1]} modos.")
        b_gt_map = {idx: b_gt for idx, b_gt in zip(b_gt_indices, b_gt_all)}
        train_indices = [idx for idx in train_indices_all if idx in b_gt_map]
        if len(train_indices) < len(train_indices_all):
             logger.warning(f"Se usarán {len(train_indices)} índices de entrenamiento que tienen b_gt (de {len(train_indices_all)}).")
        if not train_indices: raise ValueError("No hay índices válidos con b_gt asociado.")

        # Parámetros SSM
        if not os.path.exists(PCA_MEAN_VECTOR_PATH): raise FileNotFoundError(PCA_MEAN_VECTOR_PATH)
        pca_mean_vector = np.load(PCA_MEAN_VECTOR_PATH)
        if not os.path.exists(PCA_COMPONENTS_PATH): raise FileNotFoundError(PCA_COMPONENTS_PATH)
        pca_components = np.load(PCA_COMPONENTS_PATH)
        if not os.path.exists(PCA_STD_DEVS_PATH): raise FileNotFoundError(PCA_STD_DEVS_PATH)
        pca_std_devs = np.load(PCA_STD_DEVS_PATH)

        # --- Validar consistencia SSM con NUM_TOTAL_SHAPE_POINTS ---
        inferred_k_times_d = pca_mean_vector.shape[0]
        inferred_k = inferred_k_times_d // NUM_DIMS
        if inferred_k != NUM_TOTAL_SHAPE_POINTS:
             logger.error(f"¡ERROR CRÍTICO! El vector medio cargado (shape {pca_mean_vector.shape}) implica k={inferred_k}, pero se esperan {NUM_TOTAL_SHAPE_POINTS} puntos.")
             logger.error("Verifica los archivos .npy de PCA (mean_vector, components) generados por run_pca_analysis.py.")
             logger.error("El SSM DEBE representar la forma completa de 144 puntos.")
             return # No continuar si el SSM no representa 144 puntos
        if pca_components.shape != (NUM_MODES, inferred_k_times_d):
             logger.error(f"¡ERROR CRÍTICO! Shape de componentes PCA ({pca_components.shape}) inconsistente. Esperado ({NUM_MODES}, {inferred_k_times_d}).")
             return
        if pca_std_devs.shape != (NUM_MODES,):
             raise ValueError(f"Shape de Std Devs ({pca_std_devs.shape}) no coincide con NUM_MODES ({NUM_MODES})")

        logger.info(f"Parámetros SSM cargados y validados para k={inferred_k} puntos.")

        # Datos de imágenes
        project_base_dir = os.path.dirname(ALINEAMIENTO_DIR)
        _, _, image_paths_dict = load_all_data(project_base_dir)
        if not image_paths_dict: raise ValueError(f"No se pudieron cargar las rutas de imágenes desde {project_base_dir}.")

        # Cargar modelos ESL
        esl_models = load_esl_models(MODELS_DIR)
        if not esl_models: raise ValueError("No se pudieron cargar los modelos ESL.")

    except FileNotFoundError as e:
        logger.error(f"Error: Archivo no encontrado: {e}."); return
    except ValueError as e:
        logger.error(f"Error en datos de entrada: {e}"); return
    except Exception as e:
        logger.error(f"Error inesperado cargando datos: {e}", exc_info=True); return

    logger.info("Carga de datos completada.")

    # --- 2. Inicializar Estructuras de Datos ---
    mashdl_data = {k: {'features': [], 'labels': []} for k in range(NUM_MODES)}
    processed_count = 0; error_count = 0; skipped_esl_fail = 0; skipped_patch_fail = 0

    # --- 3. Bucle Principal ---
    logger.info(f"Iniciando generación de hipótesis para {len(train_indices)} imágenes...")
    total_images = len(train_indices)

    for i, img_idx in enumerate(train_indices):
        if (i + 1) % 10 == 0: logger.info(f"Procesando imagen {i+1}/{total_images} (Índice: {img_idx})...")

        # --- Obtener Imagen ---
        image_path = image_paths_dict.get(img_idx)
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"Ruta inválida o imagen no encontrada para índice {img_idx}. Saltando."); error_count += 1; continue
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None: raise IOError("cv2.imread devolvió None")
            # Pre-normalizar imagen a [0, 1] float32
            image_norm = image.astype(np.float32)
            min_val, max_val = np.min(image_norm), np.max(image_norm)
            if max_val > min_val: image_norm = (image_norm - min_val) / (max_val - min_val)
            else: image_norm = np.zeros_like(image_norm)
        except Exception as e:
            logger.error(f"Error cargando/normalizando imagen {image_path}: {e}"); error_count += 1; continue

        # --- Obtener Pose ESL ---
        try:
            predicted_pose = predict_esl_pose(image_path, esl_models)
            if predicted_pose is None: logger.warning(f"ESL falló idx {img_idx}. Saltando."); skipped_esl_fail += 1; continue
            T_hat, S_hat, theta_hat_deg = predicted_pose['T'], predicted_pose['S'], predicted_pose['theta_deg']
            theta_hat_rad = np.radians(theta_hat_deg)
            Sx_img, Sy_img = S_hat[0], S_hat[1]
            Tx_img, Ty_img = T_hat[0], T_hat[1]
            if Sx_img < 1 or Sy_img < 1: logger.warning(f"Escala ESL inválida S=({Sx_img:.1f}, {Sy_img:.1f}) idx {img_idx}. Saltando."); skipped_esl_fail +=1; continue
        except Exception as e: logger.error(f"Error predicción ESL idx {img_idx}: {e}"); error_count += 1; continue

        # --- Obtener b_gt ---
        b_gt_i = b_gt_map[img_idx]

        # --- Iterar sobre los modos k = 0..NUM_MODES-1 ---
        b_accumulated_gt = np.zeros(NUM_MODES)
        patch_extraction_failed_for_image = False

        for k in range(NUM_MODES):
            if patch_extraction_failed_for_image: break

            # --- Hipótesis Positiva ---
            # a. Reconstruir forma base x_k (144 puntos)
            shape_k_model = generate_shape_instance(pca_mean_vector, pca_components, b_accumulated_gt)
            if shape_k_model is None or shape_k_model.shape[0] != NUM_TOTAL_SHAPE_POINTS:
                 logger.error(f"Fallo al generar shape_k_model ({shape_k_model.shape if shape_k_model is not None else 'None'}) idx {img_idx}, k={k}. Saltando imagen."); error_count += 1; patch_extraction_failed_for_image = True; break
                 # logger.info(f"DEBUG: shape_k_model.shape = {shape_k_model.shape}") # Añadir para depurar

            # b. Transformar x_k a espacio imagen
            shape_k_image = apply_similarity_transform(shape_k_model, Sx_img, Sy_img, theta_hat_rad, Tx_img, Ty_img)
            if shape_k_image is None: logger.error(f"Fallo al transformar shape_k_model idx {img_idx}, k={k}. Saltando."); error_count += 1; patch_extraction_failed_for_image = True; break

            # c. Extraer M=144 parches (qxq)
            patches_pos = []
            # ***** CAMBIO CRÍTICO: Iterar sobre TODOS los puntos de la forma *****
            for point_idx in range(NUM_TOTAL_SHAPE_POINTS): # Iterar de 0 a 143
                center_x, center_y = shape_k_image[point_idx, 0], shape_k_image[point_idx, 1]
                patch = extract_patch(image_norm, center_x, center_y, Q_PATCH_SIZE)
                if patch is None:
                    logger.warning(f"Fallo extracción parche POSITIVO Punto:{point_idx} idx {img_idx}, k={k}. Saltando hipótesis."); patch_extraction_failed_for_image = True; break
                patches_pos.append(patch)
            # ***** FIN CAMBIO CRÍTICO *****

            if patch_extraction_failed_for_image: skipped_patch_fail += 1; continue

            # d. Concatenar/Aplanar parches -> feature_vector_pos (ahora más largo)
            try: feature_vector_pos = np.concatenate([p.flatten() for p in patches_pos])
            except ValueError as e: logger.error(f"Error concatenando parches POSITIVOS idx {img_idx}, k={k}: {e}. Saltando."); error_count += 1; continue

            # Verificar dimensión final del vector de características
            if feature_vector_pos.shape[0] != INPUT_DIM:
                 logger.error(f"ERROR FATAL: Dimensión del vector positivo ({feature_vector_pos.shape[0]}) no coincide con INPUT_DIM ({INPUT_DIM}) idx {img_idx}, k={k}.")
                 error_count += 1; continue # No guardar datos incorrectos

            # e. Obtener etiqueta positiva
            label_pos = discretize_b(b_gt_i[k], pca_std_devs[k], num_bins=B_BINS, clamp_std=B_CLAMP_STD)

            # f. Guardar
            mashdl_data[k]['features'].append(feature_vector_pos)
            mashdl_data[k]['labels'].append(label_pos)

            # --- Hipótesis Negativas ---
            neg_generated = 0; attempts = 0; MAX_ATTEMPTS = NUM_NEG_PER_POS * 5

            while neg_generated < NUM_NEG_PER_POS and attempts < MAX_ATTEMPTS:
                attempts += 1
                # a. Generar b_hat_k incorrecto
                b_hat_k = b_gt_i[k]
                while abs(b_hat_k - b_gt_i[k]) < 0.25 * pca_std_devs[k]:
                    b_hat_k = np.random.uniform(-B_CLAMP_STD * pca_std_devs[k], B_CLAMP_STD * pca_std_devs[k])
                b_neg = b_accumulated_gt.copy(); b_neg[k] = b_hat_k

                # c. Reconstruir forma sintética x_hat_k (144 puntos)
                shape_neg_model = generate_shape_instance(pca_mean_vector, pca_components, b_neg)
                if shape_neg_model is None or shape_neg_model.shape[0] != NUM_TOTAL_SHAPE_POINTS: continue # Intentar otro b_hat_k

                # d. Transformar a espacio imagen
                shape_neg_image = apply_similarity_transform(shape_neg_model, Sx_img, Sy_img, theta_hat_rad, Tx_img, Ty_img)
                if shape_neg_image is None: continue

                # e. Extraer M=144 parches
                patches_neg = []
                extraction_ok = True
                # ***** CAMBIO CRÍTICO: Iterar sobre TODOS los puntos *****
                for point_idx in range(NUM_TOTAL_SHAPE_POINTS): # Iterar de 0 a 143
                    center_x, center_y = shape_neg_image[point_idx, 0], shape_neg_image[point_idx, 1]
                    patch = extract_patch(image_norm, center_x, center_y, Q_PATCH_SIZE)
                    if patch is None: extraction_ok = False; break
                    patches_neg.append(patch)
                # ***** FIN CAMBIO CRÍTICO *****

                if not extraction_ok: continue

                # f. Concatenar parches
                try: feature_vector_neg = np.concatenate([p.flatten() for p in patches_neg])
                except ValueError: continue

                # Verificar dimensión final
                if feature_vector_neg.shape[0] != INPUT_DIM:
                     logger.error(f"ERROR FATAL: Dimensión del vector negativo ({feature_vector_neg.shape[0]}) no coincide con INPUT_DIM ({INPUT_DIM}) idx {img_idx}, k={k}.")
                     continue # No guardar datos incorrectos

                # g. Obtener etiqueta negativa
                label_neg = discretize_b(b_hat_k, pca_std_devs[k], num_bins=B_BINS, clamp_std=B_CLAMP_STD)

                # h. Guardar
                mashdl_data[k]['features'].append(feature_vector_neg)
                mashdl_data[k]['labels'].append(label_neg)
                neg_generated += 1
            # --- Fin Bucle Negativos ---

            # Actualizar b_accumulated_gt para la base del siguiente modo
            b_accumulated_gt[k] = b_gt_i[k]
        # --- Fin Bucle Modos k ---

        if not patch_extraction_failed_for_image: processed_count += 1
    # --- Fin Bucle Imágenes ---

    # --- 4. Guardar Datos Generados ---
    logger.info("Finalizando generación. Preparando para guardar...")
    output_dict = {}
    total_samples = 0
    for k in range(NUM_MODES):
        try:
            features_k = np.array(mashdl_data[k]['features'], dtype=np.float32)
            labels_k = np.array(mashdl_data[k]['labels'], dtype=np.int32)
            logger.info(f"Modo {k}: Features shape={features_k.shape}, Labels shape={labels_k.shape}")

            # Validar dimensión de características
            if features_k.ndim > 0 and features_k.shape[1] != INPUT_DIM:
                 logger.error(f"¡ERROR AL GUARDAR! Modo {k}: Dimensión de feature ({features_k.shape[1]}) no coincide con INPUT_DIM ({INPUT_DIM}). No se guardará este modo.")
                 continue # Saltar este modo

            if features_k.shape[0] > 0 and features_k.shape[0] == labels_k.shape[0]:
                output_dict[f'features_k{k}'] = features_k
                output_dict[f'labels_k{k}'] = labels_k
                total_samples += features_k.shape[0]
                label_dist = np.bincount(labels_k, minlength=B_BINS)
                logger.info(f"  Distribución Etiquetas Modo {k} (bins 0 a {B_BINS-1}): {label_dist}")
            else: logger.warning(f"No se generaron datos válidos o hay inconsistencia para modo {k}.")
        except Exception as e: logger.error(f"Error convirtiendo datos modo {k} a NumPy: {e}")

    if output_dict:
        logger.info(f"Total de muestras generadas (todas K válidas): {total_samples}")
        logger.info(f"Guardando datos en: {OUTPUT_MASHSDL_DATA_PATH}")
        try: np.savez_compressed(OUTPUT_MASHSDL_DATA_PATH, **output_dict)
        except Exception as e: logger.error(f"Error al guardar NPZ: {e}")
    else: logger.error("No se generaron datos válidos para guardar.")

    end_total_time = time.time()
    logger.info(f"--- Generación Hipótesis MaShDL (144 Puntos) Finalizada ---")
    logger.info(f"Tiempo total: {(end_total_time - start_total_time)/60:.2f} minutos")
    logger.info(f"Imágenes procesadas OK: {processed_count}/{total_images}")
    logger.info(f"Saltadas por ESL: {skipped_esl_fail}, Saltadas por Extracción Patch: {skipped_patch_fail}, Otros errores: {error_count}")

# --- Funciones Auxiliares (Discretización) ---
def discretize_b(b_value, std_dev_k, num_bins=B_BINS, clamp_std=B_CLAMP_STD):
    """Mapea un valor b_k continuo a un índice de clase [0, num_bins-1]."""
    # (Sin cambios respecto a tu versión)
    min_val = -clamp_std * std_dev_k
    max_val = clamp_std * std_dev_k
    range_val = max_val - min_val
    if range_val < 1e-9: return num_bins // 2
    clamped_b = np.clip(b_value, min_val, max_val)
    normalized_b = (clamped_b - min_val) / range_val
    bin_index = int(np.floor(normalized_b * num_bins))
    return np.clip(bin_index, 0, num_bins - 1)

# ===========================================================
# Punto de Entrada
# ===========================================================
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"{len(gpus)} GPUs configuradas.")
        except RuntimeError as e: logger.error(f"Error configurando GPU: {e}")
    else: logger.info("No se detectaron GPUs.")

    generate_mashdl_hypotheses()