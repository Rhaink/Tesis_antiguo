# Tesis/alineamiento/src/profile_builder.py (v4 - Usa Z-score)
import numpy as np
import cv2
import os
import logging
from scipy import ndimage
from scipy.linalg import inv, pinv
import pandas as pd
# Importar tu cargador de datos
from data_loader import load_all_data

# --- Configuración Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)
# ----------------------------

# --- Configuración de Rutas Relativas (Consistente con data_loader.py) ---
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ALINEAMIENTO_DIR = os.path.dirname(CURRENT_SCRIPT_DIR) # Tesis/alineamiento
    BASE_DIR = os.path.dirname(ALINEAMIENTO_DIR) # Directorio 'Tesis'
    RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
except NameError: # Para ejecución interactiva
     # Definir rutas manualmente si __file__ no está definido
    ALINEAMIENTO_DIR = '.' # Ajustar según sea necesario
    BASE_DIR = os.path.dirname(ALINEAMIENTO_DIR)
    RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
    logging.info("Ejecutando en modo interactivo, rutas ajustadas manualmente.")

# --- Rutas a Archivos Necesarios (Actualizado) ---
# Cargar la forma media guardada por run_gpa.py
MEAN_SHAPE_PATH = os.path.join(RESULTS_DIR, 'mean_shape.npy') # <--- USAMOS ESTE
TRAIN_INDICES_PATH = os.path.join(RESULTS_DIR, 'train_indices.txt')
PROFILE_MODELS_OUTPUT_PATH = os.path.join(RESULTS_DIR, 'asm_profile_models.npz')

# --- Parámetros del Modelo de Perfil ---
NUM_LANDMARKS = 15 # k
NUM_DIMS = 2       # d
PROFILE_LENGTH = 21
GRADIENT_KERNEL_SIZE = 5
GAUSSIAN_KSIZE_PRE = 3
COV_REGULARIZATION_LAMBDA = 1e-6
NORMALIZE_PROFILE_METHOD = 'zscore' # <-- *** CAMBIO REALIZADO AQUÍ ***
# ------------------------------------

# --- Funciones Auxiliares (calculate_normal, sample_profile, normalize_profile) ---
def calculate_normal(p_prev, p_next):
    """Calcula el vector normal unitario a un segmento."""
    segment_vec = p_next - p_prev
    segment_len = np.linalg.norm(segment_vec)
    # Si los puntos son coincidentes, usar una normal por defecto (ej. hacia arriba)
    if segment_len < 1e-6: return np.array([0.0, -1.0])
    # Normal perpendicular (girar 90 grados y normalizar)
    normal = np.array([-segment_vec[1], segment_vec[0]]) / segment_len
    return normal

def sample_profile(image, point, normal, length, spacing=1.0):
    """Muestrea un perfil de intensidad a lo largo de una normal."""
    half_length = (length - 1) / 2
    # Crear puntos de muestreo a lo largo de la normal
    distances = np.linspace(-half_length * spacing, half_length * spacing, length)
    # Asegurar que la normal sea unitaria para el espaciado correcto
    norm_mag = np.linalg.norm(normal)
    safe_normal = normal / norm_mag if norm_mag > 1e-6 else np.array([0.0, -1.0])

    # Calcular coordenadas de muestreo (fila, columna) - ¡Ojo con orden (y,x) para map_coordinates!
    sample_coords_row = point[1] + distances * safe_normal[1] # Coordenada Y (fila)
    sample_coords_col = point[0] + distances * safe_normal[0] # Coordenada X (columna)

    try:
        # Usar map_coordinates para interpolación bilineal (order=1)
        profile = ndimage.map_coordinates(image,
                                          [sample_coords_row, sample_coords_col],
                                          order=1, mode='nearest', cval=0.0)
    except Exception as e:
        # logger.warning(f"Excepción en map_coordinates para punto {point} con normal {normal}: {e}. Devolviendo perfil de ceros.")
        profile = np.zeros(length) # Devolver ceros en caso de error
    return profile

def normalize_profile(profile, method='sum'):
    """Normaliza un perfil usando el método especificado."""
    if method == 'sum':
        norm_factor = np.sum(np.abs(profile)) + 1e-6 # Añadir epsilon para evitar división por cero
        return profile / norm_factor
    elif method == 'zscore':
        mean = np.mean(profile)
        std = np.std(profile) + 1e-6 # Añadir epsilon para evitar división por cero si std es 0
        return (profile - mean) / std
    else: # Si no es 'sum' ni 'zscore', devolver sin normalizar
        logger.warning(f"Método de normalización '{method}' no reconocido. Devolviendo perfil original.")
        return profile
# -----------------------------------------------------------------------------

def build_profile_models():
    logger.info(f"--- Iniciando Construcción de Modelos de Perfil ASM (Normalización: {NORMALIZE_PROFILE_METHOD.upper()}) ---")
    logger.info(f"Usando BASE_DIR: {BASE_DIR}")
    logger.info(f"Directorio de Resultados: {RESULTS_DIR}")

    # --- Carga de Datos usando data_loader ---
    logger.info("Cargando datos generales usando data_loader...")
    # Asegurarse que load_all_data usa las rutas correctas relativas a BASE_DIR si es necesario
    index_map, landmarks_array_orig, image_paths_dict = load_all_data(BASE_DIR)
    if index_map is None or landmarks_array_orig is None or image_paths_dict is None:
        logger.error("Fallo al cargar datos iniciales. Abortando."); return

    logger.info("Cargando datos específicos para entrenamiento...")
    try:
        # Carga la FORMA MEDIA guardada por run_gpa.py
        if not os.path.exists(MEAN_SHAPE_PATH):
             raise FileNotFoundError(f"Archivo de forma media no encontrado: {MEAN_SHAPE_PATH}")
        mean_aligned_shape = np.load(MEAN_SHAPE_PATH)
        logger.info(f"Forma media alineada cargada desde {MEAN_SHAPE_PATH}. Shape: {mean_aligned_shape.shape}")
        if mean_aligned_shape.shape != (NUM_LANDMARKS, NUM_DIMS):
            raise ValueError("Dimensiones de la forma media incorrectas.")

        # Carga índices de entrenamiento
        if not os.path.exists(TRAIN_INDICES_PATH):
             raise FileNotFoundError(f"Archivo de índices de entrenamiento no encontrado: {TRAIN_INDICES_PATH}")
        train_indices = np.loadtxt(TRAIN_INDICES_PATH, dtype=int)
        if train_indices.ndim == 0: train_indices = np.array([train_indices.item()])

        # Validar índices contra el mapa de índices cargado
        valid_train_indices = [idx for idx in train_indices if idx in index_map.index]
        if len(valid_train_indices) < len(train_indices):
             logging.warning(f"Se descartaron {len(train_indices) - len(valid_train_indices)} índices de entrenamiento no presentes en el index_map.")
        if not valid_train_indices: raise ValueError("No hay índices de entrenamiento válidos en el index_map.")
        train_indices = valid_train_indices
        logger.info(f"Se usarán {len(train_indices)} índices de entrenamiento válidos.")

    except FileNotFoundError as e:
        logger.error(f"Error crítico: Archivo no encontrado: {e}"); return
    except Exception as e:
        logger.error(f"Error cargando datos de entrenamiento específicos: {e}", exc_info=True); return

    # --- Cálculo de Perfiles ---
    logger.info("Calculando perfiles de gradiente para datos de entrenamiento...")
    all_profiles = [[] for _ in range(NUM_LANDMARKS)] # Lista de listas para guardar perfiles por landmark

    num_processed = 0
    num_errors = 0
    for new_idx in train_indices:
        try:
            img_path = image_paths_dict.get(new_idx)
            # Verificar si la imagen existe y es cargable ANTES de procesar
            if img_path is None or not os.path.exists(img_path):
                 logging.warning(f"Ruta inválida o archivo no encontrado para índice {new_idx}. Saltando.")
                 num_errors += 1
                 continue
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                 logging.warning(f"No se pudo cargar la imagen para índice {new_idx} desde {img_path}. Saltando.")
                 num_errors += 1
                 continue

            # Obtener landmarks originales (asumimos 64x64) para esta imagen
            # Usamos index_map para encontrar la fila correspondiente si landmarks_array_orig está indexado de 0 a N-1
            # O si landmarks_array_orig usa new_idx directamente como índice (VERIFICAR ESTO)
            # Asumiendo que landmarks_array_orig usa índices 0..N-1 mapeados desde new_idx
            try:
                # Si landmarks_array_orig NO está indexado por new_idx, necesitamos encontrar la fila
                # Esto depende de cómo se construyó landmarks_array_orig en load_all_data
                # Asumiremos por ahora que está alineado con index_map.index si se cargó bien
                # Si load_all_data devuelve un array ordenado por new_index, podemos buscarlo.
                # MÁS SIMPLE: Asumir que load_all_data devuelve el array y que podemos usar new_idx si es un índice simple 0..N-1
                # O si es un índice disperso, necesitamos mapearlo. Vamos a asumir que load_all_data devuelve
                # un array donde la posición `i` corresponde al `i`-ésimo índice en `index_map.index`.
                # O, si `landmarks_array_orig` es un diccionario... hay que clarificar esto.
                # *** Suposición Peligrosa: Asumamos que landmarks_array_orig está indexado 0..N-1 y new_idx es ese índice ***
                #    (Si no es así, el siguiente acceso fallará o será incorrecto)
                original_landmarks = landmarks_array_orig[new_idx] # ¡VERIFICAR SI ESTO ES CORRECTO!

            except IndexError:
                 logging.error(f"Error de índice al acceder a landmarks_array_orig con new_idx={new_idx}. ¿La estructura de datos es la esperada?")
                 num_errors += 1
                 continue
            except KeyError: # Si landmarks_array_orig fuera un dict
                 logging.error(f"Error de clave al acceder a landmarks_array_orig con new_idx={new_idx}. ¿Es un diccionario?")
                 num_errors += 1
                 continue


            # Preprocesamiento y Gradiente
            processed_image = image.astype(np.float32) # Convertir a float para gradientes
            if GAUSSIAN_KSIZE_PRE > 0 and GAUSSIAN_KSIZE_PRE % 2 == 1:
                 processed_image = cv2.GaussianBlur(processed_image, (GAUSSIAN_KSIZE_PRE, GAUSSIAN_KSIZE_PRE), 0)
            grad_x = cv2.Sobel(processed_image, cv2.CV_64F, 1, 0, ksize=GRADIENT_KERNEL_SIZE)
            grad_y = cv2.Sobel(processed_image, cv2.CV_64F, 0, 1, ksize=GRADIENT_KERNEL_SIZE)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)

            # Iterar sobre cada landmark para muestrear su perfil
            for i in range(NUM_LANDMARKS):
                p_orig = original_landmarks[i] # Coordenadas del landmark en la imagen ORIGINAL (64x64)

                # Calcular normal usando la FORMA MEDIA ALINEADA (cargada, en espacio Procrustes)
                # Los puntos p_prev y p_next se toman de la forma media
                p_prev_aligned = mean_aligned_shape[(i - 1 + NUM_LANDMARKS) % NUM_LANDMARKS]
                p_next_aligned = mean_aligned_shape[(i + 1) % NUM_LANDMARKS]
                normal_vec = calculate_normal(p_prev_aligned, p_next_aligned)

                # Muestrear el perfil de MAGNITUD DE GRADIENTE en la posición ORIGINAL del landmark
                # usando la normal calculada de la forma media
                #profile = sample_profile(grad_mag, p_orig, normal_vec, PROFILE_LENGTH)
                profile = sample_profile(processed_image, p_orig, normal_vec, PROFILE_LENGTH)

                # Normalizar el perfil muestreado
                norm_profile = normalize_profile(profile, method=NORMALIZE_PROFILE_METHOD)

                # Guardar el perfil normalizado para este landmark
                all_profiles[i].append(norm_profile)

            num_processed += 1
            if num_processed % 50 == 0 or num_processed == len(train_indices):
                logger.info(f"Procesadas {num_processed}/{len(train_indices)} imágenes de entrenamiento...")

        except Exception as e:
            logger.error(f"Error procesando imagen índice {new_idx}: {e}", exc_info=True)
            num_errors += 1

    logger.info(f"Procesamiento de imágenes finalizado. {num_processed} procesadas, {num_errors} errores.")
    if num_processed == 0:
         logger.error("No se procesó ninguna imagen con éxito. Abortando construcción de modelos.")
         return

    # --- Cálculo de Modelos Estadísticos y Guardado ---
    logger.info(f"Perfiles calculados. Construyendo modelos estadísticos (Mean, InvCov) para {NUM_LANDMARKS} landmarks...")
    profile_models = []
    for i in range(NUM_LANDMARKS):
        try:
            # Convertir la lista de perfiles para este landmark a un array NumPy
            profiles_array = np.array(all_profiles[i])

            # Verificar si tenemos suficientes perfiles para calcular covarianza
            if profiles_array.shape[0] < 2:
                 logger.warning(f"Insuficientes perfiles ({profiles_array.shape[0]}) para landmark {i}. Creando placeholder (media=0, inv_cov=identidad).")
                 mean_prof = np.zeros(PROFILE_LENGTH)
                 # Usar una identidad pequeña como placeholder de inv_cov
                 inv_cov_mat = np.identity(PROFILE_LENGTH) * 1e-9
            else:
                # Calcular perfil medio
                mean_prof = np.mean(profiles_array, axis=0)
                # Calcular matriz de covarianza
                # rowvar=False porque cada columna es una variable (posición en el perfil), cada fila una observación
                cov_mat = np.cov(profiles_array, rowvar=False)
                # Regularizar la matriz de covarianza añadiendo una pequeña identidad
                reg_cov_mat = cov_mat + COV_REGULARIZATION_LAMBDA * np.identity(PROFILE_LENGTH)

                # Calcular la inversa de la matriz de covarianza regularizada
                try:
                    inv_cov_mat = inv(reg_cov_mat)
                except np.linalg.LinAlgError:
                    logger.warning(f"Matriz de covarianza singular para landmark {i} incluso con regularización. Usando pseudo-inversa (pinv).")
                    inv_cov_mat = pinv(reg_cov_mat)

            # Guardar el modelo (media e inversa de covarianza) para este landmark
            profile_models.append({'mean': mean_prof, 'inv_cov': inv_cov_mat})

        except Exception as e:
             logger.error(f"Error calculando modelo estadístico para landmark {i}: {e}", exc_info=True)
             # Añadir un placeholder si falla el cálculo para no romper la estructura
             profile_models.append({'mean': np.zeros(PROFILE_LENGTH), 'inv_cov': np.identity(PROFILE_LENGTH) * 1e-9})

    # Guardar todos los modelos calculados en un archivo .npz
    if len(profile_models) == NUM_LANDMARKS:
        logger.info(f"Guardando {len(profile_models)} modelos de perfil en: {PROFILE_MODELS_OUTPUT_PATH}")
        try:
             np.savez(PROFILE_MODELS_OUTPUT_PATH, models=profile_models)
             logger.info("Modelos guardados exitosamente.")
        except Exception as e:
             logger.error(f"Error al guardar el archivo de modelos {PROFILE_MODELS_OUTPUT_PATH}: {e}")
    else:
        logger.error(f"El número de modelos generados ({len(profile_models)}) no coincide con NUM_LANDMARKS ({NUM_LANDMARKS}). No se guardó el archivo.")

    logger.info(f"--- Construcción de Modelos de Perfil (Norm: {NORMALIZE_PROFILE_METHOD.upper()}) Finalizada ---")

if __name__ == "__main__":
    build_profile_models()