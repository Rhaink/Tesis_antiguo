# Tesis/alineamiento/test_fitter.py
# CONFIGURACIÓN BASE P2: Usa SSMFitter (con normales estables), Escala Fija, Sin CLAHE.
# Asume que SSMFitter y profile_builder ya usan Z-SCORE si se han modificado.

import os
import numpy as np
import cv2 # Importar OpenCV
import logging
import random
import time # Para medir tiempo

# --- Configuración Logging ---
# Configurar para mostrar mensajes INFO y superiores. DEBUG es muy verboso para pruebas normales.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s',
                    handlers=[logging.StreamHandler()]) # Mostrar logs en consola
logger_global = logging.getLogger() # Obtener logger raíz

# Importar módulos desde src (asumiendo que test_fitter.py está en Tesis/alineamiento/)
try:
    from src.data_loader import load_all_data
    # ¡Asegúrese que SSMFitter es la versión que usa Z-Score y Normales Estables!
    from src.ssm_fitter import SSMFitter
except ImportError as e:
    logger_global.error(f"Error importando módulos desde 'src': {e}")
    logger_global.error("Asegúrese de ejecutar este script desde el directorio 'Tesis/alineamiento/'")
    exit(1)
except Exception as e:
    logger_global.error(f"Otro error durante la importación: {e}")
    exit(1)

# ================================================================
# --- Configuración de Rutas ---
# Asume que este script se ejecuta desde Tesis/alineamiento/
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = CURRENT_SCRIPT_DIR # Directorio Tesis/alineamiento/
    SRC_DIR = os.path.join(BASE_DIR, "src")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
    Tesis_BASE_DIR = os.path.dirname(BASE_DIR) # Directorio Tesis/

    # Rutas a los archivos generados por los builders
    MEAN_VECTOR_PATH = os.path.join(RESULTS_DIR, 'pca_mean_vector.npy')
    COMPONENTS_PATH = os.path.join(RESULTS_DIR, 'pca_components.npy')
    STD_DEVS_PATH = os.path.join(RESULTS_DIR, 'pca_std_devs.npy')
    PROFILE_MODELS_PATH = os.path.join(RESULTS_DIR, 'asm_profile_models.npz') # Cargará los modelos Z-score después de ejecutar profile_builder
    TEST_INDICES_PATH = os.path.join(RESULTS_DIR, 'test_indices.txt')

    # Verificar existencia de directorios necesarios
    os.makedirs(PLOTS_DIR, exist_ok=True)
    if not os.path.exists(RESULTS_DIR):
         logger_global.error(f"El directorio de resultados no existe: {RESULTS_DIR}")
         exit(1)

except NameError: # Para ejecución interactiva
    logger_global.warning("Variable __file__ no definida (¿ejecución interactiva?). Definiendo rutas manualmente.")
    BASE_DIR = '.' # Directorio actual
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
    Tesis_BASE_DIR = os.path.dirname(BASE_DIR)
    # Definir rutas a archivos explícitamente
    MEAN_VECTOR_PATH = os.path.join(RESULTS_DIR, 'pca_mean_vector.npy')
    COMPONENTS_PATH = os.path.join(RESULTS_DIR, 'pca_components.npy')
    STD_DEVS_PATH = os.path.join(RESULTS_DIR, 'pca_std_devs.npy')
    PROFILE_MODELS_PATH = os.path.join(RESULTS_DIR, 'asm_profile_models.npz')
    TEST_INDICES_PATH = os.path.join(RESULTS_DIR, 'test_indices.txt')
    os.makedirs(PLOTS_DIR, exist_ok=True)
except Exception as e:
    logger_global.error(f"Error configurando rutas: {e}")
    exit(1)
# ----------------------------------------------------

# --- Parámetros Generales ---
LANDMARK_ORIGINAL_SIZE = 64 # Tamaño en el que se anotaron originalmente los landmarks (ej. 64x64)
GT_COLOR = (0, 255, 0) # Verde para Ground Truth
PRED_COLOR = (0, 0, 255) # Rojo para Predicción
PRED_RADIUS = 4
NUM_LANDMARKS = 15
NUM_DIMS = 2
# ----------------------------

# --- Parámetros para fit() [CONFIGURACIÓN P2: Normales Estables + Escala Fija] ---
FIT_MAX_ITERS = 100
FIT_TOLERANCE = 0.05
FIT_SEARCH_PIXELS = 10        # Rango de búsqueda local (+/- 10 píxeles)
FIT_GAUSSIAN_KSIZE = 3      # Kernel Gaussiano para suavizar gradiente en búsqueda local
FIT_DAMPING_THETA = 0.5     # Amortiguamiento para rotación
FIT_DAMPING_T = 0.5         # Amortiguamiento para traslación
FIT_DAMPING_S = 0.0         # <-- ¡ESCALA FIJA! (Parámetro clave de P2)
FIT_CLAMP_B = 3.0           # Limitar parámetros b a +/- 3 std dev
# ---------------------------

# --- Funciones Auxiliares (draw_landmarks, calculate_mean_point_error) ---
def draw_landmarks(image, landmarks, color, radius=3, thickness=-1, marker_type=cv2.MARKER_STAR):
    """Dibuja landmarks en una imagen. Modificado para robustez."""
    local_logger = logging.getLogger(__name__ + ".draw_landmarks")
    if image is None:
        local_logger.error("Imagen de entrada es None.")
        # Devolver una imagen negra pequeña para evitar crash posterior
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Asegurarse que la imagen sea BGR para dibujar en color
    vis_image = image.copy()
    if len(vis_image.shape) == 2: # Si es escala de grises
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    elif vis_image.shape[2] == 1: # Si tiene un solo canal
         vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    elif vis_image.shape[2] != 3:
         local_logger.error(f"Formato de imagen no soportado para dibujar: shape={vis_image.shape}")
         return vis_image # Devolver sin cambios

    img_h, img_w = vis_image.shape[:2]

    if landmarks is not None and landmarks.ndim == 2 and landmarks.shape[0] > 0 and landmarks.shape[1] == NUM_DIMS:
        if np.isnan(landmarks).any() or np.isinf(landmarks).any():
            local_logger.error("Se detectaron coordenadas NaN/Inf en landmarks. No se dibujarán.")
            return vis_image

        count = 0
        for i in range(landmarks.shape[0]):
            try:
                # Extraer y redondear coordenadas
                x, y = int(round(landmarks[i, 0])), int(round(landmarks[i, 1]))

                # Dibujar solo si está dentro de los límites de la imagen
                if 0 <= x < img_w and 0 <= y < img_h:
                    if marker_type == cv2.MARKER_CROSS:
                        cv2.drawMarker(vis_image, (x, y), color, marker_type, markerSize=10, thickness=1)
                    elif marker_type == cv2.MARKER_STAR:
                         cv2.drawMarker(vis_image, (x, y), color, marker_type, markerSize=10, thickness=1)
                    elif marker_type == cv2.MARKER_DIAMOND:
                         cv2.drawMarker(vis_image, (x, y), color, marker_type, markerSize=8, thickness=1)
                    else: # Círculo por defecto
                        cv2.circle(vis_image, (x, y), radius, color, thickness)
                    count += 1
                # else: # Opcional: advertir si un punto está fuera
                #    local_logger.warning(f"Landmark {i} ({x},{y}) fuera de los límites de la imagen ({img_w}x{img_h}). No dibujado.")

            except Exception as e:
                local_logger.error(f"Error dibujando landmark {i} en ({landmarks[i, 0]},{landmarks[i, 1]}): {e}")
        # local_logger.debug(f"Dibujados {count}/{landmarks.shape[0]} landmarks.")
    elif landmarks is None:
         local_logger.warning("Los landmarks proporcionados son None.")
    else:
        local_logger.error(f"Formato de landmarks incorrecto: esperado ({NUM_LANDMARKS}, {NUM_DIMS}), obtenido {landmarks.shape if landmarks is not None else 'None'}")

    return vis_image

def calculate_mean_point_error(landmarks_pred, landmarks_gt):
    """Calcula el error medio por punto (Euclidiano) entre dos conjuntos de landmarks."""
    local_logger = logging.getLogger(__name__ + ".calculate_mpe")
    if landmarks_pred is None or landmarks_gt is None:
        local_logger.error("MPE: Al menos uno de los conjuntos de landmarks es None.")
        return float('inf')
    if landmarks_pred.shape != landmarks_gt.shape:
        local_logger.error(f"MPE: Shapes no coinciden {landmarks_pred.shape} vs {landmarks_gt.shape}")
        return float('inf')
    if landmarks_pred.ndim != 2 or landmarks_pred.shape[1] != NUM_DIMS:
         local_logger.error(f"MPE: Dimensiones incorrectas: {landmarks_pred.shape}")
         return float('inf')
    if landmarks_pred.shape[0] == 0: # Si no hay landmarks
        local_logger.warning("MPE: Conjuntos de landmarks vacíos.")
        return 0.0

    # Verificar NaNs o Infs que invalidarían el cálculo
    if np.isnan(landmarks_pred).any() or np.isnan(landmarks_gt).any() or \
       np.isinf(landmarks_pred).any() or np.isinf(landmarks_gt).any():
        local_logger.error("MPE: NaN/Inf detectado en las coordenadas.")
        return float('inf')

    # Calcular distancias Euclidianas punto a punto
    point_errors = np.linalg.norm(landmarks_pred - landmarks_gt, axis=1)
    # Calcular la media de estas distancias
    mean_error = np.mean(point_errors)
    return mean_error
# ---------------------------------------------------------------

def main():
    """Función principal para ejecutar el test del fitter."""
    logger = logging.getLogger(__name__) # Logger específico para main
    logger.info(f"Usando RESULTS_DIR: {RESULTS_DIR}")
    logger.info(f"Usando PLOTS_DIR: {PLOTS_DIR}")
    logger.info("--- Iniciando Prueba del Fitter SSM [Configuración P2 Base: Normales Estables, Escala Fija, Sin CLAHE] ---")

    # --- Carga de Datos Generales ---
    logger.info("Cargando datos generales usando data_loader...")
    # Pasar Tesis_BASE_DIR para que data_loader sepa dónde buscar 'indices', 'coordenadas', etc.
    index_map, landmarks_array_orig, image_paths_dict = load_all_data(Tesis_BASE_DIR)
    if index_map is None or landmarks_array_orig is None or image_paths_dict is None:
        logger.error("Fallo al cargar datos iniciales usando load_all_data. Abortando.")
        return
    logger.info(f"Cargados datos para {len(index_map)} índices.")

    # --- Carga de Índices de Prueba ---
    logger.info(f"Cargando índices de prueba desde {TEST_INDICES_PATH}...")
    if not os.path.exists(TEST_INDICES_PATH):
        logger.error(f"Archivo de índices de prueba no encontrado: {TEST_INDICES_PATH}")
        logger.error("Ejecute 'prepare_splits.py' primero si es necesario.")
        return
    try:
        test_indices = np.loadtxt(TEST_INDICES_PATH, dtype=int)
        if test_indices.ndim == 0: # Manejar caso de un solo índice en el archivo
            test_indices = np.array([test_indices.item()])

        # Filtrar para asegurar que los índices de prueba estén en nuestro mapa cargado
        valid_test_indices = [idx for idx in test_indices if idx in index_map.index]
        if len(valid_test_indices) < len(test_indices):
            logger.warning(f"Se descartaron {len(test_indices) - len(valid_test_indices)} índices de prueba no encontrados en el index_map.")
        if not valid_test_indices:
            raise ValueError("No hay índices de prueba válidos disponibles en el index_map.")
        test_indices = valid_test_indices
        logger.info(f"Se usarán {len(test_indices)} índices de prueba válidos.")
    except ValueError as e:
         logger.error(f"Error validando índices de prueba: {e}")
         return
    except Exception as e:
        logger.error(f"Error cargando o procesando índices de prueba desde {TEST_INDICES_PATH}: {e}", exc_info=True)
        return

    # --- Seleccionar Imagen de Prueba ---
    # Usar el índice 790 como en el informe, si está disponible en el conjunto de prueba válido
    target_test_index = 790
    if target_test_index in test_indices:
        test_new_index = target_test_index
        logger.info(f"Usando índice de prueba objetivo: {test_new_index}")
    else:
        test_new_index = random.choice(test_indices) # O elegir uno aleatorio si 790 no está
        logger.warning(f"Índice objetivo {target_test_index} no encontrado en la lista de prueba válida. Usando uno aleatorio: {test_new_index}")

    # Obtener ruta de imagen y landmarks GT para el índice seleccionado
    test_image_path = image_paths_dict.get(test_new_index)
    if test_image_path is None or not os.path.exists(test_image_path):
        logger.error(f"No se encontró ruta válida o archivo para el índice de prueba {test_new_index}. Abortando.")
        return

    try:
        # Asumiendo que landmarks_array_orig está indexado 0..N-1 y corresponde a index_map.index
        # Necesitamos obtener la posición del test_new_index en el index_map para acceder a landmarks_array_orig
        # O si landmarks_array_orig está indexado directamente por new_index (ej. un diccionario), usarlo.
        # Supongamos que está indexado 0..N-1 alineado con index_map
        idx_pos = index_map.index.get_loc(test_new_index)
        landmarks_gt_orig = landmarks_array_orig[idx_pos]
    except (IndexError, KeyError):
        logger.error(f"Error obteniendo GT landmarks para índice {test_new_index}. Verificar alineación entre index_map y landmarks_array_orig.")
        return

    logger.info(f"Cargando imagen de prueba: {test_image_path}")
    original_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        logger.error(f"No se pudo cargar la imagen de prueba: {test_image_path}")
        return
    H, W = original_image.shape
    logger.info(f"Imagen cargada: {W}x{H} píxeles.")

    # --- Preprocesamiento de Imagen: NINGUNO (Configuración P2 Base) ---
    test_image_processed = original_image.copy() # Usar la imagen original directamente
    logger.info("No se aplica preprocesamiento adicional a la imagen (Configuración P2 Base).")

    # --- Escalar Landmarks GT al tamaño de la imagen ---
    # Los landmarks originales estaban en LANDMARK_ORIGINAL_SIZE x LANDMARK_ORIGINAL_SIZE
    logger.info(f"Escalando GT landmarks desde {LANDMARK_ORIGINAL_SIZE}x{LANDMARK_ORIGINAL_SIZE} a {W}x{H}...")
    scale_x = W / float(LANDMARK_ORIGINAL_SIZE)
    scale_y = H / float(LANDMARK_ORIGINAL_SIZE)
    landmarks_gt_scaled = landmarks_gt_orig.astype(np.float64).copy()
    landmarks_gt_scaled[:, 0] *= scale_x
    landmarks_gt_scaled[:, 1] *= scale_y
    # Verificar si algún GT quedó fuera de la imagen después de escalar (raro pero posible)
    if np.any(landmarks_gt_scaled[:, 0] < 0) or np.any(landmarks_gt_scaled[:, 0] >= W) or \
       np.any(landmarks_gt_scaled[:, 1] < 0) or np.any(landmarks_gt_scaled[:, 1] >= H):
        logger.warning("Algunos landmarks GT escalados quedaron fuera de los límites de la imagen.")
        # Opcional: aplicar clip a GT también, aunque usualmente no se hace
        # landmarks_gt_scaled[:, 0] = np.clip(landmarks_gt_scaled[:, 0], 0, W - 1)
        # landmarks_gt_scaled[:, 1] = np.clip(landmarks_gt_scaled[:, 1], 0, H - 1)
    logger.info("GT landmarks escalados a tamaño de imagen.")


    # --- Instanciar Fitter ---
    logger.info("Instanciando SSMFitter...")
    # Verificar que existan los archivos necesarios para el fitter
    required_files = [MEAN_VECTOR_PATH, COMPONENTS_PATH, STD_DEVS_PATH, PROFILE_MODELS_PATH]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        logger.error(f"Faltan archivos necesarios para SSMFitter en {RESULTS_DIR}: {missing_files}")
        logger.error("Asegúrese de haber ejecutado run_gpa.py, ssm_builder.py y profile_builder.py (con Z-score).")
        return
    try:
        fitter = SSMFitter(
            mean_vector_path=MEAN_VECTOR_PATH,
            components_path=COMPONENTS_PATH,
            std_devs_path=STD_DEVS_PATH,
            profile_models_path=PROFILE_MODELS_PATH, # Cargará los modelos Z-score
            num_landmarks=NUM_LANDMARKS,
            num_dims=NUM_DIMS
        )
        logger.info("SSMFitter instanciado correctamente.")
        # Loggear el método de normalización que usará el fitter
        logger.info(f"Fitter usará normalización de perfiles: {fitter.profile_norm_method.upper()}")
    except Exception as e:
        logger.error(f"Error crítico instanciando SSMFitter: {e}", exc_info=True)
        return

    # --- Ejecutar Fitting ---
    logger.info(f"Ejecutando fitter.fit en imagen índice {test_new_index} [Estrategia: P2 Base]...")
    start_time = time.time()
    try:
        # Llamar a fit() con la configuración P2
        # Dejar que fit() calcule la traslación inicial para centrar
        predicted_landmarks_img, final_b, final_pose = fitter.fit(
            image=test_image_processed,       # Imagen original (sin CLAHE)
            initial_s=1.0,                    # Escala inicial (se mantendrá fija)
            initial_theta_rad=0.0,            # Rotación inicial 0
            initial_tx=None,                  # Dejar que se auto-centre
            initial_ty=None,                  # Dejar que se auto-centre
            initial_b=None,                   # Empezar desde la forma media (b=0)
            max_iters=FIT_MAX_ITERS,
            tolerance=FIT_TOLERANCE,
            search_pixels_per_iter=FIT_SEARCH_PIXELS,
            damping_factor_theta=FIT_DAMPING_THETA,
            damping_factor_t=FIT_DAMPING_T,
            damping_factor_s=FIT_DAMPING_S,     # <-- ESCALA FIJA (0.0)
            gaussian_ksize_per_iter=FIT_GAUSSIAN_KSIZE,
            clamp_n_std_devs=FIT_CLAMP_B
        )
        end_time = time.time()
        fitting_time = end_time - start_time
        logger.info(f"Fitting finalizado en {fitting_time:.2f} segundos.")

    except Exception as e:
        logger.error(f"Error durante la ejecución de fitter.fit: {e}", exc_info=True)
        predicted_landmarks_img, final_b, final_pose = None, None, None
        fitting_time = -1

    # --- Procesamiento y guardado de resultados ---
    vis_image_base = test_image_processed # Visualizar sobre la imagen original

    if predicted_landmarks_img is None:
        logger.error("Fitting falló o no devolvió resultados.")
        # Crear imagen de fallo
        vis_image = vis_image_base.copy() if vis_image_base is not None else np.zeros((H, W, 3), dtype=np.uint8)
        if len(vis_image.shape) == 2: vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        vis_image = draw_landmarks(vis_image, landmarks_gt_scaled, GT_COLOR, marker_type=cv2.MARKER_DIAMOND)
        cv2.putText(vis_image, "FITTER FAILED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        final_error = float('inf')
        output_filename = f"FAILED_fit_result_idx{test_new_index}_P2base.png"
    else:
        # Fitting exitoso (o al menos completado)
        # Asegurar que la pose final no sea None (si fit devuelve algo)
        if final_pose is None: final_pose = {'s': -1, 'theta_rad': -1, 'tx': -1, 'ty': -1}
        if final_b is None: final_b = np.array([-999])

        # Calcular error final
        final_error = calculate_mean_point_error(predicted_landmarks_img, landmarks_gt_scaled)
        logger.info(f"Error Final (MPE): {final_error:.3f} píxeles")

        # Generar visualización final
        logger.info("Generando visualización final...")
        vis_image = vis_image_base.copy()
        # Dibujar GT (diamantes verdes)
        vis_image = draw_landmarks(vis_image, landmarks_gt_scaled, GT_COLOR, radius=5, marker_type=cv2.MARKER_DIAMOND)
        # Dibujar Predicción (estrellas rojas)
        vis_image = draw_landmarks(vis_image, predicted_landmarks_img, PRED_COLOR, radius=PRED_RADIUS, marker_type=cv2.MARKER_STAR)

        # Añadir texto con información
        cv2.putText(vis_image, f"Error: {final_error:.2f} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis_image, f"Idx: {test_new_index}", (W - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        # Indicar configuración usada
        config_text = f"P2 Base (Norm:{fitter.profile_norm_method.upper()}, Scale:Fixed)"
        cv2.putText(vis_image, config_text, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # Nombre de archivo indicando configuración y error
        output_filename = f"fit_result_idx{test_new_index}_err{final_error:.1f}_P2base_Norm{fitter.profile_norm_method.upper()}.png"

    # Guardar la imagen de visualización
    output_path = os.path.join(PLOTS_DIR, output_filename)
    try:
        if 'vis_image' in locals() and vis_image is not None:
            cv2.imwrite(output_path, vis_image)
            logger.info(f"Visualización final guardada en: {output_path}")
        else:
            logger.error("No se pudo generar la imagen de visualización.")
    except Exception as e:
        logger.error(f"No se pudo guardar la imagen final en {output_path}: {e}", exc_info=True)

    logger.info(f"--- Prueba del Fitter SSM [Configuración P2 Base] Finalizada ---")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Captura final para errores no esperados en main
        logging.exception(f"Error fatal no capturado en main: {e}")