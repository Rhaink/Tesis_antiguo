# Tesis/alineamiento/visualize_interpolation.py

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg') # Usar backend no interactivo
import matplotlib.pyplot as plt
import random
import logging

# --- Ajuste de Rutas ---
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ALINEAMIENTO_DIR = CURRENT_SCRIPT_DIR # Asume que está en alineamiento/
    if ALINEAMIENTO_DIR not in sys.path: sys.path.append(ALINEAMIENTO_DIR)
    BASE_PROJECT_DIR = os.path.dirname(ALINEAMIENTO_DIR)
    if BASE_PROJECT_DIR not in sys.path: sys.path.append(BASE_PROJECT_DIR)
    SRC_DIR = os.path.join(ALINEAMIENTO_DIR, "src")
    if SRC_DIR not in sys.path: sys.path.append(SRC_DIR)

    from src.data_loader import load_all_data # Importar después de ajustar path
except ImportError:
     print("Error importando desde src. Asegúrate de ejecutar desde Tesis/alineamiento/")
     exit()
except NameError:
     print("Ajustando sys.path manualmente")
     # Añadir rutas absolutas si es necesario
     # sys.path.append('/ruta/a/Tesis/alineamiento/src')
     from src.data_loader import load_all_data

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- Rutas y Configuración ---
RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
PLOTS_DIR = os.path.join(ALINEAMIENTO_DIR, "plots")
Tesis_DIR = os.path.dirname(ALINEAMIENTO_DIR) # Directorio Tesis/

# Archivos a cargar
TRAIN_INDICES_PATH = os.path.join(RESULTS_DIR, 'train_indices.txt')
INTERPOLATED_144_PATH = os.path.join(RESULTS_DIR, 'landmarks_train_144pts_raw.npy')

# Parámetros de Visualización
NUM_SAMPLES_TO_VISUALIZE = 5
LANDMARK_ORIGINAL_SIZE = 64 # Coordenadas originales están en 64x64
ORIGINAL_PTS_COLOR = (255, 0, 0)   # Azul para los 15 originales
INTERPOLATED_PTS_COLOR = (0, 0, 255) # Rojo para los 144 interpolados
CONTOUR_COLOR = (0, 255, 0)      # Verde para líneas de contorno
ORIGINAL_PTS_RADIUS = 5
INTERPOLATED_PTS_RADIUS = 2

os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Función Auxiliar para Dibujar (Adaptada de test_fitter) ---
def draw_points(image, points, color, radius, thickness=-1):
    """Dibuja puntos en una imagen."""
    vis_image = image
    if points is None: return vis_image
    img_h, img_w = vis_image.shape[:2]
    count = 0
    for i in range(points.shape[0]):
        try:
            if np.isnan(points[i, 0]) or np.isnan(points[i, 1]): continue
            x, y = int(round(points[i, 0])), int(round(points[i, 1]))
            if 0 <= x < img_w and 0 <= y < img_h:
                cv2.circle(vis_image, (x, y), radius, color, thickness)
                count += 1
        except Exception as e:
            logger.warning(f"Error dibujando punto {i}: {e}")
    # logger.debug(f"Dibujados {count}/{points.shape[0]} puntos.")
    return vis_image

def draw_contours(image, points_144, color, thickness=1):
    """Dibuja los contornos conectados para los 144 puntos (72 por pulmón)."""
    vis_image = image
    if points_144 is None or points_144.shape != (144, 2): return vis_image
    img_h, img_w = vis_image.shape[:2]

    # Contorno 1 (puntos 0 a 71)
    pts1 = points_144[0:72, :].round().astype(np.int32)
    # Filtrar puntos fuera de la imagen para polylines
    pts1_valid = []
    for p in pts1:
        if 0 <= p[0] < img_w and 0 <= p[1] < img_h:
            pts1_valid.append(p)
    if len(pts1_valid) > 1:
        cv2.polylines(vis_image, [np.array(pts1_valid)], isClosed=False, color=color, thickness=thickness)

    # Contorno 2 (puntos 72 a 143)
    pts2 = points_144[72:144, :].round().astype(np.int32)
    pts2_valid = []
    for p in pts2:
        if 0 <= p[0] < img_w and 0 <= p[1] < img_h:
            pts2_valid.append(p)
    if len(pts2_valid) > 1:
         cv2.polylines(vis_image, [np.array(pts2_valid)], isClosed=False, color=color, thickness=thickness)

    return vis_image

# --- Función Principal de Visualización ---
def main():
    logger.info("--- Iniciando Visualización de Interpolación (15 vs 144 puntos) ---")

    # 1. Cargar datos necesarios
    logger.info("Cargando datos...")
    index_map, landmarks_orig_15_all, image_paths = load_all_data(Tesis_DIR)
    if landmarks_orig_15_all is None or not image_paths:
        logger.error("Fallo al cargar datos originales. Abortando."); return

    if not os.path.exists(TRAIN_INDICES_PATH):
        logger.error(f"No encontrado: {TRAIN_INDICES_PATH}"); return
    train_indices = np.loadtxt(TRAIN_INDICES_PATH, dtype=int)
    if train_indices.ndim == 0: train_indices = np.array([train_indices.item()])

    if not os.path.exists(INTERPOLATED_144_PATH):
        logger.error(f"No encontrado: {INTERPOLATED_144_PATH}. Ejecute generate_144pt_data.py primero."); return
    try:
        landmarks_train_144pts = np.load(INTERPOLATED_144_PATH)
        logger.info(f"Cargado array de 144 puntos interpolados con shape: {landmarks_train_144pts.shape}")
    except Exception as e:
        logger.error(f"Error cargando {INTERPOLATED_144_PATH}: {e}"); return

    # Validar consistencia
    if len(train_indices) != landmarks_train_144pts.shape[0]:
        logger.error(f"Discrepancia: {len(train_indices)} índices de entrenamiento vs {landmarks_train_144pts.shape[0]} shapes de 144 puntos.")
        # Podríamos intentar continuar con el mínimo, pero es mejor parar si hay error.
        return

    # 2. Seleccionar muestras aleatorias del CONJUNTO DE ENTRENAMIENTO
    num_available = len(train_indices)
    num_to_plot = min(NUM_SAMPLES_TO_VISUALIZE, num_available)
    if num_available == 0: logger.error("No hay índices de entrenamiento disponibles."); return

    # Obtener índices de las posiciones en el array de entrenamiento (0 a N_train-1)
    sample_positions_in_train_array = random.sample(range(num_available), num_to_plot)

    logger.info(f"Visualizando {num_to_plot} muestras aleatorias del conjunto de entrenamiento...")

    for i, pos in enumerate(sample_positions_in_train_array):
        # Obtener el índice original de la imagen
        original_idx = train_indices[pos]
        logger.info(f"\nProcesando muestra {i+1}/{num_to_plot} (Posición en array ent.: {pos}, Índice original: {original_idx})...")

        # Obtener datos para esta muestra
        landmarks_15 = landmarks_orig_15_all[original_idx] # Obtener los 15 originales
        landmarks_144 = landmarks_train_144pts[pos]      # Obtener los 144 interpolados correspondientes

        image_path = image_paths.get(original_idx)
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"No se encontró imagen para índice {original_idx}. Saltando muestra.")
            continue

        # Cargar imagen
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR) # Cargar en color
            if image is None: raise IOError("imread devolvió None")
            H, W = image.shape[:2]
        except Exception as e:
            logger.error(f"Error cargando imagen {image_path}: {e}"); continue

        # Escalar puntos a dimensiones de la imagen
        scale_x = W / float(LANDMARK_ORIGINAL_SIZE)
        scale_y = H / float(LANDMARK_ORIGINAL_SIZE)

        landmarks_15_scaled = landmarks_15.astype(np.float64).copy()
        landmarks_15_scaled[:, 0] *= scale_x
        landmarks_15_scaled[:, 1] *= scale_y

        landmarks_144_scaled = landmarks_144.astype(np.float64).copy()
        landmarks_144_scaled[:, 0] *= scale_x
        landmarks_144_scaled[:, 1] *= scale_y

        # Dibujar en la imagen
        vis_image = image.copy()
        # Dibujar contornos interpolados primero (líneas verdes)
        vis_image = draw_contours(vis_image, landmarks_144_scaled, CONTOUR_COLOR, thickness=1)
        # Dibujar puntos originales (círculos azules grandes)
        vis_image = draw_points(vis_image, landmarks_15_scaled, ORIGINAL_PTS_COLOR, ORIGINAL_PTS_RADIUS)
        # Dibujar puntos interpolados (puntos rojos pequeños)
        vis_image = draw_points(vis_image, landmarks_144_scaled, INTERPOLATED_PTS_COLOR, INTERPOLATED_PTS_RADIUS)

        # Guardar imagen
        img_filename = os.path.basename(image_path)
        output_filename = f"interpolated_vs_original_idx{original_idx}.png"
        output_filepath = os.path.join(PLOTS_DIR, output_filename)
        try:
            cv2.imwrite(output_filepath, vis_image)
            logger.info(f"Visualización guardada en: {output_filepath}")
        except Exception as e:
            logger.error(f"Error guardando imagen de visualización: {e}")

    logger.info("--- Visualización de Interpolación Finalizada ---")

if __name__ == "__main__":
    main()