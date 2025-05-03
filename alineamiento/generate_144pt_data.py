# Tesis/alineamiento/generate_144pt_data.py

import os
import sys
import numpy as np
import logging
from scipy.interpolate import splev, splprep # Necesario para la función

# --- Ajuste de Rutas ---
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ALINEAMIENTO_DIR = CURRENT_SCRIPT_DIR # Asume que está en alineamiento/
    if ALINEAMIENTO_DIR not in sys.path: sys.path.append(ALINEAMIENTO_DIR)
    BASE_PROJECT_DIR = os.path.dirname(ALINEAMIENTO_DIR)
    if BASE_PROJECT_DIR not in sys.path: sys.path.append(BASE_PROJECT_DIR)
    # Añadir src si data_loader está ahí
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
logger_script = logging.getLogger(__name__)
logger_interp = logging.getLogger(__name__ + ".interpolation") # Logger para la función

# --- Rutas ---
RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
Tesis_DIR = os.path.dirname(ALINEAMIENTO_DIR) # Directorio Tesis/

# Archivo de entrada (15 landmarks) y salida (144 puntos)
TRAIN_INDICES_PATH = os.path.join(RESULTS_DIR, 'train_indices.txt')
OUTPUT_144PTS_PATH = os.path.join(RESULTS_DIR, 'landmarks_train_144pts_raw.npy') # Guardar los puntos raw interpolados

# --- Función de Interpolación (copiada de arriba) ---
def interpolate_shape_to_144(shape_15pts):
    # ... (pegar aquí el código de la función interpolate_shape_to_144 definida antes) ...
    if shape_15pts is None or shape_15pts.shape != (15, 2):
        logger_interp.error(f"Input shape inválido: {shape_15pts.shape if shape_15pts is not None else 'None'}. Se requieren (15, 2).")
        return None
    # Índices 0-based
    # Contorno 1 (ej. Izq): 1 -> 12 -> 3 -> 5 -> 7 -> 14 -> 2
    seq1_indices = [0, 11, 2, 4, 6, 13, 1]
    # Contorno 2 (ej. Der):  1 -> 13 -> 4 -> 6 -> 8 -> 15 -> 2
    seq2_indices = [0, 12, 3, 5, 7, 14, 1] # CORREGIDO: idx 8->7, 15->14

    contour1_pts = shape_15pts[seq1_indices, :]
    contour2_pts = shape_15pts[seq2_indices, :]
    num_points_per_contour = 72
    interpolated_points_all = []

    for contour_pts in [contour1_pts, contour2_pts]:
        try:
            tck, u = splprep([contour_pts[:, 0], contour_pts[:, 1]], s=0, k=3, per=0)
            u_new = np.linspace(u.min(), u.max(), num_points_per_contour)
            x_new, y_new = splev(u_new, tck, der=0)
            interpolated_contour = np.vstack((x_new, y_new)).T
            interpolated_points_all.append(interpolated_contour)
        except Exception as e:
            logger_interp.error(f"Error interpolación spline: {e}")
            return None

    if len(interpolated_points_all) == 2:
        final_144_shape = np.concatenate(interpolated_points_all, axis=0)
        if final_144_shape.shape == (144, 2):
            return final_144_shape
        else:
            logger_interp.error(f"Shape final inesperado: {final_144_shape.shape}")
            return None
    else:
        logger_interp.error("No se interpolaron ambos contornos.")
        return None

# --- Función Principal ---
def main():
    logger_script.info("--- Iniciando Generación de Datos de 144 Puntos mediante Interpolación ---")

    # 1. Cargar datos originales (15 landmarks)
    logger_script.info("Cargando datos originales (15 landmarks)...")
    # Necesitamos BASE_DIR = Directorio 'Tesis' para load_all_data
    index_map, landmarks_array_orig_15, _ = load_all_data(Tesis_DIR)
    if landmarks_array_orig_15 is None:
        logger_script.error("Fallo al cargar los landmarks originales de 15 puntos. Abortando.")
        return
    if landmarks_array_orig_15.shape[1] != 15:
        logger_script.error(f"Los landmarks cargados no tienen 15 puntos (shape={landmarks_array_orig_15.shape}). Abortando.")
        return
    logger_script.info(f"Cargados {landmarks_array_orig_15.shape[0]} shapes de 15 landmarks.")

    # 2. Cargar índices de entrenamiento
    logger_script.info(f"Cargando índices de entrenamiento desde {TRAIN_INDICES_PATH}...")
    if not os.path.exists(TRAIN_INDICES_PATH):
        logger_script.error(f"No encontrado: {TRAIN_INDICES_PATH}. Ejecute prepare_splits.py."); return
    try:
        train_indices = np.loadtxt(TRAIN_INDICES_PATH, dtype=int)
        if train_indices.ndim == 0: train_indices = np.array([train_indices.item()])
        # Validar índices
        max_index = landmarks_array_orig_15.shape[0] - 1
        valid_train_indices = [idx for idx in train_indices if 0 <= idx <= max_index]
        if len(valid_train_indices) < len(train_indices):
             logger_script.warning(f"Descartados {len(train_indices) - len(valid_train_indices)} índices inválidos.")
        if not valid_train_indices: raise ValueError("No quedan índices válidos.")
        train_indices = np.array(valid_train_indices)
        logger_script.info(f"Se procesarán {len(train_indices)} muestras de entrenamiento.")
    except Exception as e: logger_script.error(f"Error cargando/validando índices: {e}"); return

    # 3. Seleccionar landmarks de entrenamiento (15 puntos)
    try:
        landmarks_train_15pts = landmarks_array_orig_15[train_indices]
    except IndexError: logger_script.error("Error de índice al seleccionar landmarks de 15 puntos."); return

    # 4. Interpolar cada forma de entrenamiento a 144 puntos
    landmarks_train_144pts_list = []
    processed_count = 0
    error_count = 0
    logger_script.info("Iniciando interpolación para cada forma de entrenamiento...")
    for i in range(landmarks_train_15pts.shape[0]):
        shape_15 = landmarks_train_15pts[i]
        shape_144 = interpolate_shape_to_144(shape_15)
        if shape_144 is not None:
            landmarks_train_144pts_list.append(shape_144)
            processed_count += 1
        else:
            logger_script.warning(f"Fallo la interpolación para la muestra índice original {train_indices[i]}. Se omitirá.")
            error_count += 1
        if (i + 1) % 50 == 0:
             logger_script.info(f" Interpoladas {i+1}/{len(landmarks_train_15pts)} formas...")

    logger_script.info(f"Interpolación completada. {processed_count} formas generadas, {error_count} errores.")

    # 5. Guardar el resultado
    if not landmarks_train_144pts_list:
        logger_script.error("No se generó ninguna forma de 144 puntos. No se guardará archivo.")
        return

    try:
        landmarks_train_144pts_array = np.array(landmarks_train_144pts_list)
        logger_script.info(f"Shape final del array de 144 puntos: {landmarks_train_144pts_array.shape}") # Debería ser (N_train_procesadas, 144, 2)
        logger_script.info(f"Guardando array en: {OUTPUT_144PTS_PATH}")
        np.save(OUTPUT_144PTS_PATH, landmarks_train_144pts_array)
        logger_script.info("Archivo guardado exitosamente.")

        # Guardar también los índices correspondientes a las formas que SÍ se procesaron
        # (Necesitaríamos rastrear qué índices originales corresponden a las formas en landmarks_train_144pts_list)
        # Por simplicidad ahora, no lo hacemos, pero sería importante si hubo errores.

    except Exception as e:
        logger_script.error(f"Error al convertir a array o guardar archivo: {e}")

    logger_script.info("--- Generación de Datos de 144 Puntos Finalizada ---")

if __name__ == "__main__":
    main()