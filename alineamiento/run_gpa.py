# Tesis/alineamiento/run_gpa.py
# MODIFICADO para usar los datos interpolados de 144 puntos

import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Usar backend no interactivo
import matplotlib.pyplot as plt
import logging
from scipy.linalg import orthogonal_procrustes # Necesario para la función auxiliar

# Asumiendo ejecución desde Tesis/alineamiento/
try:
    # No necesitamos load_all_data aquí si cargamos directamente el .npy
    # from src.data_loader import load_all_data
    from src.alignment import ProcrustesAligner
except ImportError as e:
     print(f"Error importando módulos desde src: {e}")
     print("Asegúrese de ejecutar este script desde el directorio 'Tesis/alineamiento/'")
     exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')

# --- Configuración ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directorio alineamiento/
# Tesis_DIR = os.path.dirname(BASE_DIR) # Directorio Tesis/ # No necesario ahora
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# --- Archivos de Entrada y Salida ---
INPUT_144PTS_RAW_PATH = os.path.join(RESULTS_DIR, 'landmarks_train_144pts_raw.npy') # <-- ENTRADA
ALIGNED_LANDMARKS_OUTPUT_PATH = os.path.join(RESULTS_DIR, 'landmarks_train_aligned_144pts.npy') # <-- SALIDA ALINEADA
MEAN_SHAPE_OUTPUT_PATH = os.path.join(RESULTS_DIR, 'mean_shape_144pts.npy') # <-- SALIDA MEDIA
# MEAN_POSE_OUTPUT_PATH = os.path.join(RESULTS_DIR, 'mean_training_pose_144pts.npz') # Opcional: recalcular pose media si es necesario

# --- Función Auxiliar para Calcular Transformación (sin cambios) ---
def calculate_similarity_transform(shape_from, shape_to):
    # ... (pegar aquí el código de la función calculate_similarity_transform definida antes) ...
    if shape_from is None or shape_to is None or \
       shape_from.shape != shape_to.shape or shape_from.ndim != 2:
        logging.error("Inputs inválidos para calculate_similarity_transform.")
        return 1.0, 0.0, 0.0, 0.0 # Valores por defecto
    k, d = shape_from.shape
    centroid_from = np.mean(shape_from, axis=0)
    centered_from = shape_from - centroid_from
    centroid_to = np.mean(shape_to, axis=0)
    centered_to = shape_to - centroid_to
    try:
        R, scale_procrustes = orthogonal_procrustes(centered_from, centered_to)
    except Exception as e:
        logging.error(f"Error en orthogonal_procrustes: {e}. Usando identidad.")
        return 1.0, 0.0, 0.0, 0.0
    norm_from = np.linalg.norm(centered_from, 'fro')
    norm_to = np.linalg.norm(centered_to, 'fro')
    s_scale = norm_to / (norm_from + 1e-9)
    theta_rad = np.arctan2(R[1, 0], R[0, 0])
    translation = centroid_to - s_scale * (centroid_from @ R.T)
    tx, ty = translation[0], translation[1]
    return s_scale, theta_rad, tx, ty

# --- Funciones de Plotting (sin cambios) ---
def plot_aligned_landmarks(aligned_landmarks, mean_shape, output_path):
    # ... (pegar aquí el código de la función plot_aligned_landmarks definida antes) ...
    if aligned_landmarks is None or mean_shape is None: logging.error("No data for plotting aligned landmarks."); return
    N = aligned_landmarks.shape[0]; logging.info(f"Plotting {N} aligned landmarks (144 pts)...") # Mensaje actualizado
    plt.figure(figsize=(8, 8))
    for i in range(N): plt.plot(aligned_landmarks[i, :, 0], aligned_landmarks[i, :, 1], marker='.', linestyle='', ms=1, alpha=0.1, color='green') # Puntos más pequeños
    plt.plot(mean_shape[:, 0], mean_shape[:, 1], marker='o', linestyle='', ms=2, color='red', label='Forma Media GPA (144 pts)') # Label actualizado
    plt.title(f'Landmarks Alineados (144 pts) con GPA (N={N})'); plt.xlabel('X'); plt.ylabel('Y') # Título actualizado
    plt.gca().invert_yaxis(); plt.axis('equal'); plt.grid(True, linestyle='--', alpha=0.5); plt.legend()
    try: plt.savefig(output_path); logging.info(f"Aligned plot saved: {output_path}")
    except Exception as e: logging.error(f"Failed to save aligned plot: {e}")
    plt.close()

def plot_mean_shape(mean_shape, output_path):
    # ... (pegar aquí el código de la función plot_mean_shape definida antes) ...
    if mean_shape is None: logging.error("No data for plotting mean shape."); return
    logging.info("Plotting mean shape (144 pts)...") # Mensaje actualizado
    plt.figure(figsize=(8, 8))
    # Conectar solo los contornos por separado para mejor visualización
    if mean_shape.shape == (144, 2):
         plt.plot(mean_shape[0:72, 0], mean_shape[0:72, 1], marker='.', linestyle='-', ms=3, color='blue', label='Contorno 1')
         plt.plot(mean_shape[72:144, 0], mean_shape[72:144, 1], marker='.', linestyle='-', ms=3, color='cyan', label='Contorno 2')
    else: # Fallback si no tiene 144 puntos
         plt.plot(mean_shape[:, 0], mean_shape[:, 1], marker='o', linestyle='-', ms=5, color='blue')
    plt.title('Forma Media (144 pts) obtenida por GPA'); plt.xlabel('X'); plt.ylabel('Y') # Título actualizado
    plt.gca().invert_yaxis(); plt.axis('equal'); plt.grid(True, linestyle='--', alpha=0.5); plt.legend()
    try: plt.savefig(output_path); logging.info(f"Mean shape plot saved: {output_path}")
    except Exception as e: logging.error(f"Failed to save mean shape plot: {e}")
    plt.close()


def main():
    """Función principal para ejecutar GPA sobre los datos de 144 puntos."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    logging.info("--- Iniciando Fase 1: GPA (sobre 144 puntos) ---")

    # 1. Cargar datos interpolados (144 puntos)
    logging.info(f"Cargando landmarks interpolados desde: {INPUT_144PTS_RAW_PATH}")
    if not os.path.exists(INPUT_144PTS_RAW_PATH):
        logging.error(f"Archivo no encontrado: {INPUT_144PTS_RAW_PATH}. Ejecute generate_144pt_data.py primero.")
        return
    try:
        # Estos ya son solo los datos de entrenamiento
        landmarks_train_144pts_raw = np.load(INPUT_144PTS_RAW_PATH)
        logging.info(f"Landmarks raw de 144 puntos cargados. Shape: {landmarks_train_144pts_raw.shape}")
        if landmarks_train_144pts_raw.shape[1] != 144 or landmarks_train_144pts_raw.shape[2] != 2:
             raise ValueError("El shape cargado no es (N, 144, 2).")
    except Exception as e:
        logging.error(f"Error al cargar los datos de 144 puntos: {e}"); return

    # 2. Ejecutar GPA
    logging.info("Instanciando y ejecutando ProcrustesAligner en datos de 144 puntos...")
    aligner = ProcrustesAligner()
    # Pasar directamente los datos cargados
    landmarks_train_aligned, mean_shape = aligner.gpa(landmarks_train_144pts_raw, max_iters=100, tolerance=1e-7)

    if landmarks_train_aligned is None or mean_shape is None:
        logging.error("La ejecución de GPA sobre 144 puntos falló."); return
    logging.info(f"GPA (144 pts) completado. Shape alineados: {landmarks_train_aligned.shape}, Shape media: {mean_shape.shape}")

    # 3. Guardar resultados numéricos del GPA (144 puntos)
    logging.info(f"Guardando landmarks alineados (144 pts) en: {ALIGNED_LANDMARKS_OUTPUT_PATH}")
    np.save(ALIGNED_LANDMARKS_OUTPUT_PATH, landmarks_train_aligned)
    logging.info(f"Guardando forma media (144 pts) en: {MEAN_SHAPE_OUTPUT_PATH}")
    np.save(MEAN_SHAPE_OUTPUT_PATH, mean_shape)

    # 4. Opcional: Recalcular y guardar pose media (requiere cargar los raw 15pts originales tambien)
    #    (Omitido por ahora para simplificar, pero el código está en la versión anterior si lo necesitas)
    #    logging.info("Cálculo de pose media omitido en esta versión.")

    # 5. Generar visualizaciones (ahora con 144 puntos)
    aligned_overlay_path = os.path.join(PLOTS_DIR, "aligned_landmarks_overlay_144pts.png")
    plot_aligned_landmarks(landmarks_train_aligned, mean_shape, aligned_overlay_path)

    mean_shape_plot_path = os.path.join(PLOTS_DIR, "mean_shape_gpa_144pts.png")
    plot_mean_shape(mean_shape, mean_shape_plot_path)

    logging.info("--- Fase 1: GPA (sobre 144 puntos) Finalizada ---")

if __name__ == "__main__":
    main()