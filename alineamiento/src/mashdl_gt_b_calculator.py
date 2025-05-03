# Tesis/alineamiento/src/mashdl_gt_b_calculator.py

import os
import numpy as np
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración de Rutas ---
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Asume que este script está en alineamiento/src/ o alineamiento/
    if os.path.basename(CURRENT_SCRIPT_DIR) == 'src':
        ALINEAMIENTO_DIR = os.path.dirname(CURRENT_SCRIPT_DIR)
    else: # Asume que está en alineamiento/
        ALINEAMIENTO_DIR = CURRENT_SCRIPT_DIR
    RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
except NameError:
    ALINEAMIENTO_DIR = '.'
    RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
    logger.info("Ejecutando en modo interactivo, rutas ajustadas.")

# Rutas a archivos de entrada (Resultados de Fase 0)
ALIGNED_LANDMARKS_PATH = os.path.join(RESULTS_DIR, 'landmarks_train_aligned.npy')
PCA_MEAN_VECTOR_PATH = os.path.join(RESULTS_DIR, 'pca_mean_vector.npy')
PCA_COMPONENTS_PATH = os.path.join(RESULTS_DIR, 'pca_components.npy')
TRAIN_INDICES_PATH = os.path.join(RESULTS_DIR, 'train_indices.txt')

# Ruta de salida
OUTPUT_B_GT_PATH = os.path.join(RESULTS_DIR, 'mashdl_ground_truth_b_train.npz')

# --- Constantes (Asegurar que coincidan con el SSM) ---
# NUM_LANDMARKS = 15 # No estrictamente necesario aquí si las shapes son correctas
# NUM_DIMS = 2
# -----------------------------------------------------

def main():
    logger.info("--- Iniciando Cálculo de Parámetros b_gt para MaShDL (Entrenamiento) ---")

    # 1. Cargar datos necesarios
    logger.info("Cargando datos de entrada...")
    try:
        if not os.path.exists(ALIGNED_LANDMARKS_PATH): raise FileNotFoundError(ALIGNED_LANDMARKS_PATH)
        aligned_landmarks_train = np.load(ALIGNED_LANDMARKS_PATH)
        logger.info(f"Landmarks alineados cargados. Shape: {aligned_landmarks_train.shape}") # (N_train, k, d)

        if not os.path.exists(PCA_MEAN_VECTOR_PATH): raise FileNotFoundError(PCA_MEAN_VECTOR_PATH)
        pca_mean_vector = np.load(PCA_MEAN_VECTOR_PATH)
        logger.info(f"Vector medio PCA cargado. Shape: {pca_mean_vector.shape}") # (k*d,)

        if not os.path.exists(PCA_COMPONENTS_PATH): raise FileNotFoundError(PCA_COMPONENTS_PATH)
        pca_components = np.load(PCA_COMPONENTS_PATH)
        logger.info(f"Componentes PCA (P) cargados. Shape: {pca_components.shape}") # (n_modes, k*d)

        if not os.path.exists(TRAIN_INDICES_PATH): raise FileNotFoundError(TRAIN_INDICES_PATH)
        train_indices = np.loadtxt(TRAIN_INDICES_PATH, dtype=int)
        if train_indices.ndim == 0: train_indices = np.array([train_indices.item()])
        logger.info(f"Índices de entrenamiento cargados. Cantidad: {len(train_indices)}")

    except FileNotFoundError as e:
        logger.error(f"Error: Archivo no encontrado: {e}")
        logger.error("Asegúrese de haber ejecutado run_gpa.py y run_pca_analysis.py.")
        return
    except Exception as e:
        logger.error(f"Error cargando archivos de entrada: {e}", exc_info=True)
        return

    # Validaciones de consistencia
    num_aligned_shapes = aligned_landmarks_train.shape[0]
    expected_vector_len = pca_mean_vector.shape[0]
    num_modes = pca_components.shape[0]

    if num_aligned_shapes != len(train_indices):
        logger.warning(f"El número de formas alineadas ({num_aligned_shapes}) no coincide con el número de índices de entrenamiento ({len(train_indices)}). Se usarán solo las formas alineadas.")
        # Podríamos truncar train_indices o fallar, pero continuar es más flexible
        # train_indices = train_indices[:num_aligned_shapes] # Opción: truncar

    if aligned_landmarks_train.shape[1] * aligned_landmarks_train.shape[2] != expected_vector_len:
        logger.error(f"Inconsistencia: k*d de landmarks ({aligned_landmarks_train.shape[1]}*{aligned_landmarks_train.shape[2]}) no coincide con longitud de vector medio ({expected_vector_len}).")
        return
    if pca_components.shape[1] != expected_vector_len:
        logger.error(f"Inconsistencia: Dimensión de componentes PCA ({pca_components.shape[1]}) no coincide con longitud de vector medio ({expected_vector_len}).")
        return

    # 2. Calcular b_gt para cada forma
    logger.info(f"Calculando {num_modes} parámetros b_gt para {num_aligned_shapes} formas...")
    b_gt_list = []
    for i in range(num_aligned_shapes):
        try:
            x_aligned = aligned_landmarks_train[i]
            shape_vector = x_aligned.flatten() # Vectorizar (k*d,)
            diff = shape_vector - pca_mean_vector
            # Proyección: b = P @ diff^T (si P es (n, k*d) y diff es (k*d, 1))
            # O b = P @ diff si diff es (k*d,) y P es (n, k*d) -> resultado (n,)
            b = pca_components @ diff
            b_gt_list.append(b)
        except Exception as e:
            logger.error(f"Error calculando b_gt para la forma índice {i}: {e}")
            # Añadir un vector de NaNs o ceros para mantener la correspondencia? O simplemente omitir?
            # Omitir por ahora es más seguro para evitar datos malos.
            logger.warning(f"Se omitirá la forma índice {i} del resultado final.")
            # Necesitaríamos ajustar los train_indices si omitimos muestras.

    # Es más robusto calcular todo junto si la memoria lo permite
    try:
        logger.info("Calculando todos los b_gt usando operaciones vectorizadas...")
        # Reshape aligned_landmarks a (N_train, k*d)
        data_matrix = aligned_landmarks_train.reshape(num_aligned_shapes, -1)
        # Restar la media (broadcasting)
        diff_matrix = data_matrix - pca_mean_vector # Shape (N_train, k*d)
        # Proyectar: b = P @ diff^T -> transponer diff_matrix y luego P @ ...
        # O más fácil: (diff @ P^T)^T ? No.
        # Correcto: b_gt_array = (pca_components @ diff_matrix.T).T
        # pca_components: (n_modes, k*d)
        # diff_matrix.T: (k*d, N_train)
        # Resultado: (n_modes, N_train) -> Transponer a (N_train, n_modes)
        b_gt_array = (pca_components @ diff_matrix.T).T

    except Exception as e:
         logger.error(f"Error en cálculo vectorizado: {e}. Volviendo a método iterativo si es posible.")
         # Usar b_gt_list si el cálculo iterativo se completó (requiere ajuste si hubo omisiones)
         if not b_gt_list: return # Salir si ambos fallan
         logger.warning("Usando resultados del cálculo iterativo.")
         b_gt_array = np.array(b_gt_list)


    logger.info(f"Cálculo completado. Shape del array b_gt: {b_gt_array.shape}") # (N_train, n_modes)
    if b_gt_array.shape[0] != num_aligned_shapes:
         logger.warning("El número de vectores b_gt calculados no coincide con el número inicial de formas alineadas (posiblemente por errores en bucle).")
         # Asegurar que los índices coincidan si hubo omisiones
         # Esto requiere una lógica más compleja para rastrear qué índices se procesaron.
         # Por simplicidad ahora, asumimos que todas las formas se procesaron.
         if b_gt_array.shape[0] != len(train_indices):
              logger.error("Discrepancia final entre número de b_gt y train_indices. No se guardará.")
              return


    # 3. Guardar resultados
    logger.info(f"Guardando b_gt y los índices correspondientes en: {OUTPUT_B_GT_PATH}")
    try:
        output_data = {
            'b_gt': b_gt_array,
            'index': train_indices # Asume que train_indices tiene la longitud correcta
        }
        np.savez(OUTPUT_B_GT_PATH, **output_data)
        logger.info("Archivo b_gt guardado exitosamente.")
    except Exception as e:
        logger.error(f"Error al guardar el archivo NPZ: {e}")

    logger.info("--- Cálculo de Parámetros b_gt Finalizado ---")

if __name__ == "__main__":
    main()