# Tesis/alineamiento/run_pca_analysis.py
# MODIFICADO para construir SSM sobre 144 puntos

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import joblib # Para guardar/cargar el objeto PCA completo
# Asumiendo ejecución desde Tesis/alineamiento/
try:
    from src.ssm_builder import vectorize_shapes, build_pca_model, devectorize_shape
except ImportError as e:
     print(f"Error importando módulos desde src: {e}")
     print("Asegúrese de ejecutar este script desde el directorio 'Tesis/alineamiento/'")
     exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuración ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directorio alineamiento/
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# --- Archivos de Entrada y Salida ---
ALIGNED_LANDMARKS_FILE = os.path.join(RESULTS_DIR, 'landmarks_train_aligned_144pts.npy') # <-- ENTRADA (144 pts alineados)
# MEAN_SHAPE_FILE = os.path.join(RESULTS_DIR, 'mean_shape_144pts.npy') # Podríamos cargarlo para comparar

# Archivos de salida (SOBRESCRIBIRÁN los anteriores)
PCA_MEAN_VECTOR_PATH = os.path.join(RESULTS_DIR, 'pca_mean_vector.npy')
PCA_COMPONENTS_PATH = os.path.join(RESULTS_DIR, 'pca_components.npy')
PCA_STD_DEVS_PATH = os.path.join(RESULTS_DIR, 'pca_std_devs.npy')
PCA_MODEL_PATH = os.path.join(RESULTS_DIR, 'pca_model.joblib') # Objeto sklearn

# Parámetros PCA
VAR_THRESHOLD = 0.95 # Umbral informativo para gráfico
N_COMPONENTS_TO_KEEP = 4 # <-- FORZADO a 4 modos en el modelo final
NUM_MODES_TO_VISUALIZE = 3 # Visualizar los 3 primeros de los 4
VIZ_FACTORS = [-2, 0, +2]

# --- Constantes de forma (ACTUALIZADAS) ---
NUM_LANDMARKS = 144 # <-- ACTUALIZADO
NUM_DIMS = 2
# --------------------

# --- Funciones de Plotting (sin cambios lógicos, adaptan títulos/labels) ---
def plot_explained_variance(explained_variance_ratio, n_components_final, threshold=VAR_THRESHOLD):
    """Grafica la varianza explicada individual y acumulada."""
    # (Sin cambios lógicos, solo ajusta el log y el label si es necesario)
    if explained_variance_ratio is None: return
    logging.info("Generando gráfico de varianza explicada (SSM 144 pts)...")
    plt.figure(figsize=(10, 6))
    num_components = len(explained_variance_ratio)
    components_range = np.arange(1, num_components + 1)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    plt.bar(components_range, explained_variance_ratio, alpha=0.6, color='blue', label='Varianza Individual')
    plt.plot(components_range, cumulative_variance, marker='o', linestyle='-', color='red', label='Varianza Acumulada')
    # Marcar umbral informativo
    n_components_for_threshold = np.argmax(cumulative_variance >= threshold) + 1
    variance_at_threshold = cumulative_variance[n_components_for_threshold - 1]
    plt.axhline(y=threshold, color='grey', linestyle='--', label=f'{threshold*100:.0f}% Varianza Umbral ({n_components_for_threshold} comps)')
    # Marcar los componentes retenidos
    variance_at_kept = cumulative_variance[n_components_final - 1]
    plt.axvline(x=n_components_final, color='green', linestyle=':',
                label=f'{n_components_final} Componentes Retenidos ({variance_at_kept*100:.1f}% Var)')
    logging.info(f"Modelo final usa {n_components_final} componentes, explicando {variance_at_kept*100:.1f}% de la varianza.")
    logging.info(f"(Para alcanzar {threshold*100:.0f}% se necesitarían ~{n_components_for_threshold} componentes).")
    plt.xlabel('Número de Componentes Principales'); plt.ylabel('Fracción de Varianza Explicada')
    plt.title('Análisis de Varianza Explicada por PCA (SSM 144 pts)')
    plt.xticks(components_range)
    if num_components > 20: plt.xticks(np.arange(0, num_components+1, max(5, num_components // 10))) # Ajustar xticks
    plt.ylim(0, 1.05); plt.legend(loc='center right'); plt.grid(True, linestyle='--', alpha=0.5)
    output_path = os.path.join(PLOTS_DIR, "pca_explained_variance_144pts.png")
    try: plt.savefig(output_path); logging.info(f"Gráfico de varianza explicada guardado en: {output_path}")
    except Exception as e: logging.error(f"No se pudo guardar el gráfico en {output_path}: {e}")
    plt.close()

def plot_pca_modes(mean_vector, P, std_devs, k, d, output_dir, n_modes=NUM_MODES_TO_VISUALIZE, factors=VIZ_FACTORS):
    """Visualiza los primeros n_modes de variación."""
    # (Sin cambios lógicos, solo ajusta títulos/logs)
    if mean_vector is None or P is None or std_devs is None: return
    n_modes = min(n_modes, P.shape[0])
    logging.info(f"Generando visualización para los primeros {n_modes} modos de variación (SSM 144 pts)...")
    mean_shape = devectorize_shape(mean_vector, k, d)
    if mean_shape is None: logging.error("No se pudo devectorizar la forma media (144 pts)."); return

    for i in range(n_modes):
        fig, axes = plt.subplots(1, len(factors), figsize=(5 * len(factors), 5), sharex=True, sharey=True)
        fig.suptitle(f'Modo de Variación Principal {i+1} (Eigenvalue={std_devs[i]**2:.4f}) - SSM 144 pts')
        eigenvector = P[i, :]; std_dev = std_devs[i]
        for j, factor in enumerate(factors):
            ax = axes[j]
            shape_vec = mean_vector + factor * std_dev * eigenvector
            shape = devectorize_shape(shape_vec, k, d)
            if shape is not None:
                 # Dibujar contornos separados para 144 puntos
                 if shape.shape == (144, 2):
                      ax.plot(shape[0:72, 0], shape[0:72, 1], marker='.', linestyle='-', ms=2, color='blue')
                      ax.plot(shape[72:144, 0], shape[72:144, 1], marker='.', linestyle='-', ms=2, color='cyan')
                      # Superponer media
                      ax.plot(mean_shape[0:72, 0], mean_shape[0:72, 1], linestyle='-', ms=1, alpha=0.3, color='grey')
                      ax.plot(mean_shape[72:144, 0], mean_shape[72:144, 1], linestyle='-', ms=1, alpha=0.3, color='grey')
                 else: # Fallback
                      ax.plot(shape[:, 0], shape[:, 1], marker='.', linestyle='-', ms=3, color='blue')
                      ax.plot(mean_shape[:, 0], mean_shape[:, 1], marker='.', linestyle='-', ms=1, alpha=0.5, color='grey')

            ax.set_title(f'Factor = {factor} * std_dev'); ax.invert_yaxis(); ax.set_aspect('equal', adjustable='box'); ax.grid(True, linestyle='--', alpha=0.5)
        output_path = os.path.join(output_dir, f"pca_mode_{i+1}_visualization_144pts.png")
        try: plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(output_path); logging.info(f"Gráfico modo {i+1} guardado: {output_path}")
        except Exception as e: logging.error(f"No se pudo guardar gráfico modo {i+1}: {e}")
        plt.close(fig)

def main():
    """Ejecuta el análisis PCA sobre los datos alineados de 144 puntos."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logging.info(f"--- Iniciando Fase 2: Análisis PCA (sobre {NUM_LANDMARKS} puntos, FORZANDO {N_COMPONENTS_TO_KEEP} COMPONENTES) ---")

    # 1. Cargar datos alineados (144 puntos)
    if not os.path.exists(ALIGNED_LANDMARKS_FILE):
        logging.error(f"Archivo no encontrado: {ALIGNED_LANDMARKS_FILE}. Ejecute run_gpa.py (modificado) primero.")
        return
    logging.info(f"Cargando landmarks alineados (144 pts) desde: {ALIGNED_LANDMARKS_FILE}")
    try:
        landmarks_train_aligned = np.load(ALIGNED_LANDMARKS_FILE)
        logging.info(f"Landmarks alineados cargados. Shape: {landmarks_train_aligned.shape}")
        if landmarks_train_aligned.shape[1] != NUM_LANDMARKS or landmarks_train_aligned.shape[2] != NUM_DIMS:
             raise ValueError(f"Shape inesperado: {landmarks_train_aligned.shape}. Se esperaba (N, {NUM_LANDMARKS}, {NUM_DIMS}).")
    except Exception as e: logging.error(f"Error cargando datos alineados: {e}"); return

    # 2. Vectorizar formas
    data_matrix = vectorize_shapes(landmarks_train_aligned) # Ahora será (N, 288)
    if data_matrix is None: return
    logging.info(f"Matriz de datos vectorizada creada. Shape: {data_matrix.shape}")

    # 3. Construir modelo PCA inicial (todos los componentes) para análisis de varianza
    logging.info("Calculando PCA completo para análisis de varianza (sobre 144 pts)...")
    pca_full, mean_vec_full, _, exp_var_full, _ = build_pca_model(data_matrix, n_components=None)
    if pca_full is None: logging.error("Falló el cálculo inicial de PCA completo."); return

    # 4. Analizar varianza y graficar (útil para ver cuánta varianza capturan los 4 modos)
    plot_explained_variance(pca_full.explained_variance_ratio_, n_components_final=N_COMPONENTS_TO_KEEP, threshold=VAR_THRESHOLD)

    # 5. Construir modelo PCA final con EXACTAMENTE n_components_to_keep=4
    logging.info(f"Construyendo modelo PCA final con {N_COMPONENTS_TO_KEEP} componentes (sobre 144 pts)...")
    pca_final, mean_vec_final, P_final, exp_var_final, std_devs_final = build_pca_model(data_matrix, n_components=N_COMPONENTS_TO_KEEP)
    if pca_final is None: logging.error(f"Falló construcción PCA final con {N_COMPONENTS_TO_KEEP} componentes."); return
    if P_final.shape[0] != N_COMPONENTS_TO_KEEP: logging.error(f"Error: Modelo final tiene {P_final.shape[0]} comps != {N_COMPONENTS_TO_KEEP}."); return

    # 6. Guardar componentes del modelo final (SOBRESCRIBIRÁN los anteriores)
    logging.info(f"Guardando vector medio PCA (144 pts) en: {PCA_MEAN_VECTOR_PATH}")
    np.save(PCA_MEAN_VECTOR_PATH, mean_vec_final) # Shape (288,)
    logging.info(f"Guardando {P_final.shape[0]} componentes PCA (P) en: {PCA_COMPONENTS_PATH}")
    np.save(PCA_COMPONENTS_PATH, P_final) # Shape (4, 288)
    logging.info(f"Guardando {std_devs_final.shape[0]} std devs PCA en: {PCA_STD_DEVS_PATH}")
    np.save(PCA_STD_DEVS_PATH, std_devs_final) # Shape (4,)
    logging.info(f"Guardando objeto PCA sklearn en: {PCA_MODEL_PATH}")
    joblib.dump(pca_final, PCA_MODEL_PATH)

    # 7. Visualizar modos de variación del modelo final
    plot_pca_modes(mean_vec_final, P_final, std_devs_final,
                   k=NUM_LANDMARKS, d=NUM_DIMS, output_dir=PLOTS_DIR,
                   n_modes=NUM_MODES_TO_VISUALIZE, factors=VIZ_FACTORS)

    logging.info(f"--- Fase 2: Análisis PCA (Modelo {N_COMPONENTS_TO_KEEP} Componentes / {NUM_LANDMARKS} Puntos) Finalizada ---")

if __name__ == "__main__":
    main()