# Tesis/alineamiento/visualize_profiles.py
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuración de Rutas ---
# Ajustar si este script no está en Tesis/alineamiento/
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = CURRENT_SCRIPT_DIR # Asume que está en Tesis/alineamiento/
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
    PROFILE_MODELS_PATH = os.path.join(RESULTS_DIR, 'asm_profile_models.npz')
    os.makedirs(PLOTS_DIR, exist_ok=True)
except NameError: # Para ejecución interactiva (ej. Jupyter)
    # Definir rutas manualmente si __file__ no está definido
    BASE_DIR = '.' # Ajustar según sea necesario
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
    PROFILE_MODELS_PATH = os.path.join(RESULTS_DIR, 'asm_profile_models.npz')
    os.makedirs(PLOTS_DIR, exist_ok=True)
    logging.info("Ejecutando en modo interactivo, rutas ajustadas manualmente.")


NUM_LANDMARKS = 15 # Asegurarse que coincida

def visualize_mean_profiles(models_path, output_dir):
    """Carga los modelos de perfil ASM y visualiza los perfiles medios."""
    logging.info(f"Cargando modelos de perfil desde: {models_path}")
    if not os.path.exists(models_path):
        logging.error(f"Archivo no encontrado: {models_path}")
        return

    try:
        profile_data = np.load(models_path, allow_pickle=True)
        profile_models = profile_data['models']

        if len(profile_models) != NUM_LANDMARKS:
            logging.error(f"Número de modelos encontrados ({len(profile_models)}) no coincide con NUM_LANDMARKS ({NUM_LANDMARKS}).")
            return

        if 'mean' not in profile_models[0]:
             logging.error("El formato de los modelos no contiene la clave 'mean'.")
             return
        
        profile_length = len(profile_models[0]['mean'])
        logging.info(f"Modelos cargados para {len(profile_models)} landmarks. Longitud de perfil: {profile_length}.")
        
        # Determinar el nombre del archivo basado en el método de normalización (heurística)
        # Asumimos que si la media de un perfil típico está cerca de 0, es z-score.
        # Si la suma de absolutos está cerca de 1, es 'sum'.
        sample_mean = np.mean(profile_models[0]['mean'])
        sample_sum_abs = np.sum(np.abs(profile_models[0]['mean']))
        norm_method_guess = "unknown"
        if abs(sample_mean) < 1e-3 : # Z-score tiende a tener media cercana a 0
             norm_method_guess = "zscore"
        elif abs(sample_sum_abs - 1.0) < 1e-3: # Sum normalization tiende a tener suma de abs cercana a 1
             norm_method_guess = "sum"
        
        logging.info(f"Método de normalización detectado (heurística): {norm_method_guess}")
        
        # Crear la figura para los subplots
        # Ajustar layout según NUM_LANDMARKS
        n_cols = 5
        n_rows = (NUM_LANDMARKS + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), sharex=True, sharey=True)
        axes = axes.flatten() # Convertir a array 1D para fácil iteración

        x_axis = np.arange(profile_length) - (profile_length - 1) / 2 # Centrar en 0

        for i in range(NUM_LANDMARKS):
            mean_profile = profile_models[i]['mean']
            axes[i].plot(x_axis, mean_profile, marker='.')
            axes[i].set_title(f'Landmark {i}')
            axes[i].grid(True, linestyle='--', alpha=0.6)
            # Añadir línea en x=0
            axes[i].axvline(0, color='red', linestyle=':', linewidth=1)


        # Ocultar ejes sobrantes si NUM_LANDMARKS no es múltiplo de n_cols
        for j in range(NUM_LANDMARKS, n_rows * n_cols):
            fig.delaxes(axes[j])

        fig.suptitle(f'Perfiles Medios ASM (Normalización: {norm_method_guess.upper()})', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar para el título principal

        # Guardar la figura
        output_filename = f"mean_asm_profiles_{norm_method_guess}.png"
        output_path = os.path.join(output_dir, output_filename)
        try:
            plt.savefig(output_path)
            logging.info(f"Visualización de perfiles guardada en: {output_path}")
        except Exception as e:
            logging.error(f"Error guardando la visualización: {e}")
        plt.close(fig)

    except Exception as e:
        logging.error(f"Error procesando los modelos de perfil: {e}", exc_info=True)


if __name__ == "__main__":
    visualize_mean_profiles(PROFILE_MODELS_PATH, PLOTS_DIR)