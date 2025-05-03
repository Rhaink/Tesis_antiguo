# Tesis/alineamiento/prepare_splits.py

import os
import logging
from src.data_loader import load_index_map # Para obtener etiquetas para estratificar
from src.data_splitter import split_data, save_indices # Importar funciones de división y guardado

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuración ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Tesis
RESULTS_DIR = os.path.join(BASE_DIR, "alineamiento", "results")
NUM_SAMPLES = 800 # Número total de imágenes en el dataset maestro
TEST_SIZE = 0.20 # 20% para prueba
VALIDATION_SIZE = 0.15 # 15% para validación (del total)
RANDOM_STATE = 42 # Para reproducibilidad
STRATIFY = True # Intentar estratificar por categoría
# --------------------

def main():
    logging.info("--- Iniciando Preparación de Splits de Datos ---")

    stratify_labels = None
    if STRATIFY:
        logging.info("Intentando cargar etiquetas para estratificación...")
        index_map = load_index_map() # Usa la ruta por defecto
        if index_map is not None:
            # Asegurarse que los índices van de 0 a N-1
            if set(index_map.index) == set(range(len(index_map))):
                 stratify_labels = index_map['category_id'].values
                 logging.info("Etiquetas de categoría cargadas para estratificación.")
            else:
                 logging.warning("Los índices en index_map no son contiguos de 0 a N-1. No se puede garantizar la estratificación correcta.")
                 stratify_labels = None
        else:
            logging.warning("No se pudieron cargar las etiquetas, la división no será estratificada.")

    # Realizar la división
    splits = split_data(
        num_samples=NUM_SAMPLES,
        test_size=TEST_SIZE,
        validation_size=VALIDATION_SIZE,
        stratify_labels=stratify_labels,
        random_state=RANDOM_STATE
    )

    # Guardar los índices
    save_indices(splits, RESULTS_DIR)

    logging.info("--- Preparación de Splits de Datos Finalizada ---")

if __name__ == "__main__":
    main()