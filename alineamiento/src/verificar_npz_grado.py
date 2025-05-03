# Tesis/alineamiento/src/verificar_npz.py
# Modificado para leer archivos NPZ de ESL (líneas o orientación)

import numpy as np
import os
import sys # Para leer argumentos de línea de comando

# --- Define la ruta al archivo ---
try:
    # Intenta obtener la ruta relativa al script actual
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ALINEAMIENTO_DIR = os.path.dirname(CURRENT_SCRIPT_DIR) # Tesis/alineamiento
    RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
except NameError:
     # Si estás en una consola interactiva o __file__ no está definido
     # Default a la ruta que usaste, pero mejor si lo pasas como argumento
     RESULTS_DIR = '/home/donrobot/projects/Tesis/alineamiento/results'
     print(f"Usando ruta manual para results: {RESULTS_DIR}")

# --- Determinar qué archivo cargar ---
# Permitir pasar el nombre del archivo como argumento
# Ejemplo: python verificar_npz.py esl_orientation_patches_train.npz
# O:       python verificar_npz.py esl_line_patches_train.npz

default_file = 'esl_orientation_patches_train.npz' # Cambiado el default
filename_to_load = default_file

if len(sys.argv) > 1:
    filename_to_load = sys.argv[1]
    print(f"Se cargará el archivo especificado: {filename_to_load}")
else:
    print(f"No se especificó archivo. Usando default: {filename_to_load}")

NPZ_FILE_PATH = os.path.join(RESULTS_DIR, filename_to_load)
# ---------------------------------

print(f"\nCargando archivo: {NPZ_FILE_PATH}")

try:
    data = np.load(NPZ_FILE_PATH)

    # 1. Ver las claves (nombres de los arrays guardados)
    print("\nClaves encontradas en el archivo:")
    keys = list(data.keys())
    print(keys)

    # 2. Inspeccionar los arrays encontrados
    print("\nInspeccionando arrays:")

    for key in keys:
        array = data[key]
        print(f"  - Clave '{key}':")
        print(f"    - Shape: {array.shape}")
        print(f"    - Dtype: {array.dtype}")

        # Mostrar primeros valores si no es un array de parches muy grande
        if "patches" not in key: # No imprimir parches
             print(f"    - Primeros 5 valores: {array[:5]}")
        else:
             # Para parches, solo mostrar el rango de valores del primer parche
             if array.ndim >= 3 and array.shape[0] > 0: # Verificar que haya al menos un parche
                 first_patch = array[0]
                 print(f"    - Rango de valores del primer parche: [{np.min(first_patch):.4f}, {np.max(first_patch):.4f}]")

        # Si es un array de etiquetas, mostrar distribución
        if "labels" in key:
            if array.ndim == 1: # Asegurar que sea 1D para bincount
                distribution = np.bincount(array)
                print(f"    - Distribución de etiquetas (índice=etiqueta): {distribution}")
            else:
                print("    - (No se muestra distribución para array de etiquetas multidimensional)")


    # ¡Importante! Cerrar el archivo después de usarlo
    data.close()

except FileNotFoundError:
    print(f"Error: Archivo no encontrado en {NPZ_FILE_PATH}")
except Exception as e:
    print(f"Ocurrió un error al cargar o inspeccionar el archivo: {e}")