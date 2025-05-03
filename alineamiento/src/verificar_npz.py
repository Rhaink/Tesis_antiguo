import numpy as np
import os

# --- Define la ruta al archivo ---
# (Asegúrate de que esta ruta sea correcta desde donde ejecutas Python)
try:
    # Intenta obtener la ruta relativa al script actual
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ALINEAMIENTO_DIR = os.path.dirname(CURRENT_SCRIPT_DIR) # Tesis/alineamiento
    RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
except NameError:
     # Si estás en una consola interactiva o __file__ no está definido
     RESULTS_DIR = '/home/donrobot/projects/Tesis/alineamiento/results' # Pon tu ruta absoluta aquí
     print(f"Usando ruta manual: {RESULTS_DIR}")

NPZ_FILE_PATH = os.path.join(RESULTS_DIR, 'mashdl_ground_truth_b_train.npz')
# ---------------------------------


print(f"Cargando archivo: {NPZ_FILE_PATH}")

try:
    # Cargar el archivo .npz
    data = np.load(NPZ_FILE_PATH)

    # 1. Ver las claves (nombres de los arrays guardados)
    print("\nClaves encontradas en el archivo:")
    print(list(data.keys()))

    # 2. Verificar el número de muestras (debería ser 520)
    #    Podemos comprobar la longitud de cualquiera de los arrays
    if 'index' in data:
        num_samples = len(data['index'])
        print(f"\nNúmero de muestras encontradas (basado en 'index'): {num_samples}")
    else:
        # Si 'index' no estuviera, probar con otra clave
        first_key = list(data.keys())[0]
        num_samples = len(data[first_key])
        print(f"\nNúmero de muestras encontradas (basado en '{first_key}'): {num_samples}")

    # 3. Inspeccionar algunos arrays específicos
    print("\nInspeccionando algunos arrays:")

    if 'l1' in data:
        l1_array = data['l1']
        print(f"  - Clave 'l1':")
        print(f"    - Shape: {l1_array.shape}") # Debería ser (520,)
        print(f"    - Primeros 5 valores: {l1_array[:5]}") # Mostrar los primeros 5 valores de x_min

    if 'Tx' in data:
        Tx_array = data['Tx']
        print(f"  - Clave 'Tx':")
        print(f"    - Shape: {Tx_array.shape}") # Debería ser (520,)
        print(f"    - Primeros 5 valores: {Tx_array[:5]}") # Mostrar los primeros 5 centros en X

    if 'S_height' in data:
        S_height_array = data['S_height']
        print(f"  - Clave 'S_height':")
        print(f"    - Shape: {S_height_array.shape}") # Debería ser (520,)
        print(f"    - Primeros 5 valores: {S_height_array[:5]}") # Mostrar las primeras 5 alturas

    if 'theta' in data:
        theta_array = data['theta']
        print(f"  - Clave 'theta':")
        print(f"    - Shape: {theta_array.shape}") # Debería ser (520,)
        print(f"    - Todos los valores deberían ser 0.0: {np.all(theta_array == 0.0)}") # Verificar que todos son 0

    if 'index' in data:
         index_array = data['index']
         print(f"  - Clave 'index':")
         print(f"    - Shape: {index_array.shape}") # Debería ser (520,)
         print(f"    - Primeros 5 índices originales guardados: {index_array[:5]}")

    # ¡Importante! Cerrar el archivo después de usarlo
    data.close()

except FileNotFoundError:
    print(f"Error: Archivo no encontrado en {NPZ_FILE_PATH}")
except Exception as e:
    print(f"Ocurrió un error al cargar o inspeccionar el archivo: {e}")