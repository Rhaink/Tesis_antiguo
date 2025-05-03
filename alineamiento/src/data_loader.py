# Tesis/alineamiento/src/data_loader.py

import os
import pandas as pd
import numpy as np
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes de Rutas Relativas (desde Tesis/) ---
# Se asume que este script está en Tesis/alineamiento/src/
# Por lo tanto, BASE_DIR apunta a Tesis/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Va 3 niveles arriba: src -> alineamiento -> Tesis

DEFAULT_INDICES_FILE = os.path.join(BASE_DIR, "indices", "indices_maestro_1.csv")
DEFAULT_COORDS_FILE = os.path.join(BASE_DIR, "coordenadas", "coordenadas_maestro_1.csv")
DEFAULT_DATASET_DIR = os.path.join(BASE_DIR, "COVID-19_Radiography_Dataset")
# ----------------------------------------------------

def load_index_map(index_file_path=DEFAULT_INDICES_FILE):
    """
    Carga el archivo de índices maestro y crea un mapeo.

    Args:
        index_file_path (str): Ruta al archivo CSV de índices.

    Returns:
        pd.DataFrame or None: DataFrame con columnas ['new_index', 'category_id', 'original_id']
                              o None si hay error.
    """
    logging.info(f"Cargando mapa de índices desde: {index_file_path}")
    try:
        # Leer especificando tipos iniciales
        index_map = pd.read_csv(index_file_path, header=None, 
                                names=['new_index', 'category_id', 'original_id'],
                                dtype={0: int, 1: int, 2: object}) # Leer ID original como objeto primero

        # Convertir ID original a numérico, manejando errores
        index_map['original_id'] = pd.to_numeric(index_map['original_id'], errors='coerce')
        
        # Verificar y manejar NaNs (IDs no convertidos)
        initial_count = len(index_map)
        index_map.dropna(subset=['original_id'], inplace=True)
        final_count = len(index_map)
        if final_count < initial_count:
            logging.warning(f"Se eliminaron {initial_count - final_count} filas del archivo de índices debido a IDs no numéricos.")
        
        index_map['original_id'] = index_map['original_id'].astype(int) # Convertir a int después de limpiar

        # Validaciones básicas
        if not (index_map['category_id'].isin([1, 2, 3]).all()):
             logging.warning("Se encontraron category_id fuera del rango esperado [1, 2, 3].")
             # Podríamos filtrar o manejar según necesidad

        logging.info(f"Mapa de índices cargado exitosamente con {len(index_map)} entradas válidas.")
        return index_map.set_index('new_index') # Usar new_index como índice del DataFrame

    except FileNotFoundError:
        logging.error(f"Error CRÍTICO: No se encontró el archivo de índices '{index_file_path}'.")
        return None
    except Exception as e:
        logging.error(f"Error al leer o procesar el archivo de índices '{index_file_path}': {e}")
        return None

def load_landmarks(coords_file_path=DEFAULT_COORDS_FILE, num_landmarks=15):
    """
    Carga las coordenadas de los landmarks desde el archivo maestro.
    Se asume que son las coordenadas 64x64.

    Args:
        coords_file_path (str): Ruta al archivo CSV de coordenadas.
        num_landmarks (int): Número esperado de landmarks (default: 15).

    Returns:
        tuple (np.ndarray, list) or (None, None): 
            - Array NumPy de landmarks con shape (N, k, d) donde N=num_imagenes, k=num_landmarks, d=2 (x, y).
            - Lista de etiquetas/nombres originales de archivo.
            O (None, None) si hay error.
    """
    logging.info(f"Cargando landmarks desde: {coords_file_path} (asumiendo {num_landmarks} landmarks, 64x64)")
    num_coord_cols = num_landmarks * 2
    expected_cols = 1 + num_coord_cols + 1 # index + coords + label

    try:
        # Leer el archivo, sin cabecera
        coords_df = pd.read_csv(coords_file_path, header=None)

        if coords_df.shape[1] != expected_cols:
            logging.error(f"Número inesperado de columnas en {coords_file_path}. "
                          f"Esperado: {expected_cols}, Encontrado: {coords_df.shape[1]}")
            return None, None

        # Extraer coordenadas (columnas 1 a num_coord_cols)
        landmark_coords = coords_df.iloc[:, 1:1+num_coord_cols].values

        # Validar que sean numéricas (aunque read_csv suele inferir bien)
        if not np.issubdtype(landmark_coords.dtype, np.number):
             logging.warning("Las coordenadas no parecen ser numéricas. Intentando convertir...")
             landmark_coords = landmark_coords.astype(float) # O manejar errores

        # Reshape a (N, k, d)
        num_images = landmark_coords.shape[0]
        landmarks_array = landmark_coords.reshape(num_images, num_landmarks, 2)

        # Extraer etiquetas (última columna)
        labels = coords_df.iloc[:, -1].tolist()

        logging.info(f"Landmarks cargados para {num_images} imágenes. Shape: {landmarks_array.shape}")
        
        # Validación de rango (opcional pero útil)
        if np.any(landmarks_array < 0) or np.any(landmarks_array >= 64):
            logging.warning("Algunas coordenadas de landmarks están fuera del rango esperado [0, 64). Verificar resolución.")
            
        return landmarks_array, labels

    except FileNotFoundError:
        logging.error(f"Error CRÍTICO: No se encontró el archivo de coordenadas '{coords_file_path}'.")
        return None, None
    except Exception as e:
        logging.error(f"Error al leer o procesar el archivo de coordenadas '{coords_file_path}': {e}")
        return None, None

def get_image_paths(index_map, dataset_base_dir=DEFAULT_DATASET_DIR):
    """
    Construye y valida las rutas a los archivos de imagen.

    Args:
        index_map (pd.DataFrame): DataFrame del mapa de índices (indexado por new_index).
        dataset_base_dir (str): Ruta base al directorio del dataset COVID-19.

    Returns:
        dict: Diccionario mapeando new_index a la ruta completa de la imagen existente.
              Las imágenes no encontradas no se incluyen.
    """
    logging.info(f"Construyendo rutas de imágenes desde: {dataset_base_dir}")
    image_paths = {}
    missing_files = 0

    # Asegurarse que index_map es el DataFrame esperado
    if index_map is None or not isinstance(index_map, pd.DataFrame):
        logging.error("Mapa de índices inválido proporcionado a get_image_paths.")
        return {}
        
    # Iterar sobre el índice del DataFrame (que es new_index)
    for new_idx in index_map.index:
        try:
            # Acceder a las filas usando .loc con el índice
            row = index_map.loc[new_idx]
            categoria = int(row['category_id'])
            img_id = int(row['original_id'])

            path = None
            sub_dir = ""
            filename = ""

            if categoria == 1:
                sub_dir = "COVID/images"
                filename = f"COVID-{img_id}.png"
            elif categoria == 2:
                sub_dir = "Normal/images"
                filename = f"Normal-{img_id}.png"
            elif categoria == 3:
                sub_dir = "Viral Pneumonia/images"
                filename = f"Viral Pneumonia-{img_id}.png"
            else:
                logging.warning(f"Categoría desconocida '{categoria}' para new_index {new_idx}. Omitiendo.")
                continue

            path = os.path.join(dataset_base_dir, sub_dir, filename)

            # Verificar existencia
            if os.path.exists(path):
                image_paths[new_idx] = path
            else:
                logging.warning(f"Archivo de imagen no encontrado: {path}. Omitiendo.")
                missing_files += 1
                
        except KeyError:
             logging.error(f"Error procesando new_index {new_idx}. ¿Faltan columnas 'category_id' o 'original_id'?")
             continue
        except Exception as e:
             logging.error(f"Error inesperado procesando new_index {new_idx}: {e}")
             continue


    logging.info(f"Se encontraron {len(image_paths)} rutas de imágenes válidas.")
    if missing_files > 0:
        logging.warning(f"No se encontraron {missing_files} archivos de imagen esperados.")
    return image_paths

def load_all_data(base_dir=None):
    """
    Carga todos los datos necesarios: mapa de índices, landmarks y rutas de imágenes.

    Args:
        base_dir (str, optional): Ruta al directorio 'Tesis'. Si es None, se infiere.

    Returns:
        tuple: (index_map, landmarks_array, image_paths)
               o (None, None, None) si ocurre un error crítico.
    """
    if base_dir is None:
        # Inferir base_dir asumiendo que estamos en Tesis/alineamiento/src
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        logging.info(f"Directorio base inferido: {base_dir}")

    indices_file = os.path.join(base_dir, "indices", "indices_maestro_1.csv")
    coords_file = os.path.join(base_dir, "coordenadas", "coordenadas_maestro_1.csv")
    dataset_dir = os.path.join(base_dir, "COVID-19_Radiography_Dataset")

    index_map = load_index_map(indices_file)
    if index_map is None:
        return None, None, None

    landmarks_array, _ = load_landmarks(coords_file) # Ignoramos las etiquetas por ahora
    if landmarks_array is None:
        return None, None, None
        
    # Asegurarse que el número de imágenes coincida
    if len(index_map) != landmarks_array.shape[0]:
        logging.warning(f"Discrepancia en número de muestras: {len(index_map)} en índices, {landmarks_array.shape[0]} en coordenadas.")
        # Podríamos necesitar alinear los datos aquí si hay discrepancias reales
        
    image_paths = get_image_paths(index_map, dataset_dir)
    # Verificar si todas las entradas del index_map tienen una ruta válida
    if len(image_paths) < len(index_map):
         logging.warning(f"No todas las entradas del índice tienen una imagen válida asociada ({len(image_paths)} vs {len(index_map)}).")
         # Podríamos necesitar filtrar index_map y landmarks_array para que coincidan con image_paths

    # Por ahora, devolvemos todo, la consistencia se revisará en los pasos siguientes
    return index_map, landmarks_array, image_paths

# --- Ejemplo de uso (se puede comentar o quitar) ---
# if __name__ == "__main__":
#     print(f"Ejecutando data_loader desde: {os.getcwd()}")
#     print(f"Directorio base Tesis: {BASE_DIR}")
#     
#     index_map_df, landmarks, paths = load_all_data(BASE_DIR)
# 
#     if index_map_df is not None:
#         print("\nMapa de Índices (primeras 5 filas):")
#         print(index_map_df.head())
# 
#     if landmarks is not None:
#         print(f"\nLandmarks shape: {landmarks.shape}")
#         print("Primer landmark de la primera imagen:", landmarks[0, 0, :])
# 
#     if paths:
#         print(f"\nNúmero de rutas de imagen válidas: {len(paths)}")
#         print("Ejemplo de ruta (índice 0):", paths.get(0, "No encontrado"))