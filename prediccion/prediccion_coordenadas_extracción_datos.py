import json
from pathlib import Path
import pickle
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import mean_squared_error
import math
import os  # Importa el módulo 'os'

# --- Rutas y Configuraciones ---
base_dir = Path("/home/donrobot/projects/Tesis")
models_dir = base_dir / "resultados/entrenamiento/dataset_aligned_maestro_1/models"
images_dir = base_dir / "dataset" / "dataset_aligned_prueba_1"  
json_path = base_dir / "resultados/region_busqueda/dataset_aligned_maestro_1/json/all_search_coordinates.json"
output_dir = base_dir / "resultados/prediccion/dataset_aligned_maestro_1"
output_dir.mkdir(parents=True, exist_ok=True)

# Tamaños de los crops y centroides (height,width)
coord1_crop_size = (38, 62)
coord2_crop_size = (39, 62)
coord1_centroid_local = (0, 31)
coord2_centroid_local = (38, 31)
resized_image_size = (64, 64)

# --- Funciones ---
def load_image(image_path):
    """Carga una imagen y la convierte a escala de grises."""
    try:
        img = Image.open(image_path).convert('L')
        return img
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar la imagen en {image_path}")
        return None
    except Exception as e:
        print(f"Error al cargar la imagen {image_path}: {e}")
        return None


def resize_image(image, size):
    """Redimensiona una imagen al tamaño dado."""
    return image.resize(size)


def load_pca_model(model_path):
    """Carga el modelo PCA desde un archivo pickle."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def load_coordinates(json_path):
    """Carga las coordenadas desde un archivo JSON."""
    with open(json_path, 'r') as f:
        all_coords = json.load(f)
    return all_coords.get('coord1', []), all_coords.get('coord2', [])


def crop_image(image, top_left, size):
    """Recorta una región de la imagen."""
    return image.crop((top_left[1], top_left[0], top_left[1] + size[1], top_left[0] + size[0]))


def image_to_vector(image):
    """Convierte una imagen a un vector NumPy."""
    return np.array(image).flatten()


def vector_to_image(vector, size):
    """Reconstruye una imagen desde un vector."""
    return Image.fromarray(vector.reshape(size).astype(np.uint8))


def calculate_mse(image1, image2):
    """Calcula el Mean Squared Error entre dos imágenes."""
    vector1 = image_to_vector(image1).astype(np.float64)
    vector2 = image_to_vector(image2).astype(np.float64)
    return mean_squared_error(vector1, vector2)


def calculate_euclidean_distance(image1, image2):
    """Calcula la distancia euclidiana entre dos imágenes."""
    vector1 = image_to_vector(image1).astype(np.float64)
    vector2 = image_to_vector(image2).astype(np.float64)
    return math.sqrt(np.sum((vector1 - vector2) ** 2))


def apply_pca(image_vector, model_data):
    """Aplica la transformación PCA a un vector de imagen."""
    float_vector = image_vector.astype(float)
    centered_vector = float_vector - model_data['mean_face'].flatten()
    return model_data['pca'].transform(centered_vector.reshape(1, -1))[0]


def reconstruct_pca(projected_vector, model_data):
    """Reconstruye la imagen desde su proyección PCA."""
    reconstructed_centered = model_data['pca'].inverse_transform(projected_vector.reshape(1, -1))[0]
    reconstructed = reconstructed_centered + model_data['mean_face'].flatten()
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    return reconstructed


def mark_coordinate(image, coordinate, color=(255, 0, 0)):
    """Marca un píxel en la imagen con un color RGB."""
    image.putpixel((coordinate[1], coordinate[0]), color)
    return image


def process_region(image, coords, centroid_local, crop_size, pca_model, metric='mse'):
    """
    Procesa una región (coord1 o coord2) de la imagen, buscando la mejor coincidencia PCA.

    Args:
        image (PIL.Image): La imagen redimensionada.
        coords (list): Lista de coordenadas (y, x) a evaluar.
        centroid_local (tuple): Centroide local de la región.
        crop_size (tuple): Tamaño del crop de la región.
        pca_model (dict): Diccionario del modelo PCA cargado.
        metric (str): 'mse' para Mean Squared Error, 'euclidean' para distancia euclidiana.

    Returns:
        tuple: (mejor_coordenada, error_mínimo) o (None, None) si no se encontraron coordenadas válidas.
    """

    min_error = float('inf')
    best_coord = None

    for y_c, x_c in coords:
        top_left_y = y_c - centroid_local[0]
        top_left_x = x_c - centroid_local[1]
        top_left = (top_left_y, top_left_x)

        if 0 <= top_left_y and top_left_y + crop_size[0] <= resized_image_size[0] and \
           0 <= top_left_x and top_left_x + crop_size[1] <= resized_image_size[1]:
            try:
                cropped_region = crop_image(image, top_left, crop_size)
                cropped_vector = image_to_vector(cropped_region).astype(np.float64)

                projected_vector = apply_pca(cropped_vector, pca_model)
                reconstructed_vector = reconstruct_pca(projected_vector, pca_model)
                reconstructed_image = vector_to_image(reconstructed_vector, crop_size)

                if metric == 'mse':
                    error = calculate_mse(cropped_region, reconstructed_image)
                elif metric == 'euclidean':
                    error = calculate_euclidean_distance(cropped_region, reconstructed_image)
                else:
                    raise ValueError("Métrica no válida. Debe ser 'mse' o 'euclidean'.")

                if error < min_error:
                    min_error = error
                    best_coord = (y_c, x_c)

            except Exception as e:
                print(f"Error processing coordinate ({y_c}, {x_c}): {e}")

    return best_coord, min_error


# --- Carga de Modelos y Coordenadas ---
coord1_coords, coord2_coords = load_coordinates(json_path)
pca_model_coord1 = load_pca_model(models_dir / "coord1_model.pkl")
pca_model_coord2 = load_pca_model(models_dir / "coord2_model.pkl")

# --- Bucle Principal para Procesar Imágenes ---
results = {}  # Diccionario para almacenar los resultados de cada imagen

for image_name in os.listdir(images_dir):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')): #Verificar que el archivo sea una imagen
        image_path = images_dir / image_name
        print(f"Procesando imagen: {image_name}")

        test_image = load_image(image_path)
        if test_image is None:  # Saltar a la siguiente imagen si falla la carga
            continue

        resized_test_image = resize_image(test_image, resized_image_size)
        output_image = resized_test_image.convert('RGB')  # Para dibujar en color

        # Procesar coord1 (MSE)
        best_coord1_mse, min_error_coord1_mse = process_region(
            resized_test_image, coord1_coords, coord1_centroid_local, coord1_crop_size, pca_model_coord1, metric='mse'
        )

        # Procesar coord1 (Euclidiana)
        best_coord1_euclidean, min_error_coord1_euclidean = process_region(
            resized_test_image, coord1_coords, coord1_centroid_local, coord1_crop_size, pca_model_coord1, metric='euclidean'
        )

        # Procesar coord2 (MSE)
        best_coord2_mse, min_error_coord2_mse = process_region(
            resized_test_image, coord2_coords, coord2_centroid_local, coord2_crop_size, pca_model_coord2, metric='mse'
        )

        # Procesar coord2 (Euclidiana)
        best_coord2_euclidean, min_error_coord2_euclidean = process_region(
            resized_test_image, coord2_coords, coord2_centroid_local, coord2_crop_size, pca_model_coord2, metric='euclidean'
        )

        # Almacenar resultados
        results[image_name] = {
            'coord1_mse': {'coordinate': best_coord1_mse, 'error': min_error_coord1_mse},
            'coord1_euclidean': {'coordinate': best_coord1_euclidean, 'error': min_error_coord1_euclidean},
            'coord2_mse': {'coordinate': best_coord2_mse, 'error': min_error_coord2_mse},
            'coord2_euclidean': {'coordinate': best_coord2_euclidean, 'error': min_error_coord2_euclidean},
        }

        # Marcar las mejores coordenadas en la imagen
        if best_coord1_mse:
            output_image = mark_coordinate(output_image, best_coord1_mse, color=(0, 255, 0))  # Verde
        if best_coord1_euclidean:
            output_image = mark_coordinate(output_image, best_coord1_euclidean, color=(255, 0, 0))  # Rojo
        if best_coord2_mse:
            output_image = mark_coordinate(output_image, best_coord2_mse, color=(0, 0, 255))  # Azul
        if best_coord2_euclidean:
            output_image = mark_coordinate(output_image, best_coord2_euclidean, color=(255, 255, 0))  # Amarillo

        # Guardar la imagen con las marcas
        output_image_path = output_dir / "lote" / f"{Path(image_name).stem}_matches.png"
        output_image.save(output_image_path)
        print(f"Imagen guardada en: {output_image_path}")


# Guardar todos los resultados en un archivo JSON
results_path = output_dir / "lote" / "json"/ "results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"Todos los resultados guardados en: {results_path}")