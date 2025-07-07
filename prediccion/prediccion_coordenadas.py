import json
from pathlib import Path
import pickle
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import mean_squared_error
import math

#  definimos las rutas
base_dir = Path("/home/donrobot/Projects/Tesis_antiguo")
models_dir = base_dir / "resultados/entrenamiento/dataset_entrenamiento_1/models"
test_image_path = base_dir / "COVID-19_Radiography_Dataset" / "Normal" / "images" / "Normal-2.png"
json_path = base_dir / "resultados/analisis_regiones/dataset_entrenamiento_1/analisis/template_analysis_results.json"
output_dir = base_dir / "resultados/prediccion"  
output_dir.mkdir(parents=True, exist_ok=True)  # Crear directorio padre

# tamaño de las imagenes en el entrenamiento (altura, ancho)
coord1_crop_size = (45, 46) 
coord2_crop_size = (35, 46)  

# definimos el centroide (y,x)
coord1_centroid_local = (0, 24) 
coord2_centroid_local = (35, 24) 

# definimos el tamaño a escalar la imagen de prueba
resized_image_size = (64, 64)

#cargamos la imagen de prubea y la convertimos a escala de grises
def load_image(image_path):
    """Loads an image and converts it to grayscale."""
    img = Image.open(image_path).convert('L')
    return img

#reescalamos la imagen 
def resize_image(image, size):
    """Resizes an image to the given size."""
    return image.resize(size)

def load_pca_model(model_path):
    """Loads the full trained PCA model data from a pickle file."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data  # Devuelve el diccionario completo

#cargamos las coordenadas dentro de la region de busqueda
def load_coordinates(json_path):
    """Loads coordinate pairs from a JSON file for coord1 and coord2."""
    with open(json_path, 'r') as f:
        all_coords = json.load(f)
    return all_coords.get('coord1', []), all_coords.get('coord2', [])

def crop_image(image, top_left, size):
    """Crops a region from the image given the top-left corner and size."""
    return image.crop((top_left[1], top_left[0], top_left[1] + size[1], top_left[0] + size[0])) # (left, top, right, bottom)

def image_to_vector(image):
    """Converts a PIL Image object to a 1D NumPy array."""
    return np.array(image).flatten()

def vector_to_image(vector, size):
    """Reshapes a 1D NumPy array back into a PIL Image object."""
    return Image.fromarray(vector.reshape(size).astype(np.uint8))

def calculate_mse(image1, image2):
    """Calculates the Mean Squared Error between two PIL Image objects."""
    vector1 = image_to_vector(image1).astype(np.float64)
    vector2 = image_to_vector(image2).astype(np.float64)
    return mean_squared_error(vector1, vector2)

def calculate_euclidean_distance(image1, image2):
    """Calculates the Euclidean distance between two PIL Image objects."""
    vector1 = image_to_vector(image1).astype(np.float64)
    vector2 = image_to_vector(image2).astype(np.float64)
    return math.sqrt(np.sum((vector1 - vector2)**2))

def apply_pca(image_vector, model_data):
    """Applies the PCA transformation to an image vector with proper preprocessing."""
    # Convertir a float similar al entrenamiento
    float_vector = image_vector.astype(float)
    
    # Restar la media (mean_face) antes de proyectar
    centered_vector = float_vector - model_data['mean_face'].flatten()
    
    # Proyectar en espacio PCA
    return model_data['pca'].transform(centered_vector.reshape(1, -1))[0]

def reconstruct_pca(projected_vector, model_data):
    """Reconstructs the image vector from its PCA projection with proper postprocessing."""
    # Reconstruir en el espacio original centrado
    reconstructed_centered = model_data['pca'].inverse_transform(projected_vector.reshape(1, -1))[0]
    
    # Añadir la media para obtener la imagen reconstruida
    reconstructed = reconstructed_centered + model_data['mean_face'].flatten()
    
    # Asegurar que los valores estén en el rango adecuado para imágenes
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    return reconstructed

def mark_coordinate(image, coordinate, color=(255, 0, 0)):
    """Marca un solo píxel en la imagen a color con el color RGB especificado."""
    image.putpixel((coordinate[1], coordinate[0]), color)
    return image

# Load the test image, resize it
test_image = load_image(test_image_path)
resized_test_image = resize_image(test_image, resized_image_size)
resized_test_image_np = np.array(resized_test_image) # Keep a NumPy array version for cropping

# Load the coordinate pairs for both regions
coord1_coords, coord2_coords = load_coordinates(json_path)

# Load the trained PCA models
pca_model_coord1 = load_pca_model(models_dir / "coord1_model.pkl")
pca_model_coord2 = load_pca_model(models_dir / "coord2_model.pkl")

# Initialize variables to store the best results for each region and metric
min_error_coord1_mse = float('inf')
best_coord1_mse = None
min_error_coord1_euclidean = float('inf')
best_coord1_euclidean = None

min_error_coord2_mse = float('inf')
best_coord2_mse = None
min_error_coord2_euclidean = float('inf')
best_coord2_euclidean = None

# Process region 1 (coord1)
print("Processing region coord1...")
for y_c, x_c in coord1_coords:
    # Calculate the top-left corner for cropping
    top_left_y = y_c - coord1_centroid_local[0]
    top_left_x = x_c - coord1_centroid_local[1]
    top_left = (top_left_y, top_left_x)

    # Check if the crop is within the bounds of the 64x64 image
    if 0 <= top_left_y and top_left_y + coord1_crop_size[0] <= resized_image_size[0] and \
       0 <= top_left_x and top_left_x + coord1_crop_size[1] <= resized_image_size[1]:
        try:
            cropped_region = crop_image(resized_test_image, top_left, coord1_crop_size)
            cropped_vector = image_to_vector(cropped_region).astype(np.float64)

            # Apply PCA and reconstruct
            projected_vector = apply_pca(cropped_vector, pca_model_coord1)
            reconstructed_vector = reconstruct_pca(projected_vector, pca_model_coord1)
            reconstructed_image = vector_to_image(reconstructed_vector, coord1_crop_size)

            # Calculate the errors
            mse = calculate_mse(cropped_region, reconstructed_image)
            euclidean_distance = calculate_euclidean_distance(cropped_region, reconstructed_image)

            # Update the best results if the current error is smaller
            if mse < min_error_coord1_mse:
                min_error_coord1_mse = mse
                best_coord1_mse = (y_c, x_c)
            if euclidean_distance < min_error_coord1_euclidean:
                min_error_coord1_euclidean = euclidean_distance
                best_coord1_euclidean = (y_c, x_c)

        except Exception as e:
            print(f"Error processing coord1 at ({y_c}, {x_c}): {e}")

# Process region 2 (coord2)
print("Processing region coord2...")
for y_c, x_c in coord2_coords:
    # Calculate the top-left corner for cropping
    top_left_y = y_c - coord2_centroid_local[0]
    top_left_x = x_c - coord2_centroid_local[1]
    top_left = (top_left_y, top_left_x)

    # Check if the crop is within the bounds of the 64x64 image
    if 0 <= top_left_y and top_left_y + coord2_crop_size[0] <= resized_image_size[0] and \
       0 <= top_left_x and top_left_x + coord2_crop_size[1] <= resized_image_size[1]:
        try:
            cropped_region = crop_image(resized_test_image, top_left, coord2_crop_size)
            cropped_vector = image_to_vector(cropped_region).astype(np.float64)

            # Apply PCA and reconstruct
            projected_vector = apply_pca(cropped_vector, pca_model_coord2)
            reconstructed_vector = reconstruct_pca(projected_vector, pca_model_coord2)
            reconstructed_image = vector_to_image(reconstructed_vector, coord2_crop_size)

            # Calculate the errors
            mse = calculate_mse(cropped_region, reconstructed_image)
            euclidean_distance = calculate_euclidean_distance(cropped_region, reconstructed_image)

            # Update the best results if the current error is smaller
            if mse < min_error_coord2_mse:
                min_error_coord2_mse = mse
                best_coord2_mse = (y_c, x_c)
            if euclidean_distance < min_error_coord2_euclidean:
                min_error_coord2_euclidean = euclidean_distance
                best_coord2_euclidean = (y_c, x_c)

        except Exception as e:
            print(f"Error processing coord2 at ({y_c}, {x_c}): {e}")

# Convertir la imagen de escala de grises a RGB para marcar los puntos con colores
output_image = resized_test_image.convert('RGB')

# Mark the best coordinates on the original resized test image
if best_coord1_mse:
    output_image = mark_coordinate(output_image, best_coord1_mse, color=(0, 255, 0))  
    print(f"Mejor coord1 (MSE): {best_coord1_mse}, Error: {min_error_coord1_mse:.4f}")
else:
    print("No valid coordinates found for coord1 (MSE).")

if best_coord1_euclidean:
    output_image = mark_coordinate(output_image, best_coord1_euclidean, color=(255, 0, 0))  
    print(f"Mejor coord1 (Euclidiana): {best_coord1_euclidean}, Distancia: {min_error_coord1_euclidean:.4f}")
else:
    print("No valid coordinates found for coord1 (Euclidean).")

if best_coord2_mse:
    output_image = mark_coordinate(output_image, best_coord2_mse, color=(0, 255, 0))  
    print(f"Mejor coord2 (MSE): {best_coord2_mse}, Error: {min_error_coord2_mse:.4f}")
else:
    print("No valid coordinates found for coord2 (MSE).")

if best_coord2_euclidean:
    output_image = mark_coordinate(output_image, best_coord2_euclidean, color=(255, 0, 0))  
    print(f"Mejor coord2 (Euclidiana): {best_coord2_euclidean}, Distancia: {min_error_coord2_euclidean:.4f}")
else:
    print("No valid coordinates found for coord2 (Euclidean).")

# Save the output image
output_image_path = output_dir / "test_image_with_matches.png"
output_image.save(output_image_path)
print(f"Output image saved to: {output_image_path}")