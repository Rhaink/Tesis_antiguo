import json
from pathlib import Path
import pickle
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import time
import os

# Define the paths to the necessary files and directories
base_dir = Path("/home/donrobot/projects/Tesis")
models_dir = base_dir / "models"
test_image_path = base_dir / "COVID-19_Radiography_Dataset" / "COVID" / "images" / "COVID-3.png"
json_path = base_dir / "all_search_coordinates.json"
output_dir = base_dir / "output"  # Create an output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)  # Create parent directories if needed

# Define the expected sizes of the cropped images
coord1_crop_size = (46, 45)  # (height, width)
coord2_crop_size = (46, 35)  # (height, width)

# Define the local centroid anchors for cropping
coord1_centroid_local = (0, 24)  # (y, x)
coord2_centroid_local = (35, 24)  # (y, x)

# Define the size of the resized test image
resized_image_size = (64, 64)

# Crear un directorio para los frames de la animación
frames_dir = output_dir / "animation_frames"
frames_dir.mkdir(parents=True, exist_ok=True)

# Crear un directorio para la visualización de las regiones de búsqueda
search_regions_dir = output_dir / "search_regions"
search_regions_dir.mkdir(parents=True, exist_ok=True)

def load_image(image_path):
    """Loads an image and converts it to grayscale."""
    img = Image.open(image_path).convert('L')
    return img

def resize_image(image, size):
    """Resizes an image to the given size."""
    return image.resize(size)

def load_pca_model(model_path):
    """Loads a trained PCA model from a pickle file."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['pca']

def load_coordinates(json_path):
    """Loads coordinate pairs from a JSON file for coord1 and coord2."""
    with open(json_path, 'r') as f:
        all_coords = json.load(f)
    return all_coords.get('coord1', []), all_coords.get('coord2', [])

def crop_image(image, top_left, size):
    """Crops a region from the image given the top-left corner and size."""
    return image.crop((top_left[1], top_left[0], top_left[1] + size[1], top_left[0] + size[0]))  # (left, top, right, bottom)

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

def apply_pca(image_vector, pca_model):
    """Applies the PCA transformation to an image vector."""
    return pca_model.transform(image_vector.reshape(1, -1))[0]

def reconstruct_pca(projected_vector, pca_model):
    """Reconstructs the image vector from its PCA projection."""
    return pca_model.inverse_transform(projected_vector.reshape(1, -1))[0]

def mark_coordinate(image, coordinate, color='green', marker_size=5):
    """Marks a coordinate on the image with a cross."""
    draw = ImageDraw.Draw(image)
    y, x = coordinate
    draw.line((x - marker_size, y, x + marker_size, y), fill=color, width=2)
    draw.line((x, y - marker_size, x, y + marker_size), fill=color, width=2)
    return image

def visualize_search_region(image, coordinates, centroid_local, crop_size, region_name):
    """
    Visualiza la región completa de búsqueda y marca todas las coordenadas que se evaluarán.
    """
    # Crear una copia de la imagen original para dibujar sobre ella
    img_copy = image.copy()
    img_draw = ImageDraw.Draw(img_copy)
    
    # Convertir a RGB para poder usar colores diferentes
    img_rgb = img_copy.convert('RGB')
    img_draw = ImageDraw.Draw(img_rgb)
    
    # Definir una paleta de colores para el mapa de calor
    error_colors = []  # Se llenará con los valores de error más adelante
    
    # Inicializar un mapa para almacenar los errores de cada coordenada
    error_map = np.full(resized_image_size, np.nan)
    valid_points = []
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Dibujar todas las coordenadas en la imagen
    for i, (y_c, x_c) in enumerate(coordinates):
        # Calcular la esquina superior izquierda para el recorte
        top_left_y = y_c - centroid_local[0]
        top_left_x = x_c - centroid_local[1]
        
        # Verificar si el recorte está dentro de los límites de la imagen 64x64
        if 0 <= top_left_y and top_left_y + crop_size[0] <= resized_image_size[0] and \
           0 <= top_left_x and top_left_x + crop_size[1] <= resized_image_size[1]:
            # Marcar este punto como válido
            valid_points.append((y_c, x_c))
            
            # Dibujar un pequeño punto para esta coordenada
            img_draw.ellipse([(x_c-1, y_c-1), (x_c+1, y_c+1)], fill="red")
            
            # Dibujar el área de recorte para algunas coordenadas (cada 20 puntos)
            if i % 20 == 0:
                img_draw.rectangle(
                    [(top_left_x, top_left_y), (top_left_x + crop_size[1], top_left_y + crop_size[0])],
                    outline="yellow", width=1
                )
    
    # Dibujar la imagen con todas las coordenadas marcadas
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Región de búsqueda {region_name} - {len(valid_points)} puntos válidos")
    axes[0].axis('off')
    
    # Crear una visualización de la cobertura de la región de búsqueda
    coverage_map = np.zeros(resized_image_size)
    for y_c, x_c in valid_points:
        top_left_y = y_c - centroid_local[0]
        top_left_x = x_c - centroid_local[1]
        
        # Marcar el área de recorte en el mapa de cobertura
        for y in range(top_left_y, min(top_left_y + crop_size[0], resized_image_size[0])):
            for x in range(top_left_x, min(top_left_x + crop_size[1], resized_image_size[1])):
                if 0 <= y < resized_image_size[0] and 0 <= x < resized_image_size[1]:
                    coverage_map[y, x] += 1
    
    # Mostrar el mapa de cobertura
    coverage_im = axes[1].imshow(coverage_map, cmap='hot', interpolation='nearest')
    axes[1].set_title(f"Mapa de cobertura - Región {region_name}")
    axes[1].axis('off')
    
    # Añadir una barra de color para el mapa de cobertura
    cbar = plt.colorbar(coverage_im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Número de recortes que cubren cada píxel')
    
    plt.tight_layout()
    plt.savefig(search_regions_dir / f"{region_name}_search_region.png")
    plt.close(fig)
    
    return valid_points

def visualize_iteration(original_img, current_coord, centroid_local, crop_size, iteration, region_name, crop=None, reconstructed=None, error=None):
    """
    Crea una visualización de la iteración actual y la guarda como imagen
    """
    y_c, x_c = current_coord
    
    # Calcular la esquina superior izquierda para el recorte
    top_left_y = y_c - centroid_local[0]
    top_left_x = x_c - centroid_local[1]
    
    # Crear una copia de la imagen original para dibujar sobre ella
    img_copy = original_img.copy()
    img_draw = ImageDraw.Draw(img_copy)
    
    # Dibujar el rectángulo que representa el área de recorte
    img_draw.rectangle(
        [(top_left_x, top_left_y), (top_left_x + crop_size[1], top_left_y + crop_size[0])],
        outline="red", width=1
    )
    
    # Marcar el centroide actual
    img_draw.ellipse([(x_c-3, y_c-3), (x_c+3, y_c+3)], fill="blue")
    
    # Configurar la figura para visualización
    fig, axes = plt.subplots(1, 3 if crop is not None else 1, figsize=(15, 5))
    
    # Mostrar la imagen con el rectángulo de recorte y el centroide
    if crop is not None:
        axes[0].imshow(img_copy, cmap='gray')
        axes[0].set_title(f"Región {region_name} - Iteración {iteration}")
        axes[0].axis('off')
        
        # Mostrar el recorte actual
        axes[1].imshow(crop, cmap='gray')
        axes[1].set_title(f"Recorte ({y_c}, {x_c})")
        axes[1].axis('off')
        
        # Mostrar la reconstrucción PCA
        axes[2].imshow(reconstructed, cmap='gray')
        axes[2].set_title(f"Reconstrucción PCA\nError: {error:.4f}")
        axes[2].axis('off')
    else:
        # Si no hay recorte, solo mostrar la imagen con el rectángulo
        axes.imshow(img_copy, cmap='gray')
        axes.set_title(f"Región {region_name} - Iteración {iteration}")
        axes.axis('off')
    
    # Ajustar el diseño y guardar la figura
    plt.tight_layout()
    plt.savefig(frames_dir / f"{region_name}_iteration_{iteration:05d}.png")
    plt.close(fig)

def visualize_error_progression(region_name, coords, errors):
    """
    Visualiza la progresión del error a lo largo de las iteraciones.
    """
    if not errors:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de línea que muestra la progresión del error
    axes[0].plot(range(len(errors)), errors)
    axes[0].set_title(f"Progresión del error - Región {region_name}")
    axes[0].set_xlabel("Iteración")
    axes[0].set_ylabel("Error (MSE)")
    axes[0].grid(True)
    
    # Marcar el punto de error mínimo
    min_error_idx = errors.index(min(errors))
    axes[0].scatter(min_error_idx, errors[min_error_idx], color='red', s=100, marker='*')
    axes[0].annotate(f"Min: {errors[min_error_idx]:.4f}", 
                    (min_error_idx, errors[min_error_idx]),
                    xytext=(10, -20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    # Crear un histograma de errores
    axes[1].hist(errors, bins=20, color='skyblue', edgecolor='black')
    axes[1].set_title(f"Distribución de errores - Región {region_name}")
    axes[1].set_xlabel("Error (MSE)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{region_name}_error_progression.png")
    plt.close(fig)

# Load the test image, resize it
test_image = load_image(test_image_path)
resized_test_image = resize_image(test_image, resized_image_size)
resized_test_image_np = np.array(resized_test_image)  # Keep a NumPy array version for cropping

# Load the coordinate pairs for both regions
coord1_coords, coord2_coords = load_coordinates(json_path)

# Visualizar las regiones completas de búsqueda antes de procesarlas
print("Visualizando regiones de búsqueda...")
valid_coord1_points = visualize_search_region(resized_test_image, coord1_coords, coord1_centroid_local, coord1_crop_size, "coord1")
valid_coord2_points = visualize_search_region(resized_test_image, coord2_coords, coord2_centroid_local, coord2_crop_size, "coord2")

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

# Listas para almacenar errores para visualización
coord1_errors = []
coord2_errors = []

# Process region 1 (coord1)
print("Processing region coord1...")
for i, (y_c, x_c) in enumerate(coord1_coords):
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
            
            # Almacenar el error para visualización
            coord1_errors.append(mse)

            # Visualizar esta iteración
            visualize_iteration(
                resized_test_image, 
                (y_c, x_c), 
                coord1_centroid_local, 
                coord1_crop_size, 
                i, 
                "coord1",
                crop=cropped_region,
                reconstructed=reconstructed_image,
                error=mse
            )

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
for i, (y_c, x_c) in enumerate(coord2_coords):
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
            
            # Almacenar el error para visualización
            coord2_errors.append(mse)

            # Visualizar esta iteración
            visualize_iteration(
                resized_test_image, 
                (y_c, x_c), 
                coord2_centroid_local, 
                coord2_crop_size, 
                i, 
                "coord2",
                crop=cropped_region,
                reconstructed=reconstructed_image,
                error=mse
            )

            # Update the best results if the current error is smaller
            if mse < min_error_coord2_mse:
                min_error_coord2_mse = mse
                best_coord2_mse = (y_c, x_c)
            if euclidean_distance < min_error_coord2_euclidean:
                min_error_coord2_euclidean = euclidean_distance
                best_coord2_euclidean = (y_c, x_c)

        except Exception as e:
            print(f"Error processing coord2 at ({y_c}, {x_c}): {e}")

# Visualizar la progresión de errores
visualize_error_progression("coord1", coord1_coords, coord1_errors)
visualize_error_progression("coord2", coord2_coords, coord2_errors)

# Mark the best coordinates on the original resized test image
output_image = resized_test_image.copy()

if best_coord1_mse:
    output_image = mark_coordinate(output_image, best_coord1_mse, color='green', marker_size=7)
    print(f"Best coord1 (MSE): {best_coord1_mse} with error: {min_error_coord1_mse:.4f}")
else:
    print("No valid coordinates found for coord1 (MSE).")

if best_coord1_euclidean:
    output_image = mark_coordinate(output_image, best_coord1_euclidean, color='lime', marker_size=5)  # Use a slightly different green
    print(f"Best coord1 (Euclidean): {best_coord1_euclidean} with distance: {min_error_coord1_euclidean:.4f}")
else:
    print("No valid coordinates found for coord1 (Euclidean).")

if best_coord2_mse:
    output_image = mark_coordinate(output_image, best_coord2_mse, color='blue', marker_size=7)  # Use a different color for region 2
    print(f"Best coord2 (MSE): {best_coord2_mse} with error: {min_error_coord2_mse:.4f}")
else:
    print("No valid coordinates found for coord2 (MSE).")

if best_coord2_euclidean:
    output_image = mark_coordinate(output_image, best_coord2_euclidean, color='cyan', marker_size=5)  # Use a slightly different blue
    print(f"Best coord2 (Euclidean): {best_coord2_euclidean} with distance: {min_error_coord2_euclidean:.4f}")
else:
    print("No valid coordinates found for coord2 (Euclidean).")

# Save the output image
output_image_path = output_dir / "test_image_with_matches.png"
output_image.save(output_image_path)
print(f"Output image saved to: {output_image_path}")

# Create an animation from all the frames
def create_animation(region, output_path):
    # Get all frames for this region
    frames = sorted([str(f) for f in frames_dir.glob(f"{region}_iteration_*.png")])
    
    if not frames:
        print(f"No frames found for region {region}")
        return
    
    print(f"Creating animation for {region} with {len(frames)} frames...")
    
    # Crear la figura para la animación
    fig = plt.figure(figsize=(12, 4))
    
    # Función para actualizar cada frame
    def update(frame_number):
        plt.clf()
        img = plt.imread(frames[frame_number])
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{region} - Frame {frame_number}/{len(frames)-1}")
    
    # Crear la animación
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100)
    
    # Guardar como GIF o MP4
    ani.save(output_path, writer='pillow', fps=5)
    plt.close(fig)

# Crear animaciones para ambas regiones
create_animation("coord1", output_dir / "coord1_animation.gif")
create_animation("coord2", output_dir / "coord2_animation.gif")

# Crear visualización para mostrar cómo evoluciona el error en la imagen
def visualize_error_map():
    """
    Crea un mapa de error que muestra cómo varía el error a través de la imagen.
    """
    # Crear mapas de error para ambas regiones
    error_map_coord1 = np.full(resized_image_size, np.nan)
    error_map_coord2 = np.full(resized_image_size, np.nan)
    
    # Llenar el mapa de error para coord1
    for i, (y_c, x_c) in enumerate(coord1_coords):
        top_left_y = y_c - coord1_centroid_local[0]
        top_left_x = x_c - coord1_centroid_local[1]
        
        if 0 <= top_left_y and top_left_y + coord1_crop_size[0] <= resized_image_size[0] and \
           0 <= top_left_x and top_left_x + coord1_crop_size[1] <= resized_image_size[1]:
            try:
                cropped_region = crop_image(resized_test_image, (top_left_y, top_left_x), coord1_crop_size)
                cropped_vector = image_to_vector(cropped_region).astype(np.float64)
                projected_vector = apply_pca(cropped_vector, pca_model_coord1)
                reconstructed_vector = reconstruct_pca(projected_vector, pca_model_coord1)
                reconstructed_image = vector_to_image(reconstructed_vector, coord1_crop_size)
                error = calculate_mse(cropped_region, reconstructed_image)
                error_map_coord1[y_c, x_c] = error
            except Exception:
                pass
    
    # Llenar el mapa de error para coord2
    for i, (y_c, x_c) in enumerate(coord2_coords):
        top_left_y = y_c - coord2_centroid_local[0]
        top_left_x = x_c - coord2_centroid_local[1]
        
        if 0 <= top_left_y and top_left_y + coord2_crop_size[0] <= resized_image_size[0] and \
           0 <= top_left_x and top_left_x + coord2_crop_size[1] <= resized_image_size[1]:
            try:
                cropped_region = crop_image(resized_test_image, (top_left_y, top_left_x), coord2_crop_size)
                cropped_vector = image_to_vector(cropped_region).astype(np.float64)
                projected_vector = apply_pca(cropped_vector, pca_model_coord2)
                reconstructed_vector = reconstruct_pca(projected_vector, pca_model_coord2)
                reconstructed_image = vector_to_image(reconstructed_vector, coord2_crop_size)
                error = calculate_mse(cropped_region, reconstructed_image)
                error_map_coord2[y_c, x_c] = error
            except Exception:
                pass
    
    # Crear visualización de mapa de error
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Mostrar la imagen original
    axes[0].imshow(resized_test_image, cmap='gray')
    axes[0].set_title("Imagen original")
    axes[0].axis('off')
    
    # Mostrar el mapa de error para coord1
    error_map1 = axes[1].imshow(error_map_coord1, cmap='hot', interpolation='nearest')
    axes[1].set_title("Mapa de error para coord1")
    axes[1].axis('off')
    cbar1 = plt.colorbar(error_map1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label('Error (MSE)')
    
    # Mostrar el mapa de error para coord2
    error_map2 = axes[2].imshow(error_map_coord2, cmap='hot', interpolation='nearest')
    axes[2].set_title("Mapa de error para coord2")
    axes[2].axis('off')
    cbar2 = plt.colorbar(error_map2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label('Error (MSE)')
    
    # Marcar los puntos de error mínimo
    if best_coord1_mse:
        y, x = best_coord1_mse
        axes[1].plot(x, y, 'o', markerfacecolor='none', markeredgecolor='cyan', markersize=10, markeredgewidth=2)
    
    if best_coord2_mse:
        y, x = best_coord2_mse
        axes[2].plot(x, y, 'o', markerfacecolor='none', markeredgecolor='cyan', markersize=10, markeredgewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / "error_maps.png")
    plt.close(fig)

# Crear una visualización final que muestre los mejores resultados
def visualize_best_results():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image with all best coordinates marked
    axes[0, 0].imshow(output_image, cmap='gray')
    axes[0, 0].set_title("Imagen con mejores coincidencias")
    axes[0, 0].axis('off')
    
# Best coord1 (MSE)
    if best_coord1_mse:
        y_c, x_c = best_coord1_mse
        top_left_y = y_c - coord1_centroid_local[0]
        top_left_x = x_c - coord1_centroid_local[1]
        
        # Crop best region
        best_crop = crop_image(resized_test_image, (top_left_y, top_left_x), coord1_crop_size)
        
        # Show it
        axes[0, 1].imshow(best_crop, cmap='gray')
        axes[0, 1].set_title(f"Mejor coord1 (MSE)\nCoordenadas: {best_coord1_mse}\nError: {min_error_coord1_mse:.4f}")
        axes[0, 1].axis('off')
        
        # Show reconstruction
        cropped_vector = image_to_vector(best_crop).astype(np.float64)
        projected_vector = apply_pca(cropped_vector, pca_model_coord1)
        reconstructed_vector = reconstruct_pca(projected_vector, pca_model_coord1)
        reconstructed_image = vector_to_image(reconstructed_vector, coord1_crop_size)
        
        axes[0, 2].imshow(reconstructed_image, cmap='gray')
        axes[0, 2].set_title("Reconstrucción PCA (coord1)")
        axes[0, 2].axis('off')
    
    # Best coord2 (MSE)
    if best_coord2_mse:
        y_c, x_c = best_coord2_mse
        top_left_y = y_c - coord2_centroid_local[0]
        top_left_x = x_c - coord2_centroid_local[1]
        
        # Crop best region
        best_crop = crop_image(resized_test_image, (top_left_y, top_left_x), coord2_crop_size)
        
        # Show it
        axes[1, 1].imshow(best_crop, cmap='gray')
        axes[1, 1].set_title(f"Mejor coord2 (MSE)\nCoordenadas: {best_coord2_mse}\nError: {min_error_coord2_mse:.4f}")
        axes[1, 1].axis('off')
        
        # Show reconstruction
        cropped_vector = image_to_vector(best_crop).astype(np.float64)
        projected_vector = apply_pca(cropped_vector, pca_model_coord2)
        reconstructed_vector = reconstruct_pca(projected_vector, pca_model_coord2)
        reconstructed_image = vector_to_image(reconstructed_vector, coord2_crop_size)
        
        axes[1, 2].imshow(reconstructed_image, cmap='gray')
        axes[1, 2].set_title("Reconstrucción PCA (coord2)")
        axes[1, 2].axis('off')
    
    # Clear unused axes
    if not best_coord1_mse:
        fig.delaxes(axes[0, 1])
        fig.delaxes(axes[0, 2])
    if not best_coord2_mse:
        fig.delaxes(axes[1, 1])
        fig.delaxes(axes[1, 2])
    
    # Mostrar mapa de errores
    error_map_coord1 = np.full(resized_image_size, np.nan)
    for y_c, x_c in coord1_coords:
        top_left_y = y_c - coord1_centroid_local[0]
        top_left_x = x_c - coord1_centroid_local[1]
        
        if 0 <= top_left_y and top_left_y + coord1_crop_size[0] <= resized_image_size[0] and \
           0 <= top_left_x and top_left_x + coord1_crop_size[1] <= resized_image_size[1]:
            try:
                cropped_region = crop_image(resized_test_image, (top_left_y, top_left_x), coord1_crop_size)
                cropped_vector = image_to_vector(cropped_region).astype(np.float64)
                projected_vector = apply_pca(cropped_vector, pca_model_coord1)
                reconstructed_vector = reconstruct_pca(projected_vector, pca_model_coord1)
                reconstructed_image = vector_to_image(reconstructed_vector, coord1_crop_size)
                error = calculate_mse(cropped_region, reconstructed_image)
                error_map_coord1[y_c, x_c] = error
            except Exception:
                pass
    
    # Plot error map
    axes[1, 0].imshow(error_map_coord1, cmap='hot', interpolation='nearest')
    axes[1, 0].set_title("Mapa de errores (MSE)")
    axes[1, 0].axis('off')
    
    # Marcar el punto de mejor coincidencia en el mapa de errores
    if best_coord1_mse:
        y, x = best_coord1_mse
        axes[1, 0].plot(x, y, 'o', markerfacecolor='none', markeredgecolor='cyan', markersize=10, markeredgewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / "best_results_summary.png")
    plt.close(fig)

# Crear un dashboard interactivo que muestra todas las coordenadas evaluadas
def create_interactive_visual():
    """
    Crea una visualización interactiva que muestra todas las coordenadas evaluadas
    para una mejor comprensión del proceso de búsqueda.
    """
    # Primero, crear un mapa que muestre todas las coordenadas evaluadas con su error asociado
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Visualización de la región 1 - Mapa de puntos
    axes[0, 0].imshow(resized_test_image, cmap='gray')
    axes[0, 0].set_title("Coordenadas evaluadas - Región 1")
    
    # Crear una lista de errores para determinar el color
    errors1 = []
    valid_coords1 = []
    
    for y_c, x_c in coord1_coords:
        top_left_y = y_c - coord1_centroid_local[0]
        top_left_x = x_c - coord1_centroid_local[1]
        
        if 0 <= top_left_y and top_left_y + coord1_crop_size[0] <= resized_image_size[0] and \
           0 <= top_left_x and top_left_x + coord1_crop_size[1] <= resized_image_size[1]:
            try:
                cropped_region = crop_image(resized_test_image, (top_left_y, top_left_x), coord1_crop_size)
                cropped_vector = image_to_vector(cropped_region).astype(np.float64)
                projected_vector = apply_pca(cropped_vector, pca_model_coord1)
                reconstructed_vector = reconstruct_pca(projected_vector, pca_model_coord1)
                reconstructed_image = vector_to_image(reconstructed_vector, coord1_crop_size)
                error = calculate_mse(cropped_region, reconstructed_image)
                
                valid_coords1.append((y_c, x_c))
                errors1.append(error)
            except Exception:
                continue
    
    # Normalizar los errores para la escala de colores
    if errors1:
        norm = plt.Normalize(min(errors1), max(errors1))
        sc = axes[0, 0].scatter([x for y, x in valid_coords1], [y for y, x in valid_coords1], 
                              c=errors1, cmap='viridis_r', norm=norm, alpha=0.7, s=30)
        plt.colorbar(sc, ax=axes[0, 0], label='Error (MSE)')
    
    # Marcar el mejor punto
    if best_coord1_mse:
        y, x = best_coord1_mse
        axes[0, 0].plot(x, y, 'o', markerfacecolor='none', markeredgecolor='red', markersize=15, markeredgewidth=2)
        
        # Dibujar el rectángulo del área de recorte para el mejor punto
        top_left_y = y - coord1_centroid_local[0]
        top_left_x = x - coord1_centroid_local[1]
        rect = Rectangle((top_left_x, top_left_y), coord1_crop_size[1], coord1_crop_size[0], 
                        edgecolor='red', facecolor='none', linewidth=2)
        axes[0, 0].add_patch(rect)
    
    axes[0, 0].axis('off')
    
    # Visualización de la región 2 - Mapa de puntos
    axes[0, 1].imshow(resized_test_image, cmap='gray')
    axes[0, 1].set_title("Coordenadas evaluadas - Región 2")
    
    # Crear una lista de errores para determinar el color
    errors2 = []
    valid_coords2 = []
    
    for y_c, x_c in coord2_coords:
        top_left_y = y_c - coord2_centroid_local[0]
        top_left_x = x_c - coord2_centroid_local[1]
        
        if 0 <= top_left_y and top_left_y + coord2_crop_size[0] <= resized_image_size[0] and \
           0 <= top_left_x and top_left_x + coord2_crop_size[1] <= resized_image_size[1]:
            try:
                cropped_region = crop_image(resized_test_image, (top_left_y, top_left_x), coord2_crop_size)
                cropped_vector = image_to_vector(cropped_region).astype(np.float64)
                projected_vector = apply_pca(cropped_vector, pca_model_coord2)
                reconstructed_vector = reconstruct_pca(projected_vector, pca_model_coord2)
                reconstructed_image = vector_to_image(reconstructed_vector, coord2_crop_size)
                error = calculate_mse(cropped_region, reconstructed_image)
                
                valid_coords2.append((y_c, x_c))
                errors2.append(error)
            except Exception:
                continue
    
    # Normalizar los errores para la escala de colores
    if errors2:
        norm = plt.Normalize(min(errors2), max(errors2))
        sc = axes[0, 1].scatter([x for y, x in valid_coords2], [y for y, x in valid_coords2], 
                              c=errors2, cmap='viridis_r', norm=norm, alpha=0.7, s=30)
        plt.colorbar(sc, ax=axes[0, 1], label='Error (MSE)')
    
    # Marcar el mejor punto
    if best_coord2_mse:
        y, x = best_coord2_mse
        axes[0, 1].plot(x, y, 'o', markerfacecolor='none', markeredgecolor='red', markersize=15, markeredgewidth=2)
        
        # Dibujar el rectángulo del área de recorte para el mejor punto
        top_left_y = y - coord2_centroid_local[0]
        top_left_x = x - coord2_centroid_local[1]
        rect = Rectangle((top_left_x, top_left_y), coord2_crop_size[1], coord2_crop_size[0], 
                        edgecolor='red', facecolor='none', linewidth=2)
        axes[0, 1].add_patch(rect)
    
    axes[0, 1].axis('off')
    
    # Crear mapas de calor 3D para mejor visualización de la distribución de errores
    # Para la región 1
    ax3d1 = fig.add_subplot(2, 2, 3, projection='3d')
    
    if valid_coords1:
        xs = [x for y, x in valid_coords1]
        ys = [y for y, x in valid_coords1]
        zs = errors1
        
        # Gráfico de superficie con los errores
        surf = ax3d1.plot_trisurf(xs, ys, zs, cmap='viridis_r', edgecolor='none', alpha=0.8)
        
        # Marcar el punto de menor error
        if best_coord1_mse:
            y, x = best_coord1_mse
            min_error = min_error_coord1_mse
            ax3d1.scatter([x], [y], [min_error], color='red', s=100, marker='*')
    
    ax3d1.set_title("Distribución 3D de errores - Región 1")
    ax3d1.set_xlabel('X')
    ax3d1.set_ylabel('Y')
    ax3d1.set_zlabel('Error MSE')
    
    # Para la región 2
    ax3d2 = fig.add_subplot(2, 2, 4, projection='3d')
    
    if valid_coords2:
        xs = [x for y, x in valid_coords2]
        ys = [y for y, x in valid_coords2]
        zs = errors2
        
        # Gráfico de superficie con los errores
        surf = ax3d2.plot_trisurf(xs, ys, zs, cmap='viridis_r', edgecolor='none', alpha=0.8)
        
        # Marcar el punto de menor error
        if best_coord2_mse:
            y, x = best_coord2_mse
            min_error = min_error_coord2_mse
            ax3d2.scatter([x], [y], [min_error], color='red', s=100, marker='*')
    
    ax3d2.set_title("Distribución 3D de errores - Región 2")
    ax3d2.set_xlabel('X')
    ax3d2.set_ylabel('Y')
    ax3d2.set_zlabel('Error MSE')
    
    plt.tight_layout()
    plt.savefig(output_dir / "interactive_visualization.png", dpi=300)
    plt.close(fig)

# Visualizar los mapas de error
visualize_error_map()

# Visualizar los mejores resultados
visualize_best_results()

# Crear visualización interactiva
create_interactive_visual()

print("\nVisualización completa. Resultados guardados en:", output_dir)
print("Frames de animación guardados en:", frames_dir)
print("Visualizaciones de regiones de búsqueda guardadas en:", search_regions_dir)
print("Animaciones guardadas como GIFs en:", output_dir)