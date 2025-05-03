# Tesis/alineamiento/visualize_data.py

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import logging
# Asegúrate que la importación funcione desde donde ejecutas el script
try:
    from src.data_loader import load_all_data 
except ImportError:
    # Si ejecutas directamente desde Tesis/alineamiento/
    from alineamiento.src.data_loader import load_all_data

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuración ---
# Directorio base (Tesis) - Ajustado para funcionar si se ejecuta desde Tesis/alineamiento/
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directorio alineamiento/
Tesis_DIR = os.path.dirname(BASE_DIR) # Directorio Tesis/
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
NUM_SAMPLE_IMAGES = 5 
LANDMARK_ORIGINAL_SIZE = 64 # Las coordenadas están en este espacio (64x64)
# --------------------

def plot_landmarks_on_image(image_path, landmarks_64, output_path, original_coord_size=LANDMARK_ORIGINAL_SIZE):
    """
    Dibuja landmarks (escalados desde 64x64) en una imagen y la guarda.
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Leer en escala de grises
        if image is None:
            logging.warning(f"No se pudo cargar la imagen: {image_path}")
            return

        # Obtener dimensiones reales de la imagen
        H_img, W_img = image.shape[:2]
        logging.debug(f"Dimensiones de {os.path.basename(image_path)}: {W_img}x{H_img}")

        # Calcular factores de escala
        scale_x = W_img / float(original_coord_size)
        scale_y = H_img / float(original_coord_size)
        logging.debug(f"Factores de escala: sx={scale_x:.4f}, sy={scale_y:.4f}")

        # Convertir a color para dibujar landmarks en color
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Dibujar cada landmark escalado
        for i in range(landmarks_64.shape[0]): # Iterar sobre k=15 landmarks
            x_64, y_64 = landmarks_64[i, 0], landmarks_64[i, 1]

            # Escalar coordenadas
            x_scaled = x_64 * scale_x
            y_scaled = y_64 * scale_y

            # Convertir a enteros para dibujar
            x_img, y_img = int(round(x_scaled)), int(round(y_scaled))

            # Validar coordenadas *escaladas* antes de dibujar
            if 0 <= x_img < W_img and 0 <= y_img < H_img:
                 # Dibujar un círculo rojo más visible
                 cv2.circle(image_color, (x_img, y_img), radius=3, color=(0, 0, 255), thickness=-1) # Círculo rojo relleno (radio 3)
                 # Opcional: añadir número de landmark
                 # cv2.putText(image_color, str(i), (x_img + 2, y_img + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            else:
                 logging.warning(f"Coordenada escalada ({x_img},{y_img}) fuera de los límites [{W_img}x{H_img}] para landmark {i} en {os.path.basename(image_path)}")

        # Guardar la imagen resultante
        success = cv2.imwrite(output_path, image_color)
        if success:
            logging.info(f"Imagen con landmarks (escalados) guardada en: {output_path}")
        else:
             logging.error(f"No se pudo guardar la imagen en: {output_path}")

    except Exception as e:
        logging.error(f"Error al procesar/dibujar landmarks en {image_path}: {e}")

# La función plot_raw_centered_landmarks NO necesita cambios, ya que
# opera en el espacio de coordenadas abstracto, no en píxeles de imagen.
def plot_raw_centered_landmarks(all_landmarks, output_path):
    """Centra y superpone todos los conjuntos de landmarks (sin cambios)."""
    if all_landmarks is None or len(all_landmarks) == 0:
        logging.error("No hay landmarks para graficar.")
        return
        
    logging.info("Generando gráfico de landmarks crudos centrados superpuestos...")
    plt.figure(figsize=(8, 8))
    
    all_centered_landmarks = []

    for i in range(all_landmarks.shape[0]):
        landmarks = all_landmarks[i, :, :] # Shape (15, 2)
        centroid = np.mean(landmarks, axis=0)
        centered_landmarks = landmarks - centroid
        all_centered_landmarks.append(centered_landmarks)
        plt.plot(centered_landmarks[:, 0], centered_landmarks[:, 1], marker='.', linestyle='', ms=2, alpha=0.1, color='blue')

    if all_centered_landmarks:
        mean_raw_centered_shape = np.mean(np.array(all_centered_landmarks), axis=0)
        plt.plot(mean_raw_centered_shape[:, 0], mean_raw_centered_shape[:, 1], marker='o', linestyle='', ms=4, color='red', label='Forma Media Cruda Centrada')

    plt.title('Landmarks Crudos Centrados Superpuestos (N={})'.format(all_landmarks.shape[0]))
    plt.xlabel('Coordenada X (centrada, espacio 64x64)') # Aclarar espacio
    plt.ylabel('Coordenada Y (centrada, espacio 64x64)')
    plt.gca().invert_yaxis() 
    plt.axis('equal') 
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    try:
        plt.savefig(output_path)
        logging.info(f"Gráfico de landmarks superpuestos guardado en: {output_path}")
    except Exception as e:
        logging.error(f"No se pudo guardar el gráfico en {output_path}: {e}")
    plt.close()

def main():
    """Función principal para ejecutar las visualizaciones (sin cambios en lógica principal)."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    logging.info("--- Iniciando Visualización de Datos (Corregida) ---")
    
    # Cargar datos
    # Usar Tesis_DIR para asegurar que encuentra los datos correctamente
    index_map, landmarks_array, image_paths = load_all_data(Tesis_DIR) 

    if index_map is None or landmarks_array is None or not image_paths:
        logging.error("Fallo al cargar los datos. Abortando visualización.")
        return

    # --- Plot 1: Landmarks en Imágenes de Muestra (usará la función corregida) ---
    logging.info("Generando imágenes de muestra con landmarks (escalados)...")
    available_indices = list(image_paths.keys())
    if not available_indices:
         logging.error("No hay imágenes válidas para mostrar.")
         return
         
    num_to_plot = min(NUM_SAMPLE_IMAGES, len(available_indices))
    sample_indices = random.sample(available_indices, num_to_plot) 

    for idx in sample_indices:
        try:
            img_path = image_paths[idx]
            # Asumiendo que landmarks_array tiene el mismo orden 0..N-1 que los índices originales
            if 0 <= idx < landmarks_array.shape[0]:
                 landmark_set_64 = landmarks_array[idx] # Coordenadas originales 64x64
                 
                 img_filename = os.path.basename(img_path)
                 output_filename = f"sample_{img_filename.replace('.png', '')}_landmarks_scaled.png" # Nuevo nombre
                 output_filepath = os.path.join(PLOTS_DIR, output_filename)
                 
                 # Llamar a la función corregida
                 plot_landmarks_on_image(img_path, landmark_set_64, output_filepath) 
            else:
                logging.error(f"Índice {idx} fuera de rango para landmarks_array ({landmarks_array.shape[0]} filas).")

        except KeyError:
            logging.error(f"Índice {idx} no encontrado en image_paths.")
        except Exception as e:
             logging.error(f"Error inesperado procesando muestra con índice {idx}: {e}")

    # --- Plot 2: Superposición de Landmarks Crudos Centrados (sin cambios) ---
    output_overlay_path = os.path.join(PLOTS_DIR, "raw_centered_landmarks_overlay.png")
    plot_raw_centered_landmarks(landmarks_array, output_overlay_path)

    logging.info("--- Visualización de Datos (Corregida) Finalizada ---")

if __name__ == "__main__":
    # Ajustar CWD temporalmente o usar rutas absolutas si hay problemas de importación
    # print(f"Current working directory: {os.getcwd()}")
    main()