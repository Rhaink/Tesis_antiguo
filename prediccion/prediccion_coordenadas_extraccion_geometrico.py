# prediccion_coordenadas_extraccion_datos.py (Versión Multi-Transformación)

import json
from pathlib import Path
import pickle
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import mean_squared_error
import math
import os
import cv2  # Necesario para transformaciones
import itertools

# --- Rutas y Configuraciones ---
base_dir = Path("/home/donrobot/projects/Tesis") # Ajusta si es necesario
models_dir = base_dir / "resultados/entrenamiento/dataset_aligned_maestro_1/models"
images_dir = base_dir / "dataset" / "dataset_aligned_prueba_1" # Directorio con imágenes de prueba
json_path = base_dir / "resultados/region_busqueda/dataset_aligned_maestro_1/json/all_search_coordinates.json" # Coords. de búsqueda alineadas

# Directorio base para los resultados de predicción
output_base_dir = base_dir / "resultados/prediccion/dataset_aligned_maestro_1"
# Subdirectorios específicos para visualización y JSON (con sufijo _multi_tf)
output_viz_dir = output_base_dir / "lote_multi_tf" / "viz"
output_json_dir = output_base_dir / "lote_multi_tf" / "json"
output_viz_dir.mkdir(parents=True, exist_ok=True)
output_json_dir.mkdir(parents=True, exist_ok=True)

# Tamaño de la imagen y centro (consistente con align_dataset.py)
IMG_SIZE = 64
IMG_CENTER = (IMG_SIZE / 2 - 0.5, IMG_SIZE / 2 - 0.5) # (31.5, 31.5)
resized_image_size = (IMG_SIZE, IMG_SIZE) # (height, width)

# Tamaños de crops y centroides locales (de template_analyzer.py)
coord1_crop_size = (38, 62) # (height, width)
coord2_crop_size = (39, 62) # (height, width)
coord1_centroid_local = (0, 31) # (y_offset, x_offset)
coord2_centroid_local = (38, 31) # (y_offset, x_offset)

# --- Configuraciones para Múltiples Transformaciones ---
# Angulos (de -30 a +30, paso 1) - Usando range es más práctico aquí
ANGLE_STEP = 1

# Define qué rotaciones y traslaciones probar
ANGLES_TO_TRY = list(range(-5, 5 + ANGLE_STEP, ANGLE_STEP)) # 61 ángulos
TRANSLATIONS_TO_TRY = list(itertools.product(range(-3, 3 + 1, 1), repeat=2))
# ANGLES_TO_TRY = [0] # Para probar solo traslaciones
# TRANSLATIONS_TO_TRY = [[0,0]] # Para probar solo rotaciones (o ninguna tf si angle=0)


# --- Funciones Auxiliares ---

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

def resize_image(image, size_hw):
    """Redimensiona una imagen PIL al tamaño dado (height, width)."""
    return image.resize((size_hw[1], size_hw[0])) # PIL usa (width, height)

def load_pca_model(model_path):
    """Carga el modelo PCA desde un archivo pickle."""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        # Validar contenido mínimo esperado
        if 'pca' not in model_data or 'mean_face' not in model_data:
             print(f"Error: Modelo {model_path} no contiene 'pca' o 'mean_face'.")
             return None
        return model_data
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de modelo: {model_path}")
        return None
    except Exception as e:
        print(f"Error al cargar el modelo {model_path}: {e}")
        return None

def load_search_coordinates(json_path):
    """Carga las coordenadas de búsqueda [y,x] (del espacio alineado) desde JSON."""
    try:
        with open(json_path, 'r') as f:
            all_coords = json.load(f)
        coord1 = all_coords.get('coord1', [])
        coord2 = all_coords.get('coord2', [])
        print(f"Coordenadas de búsqueda cargadas: Coord1={len(coord1)} puntos, Coord2={len(coord2)} puntos.")
        return coord1, coord2
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo JSON de coordenadas: {json_path}")
        return [], []
    except json.JSONDecodeError:
        print(f"Error: El archivo JSON {json_path} está mal formado.")
        return [], []
    except Exception as e:
        print(f"Error al cargar coordenadas desde {json_path}: {e}")
        return [], []

def create_delta_transform(angle_deg, translation_xy, center_xy):
    """Crea matriz afín 2x3 para rotación(angle_deg) حول center_xy + traslación(translation_xy)."""
    M_rot = cv2.getRotationMatrix2D(center_xy, angle_deg, 1.0)
    tx, ty = translation_xy
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    M_rot_h = np.vstack([M_rot, [0, 0, 1]])
    M_trans_h = np.vstack([M_trans, [0, 0, 1]])
    M_final_h = M_trans_h @ M_rot_h
    M_final = M_final_h[0:2, 0:3]
    return M_final

def invert_affine_transform(M):
    """Calcula la matriz de transformación afín inversa."""
    if M is None: return None
    a, b, tx = M[0]
    c, d, ty = M[1]
    R_sub = np.array([[a, b], [c, d]])
    T_sub = np.array([tx, ty])
    try:
        # Verificar determinante antes de invertir
        det = np.linalg.det(R_sub)
        if abs(det) < 1e-6:
             # print(f"Advertencia: Matriz de rotación/escala singular (det={det}). No se puede invertir.")
             return None
        R_sub_inv = np.linalg.inv(R_sub)
    except np.linalg.LinAlgError:
        # print("Advertencia: Error LinAlg al invertir matriz de rotación/escala.")
        return None
    T_sub_inv = -R_sub_inv @ T_sub
    M_inv = np.zeros((2, 3))
    M_inv[0:2, 0:2] = R_sub_inv
    M_inv[0, 2] = T_sub_inv[0]
    M_inv[1, 2] = T_sub_inv[1]
    return M_inv

def transform_coordinates(coords_list_yx, M_inv):
    """Aplica M_inv a coords [y, x] y devuelve [y', x'] acotadas."""
    if M_inv is None or not coords_list_yx:
        # Si no hay transformación o lista, devolver original
        return coords_list_yx

    transformed_coords = []
    M_inv_np = np.array(M_inv)
    # Convertir lista [y,x] a array [[x,y],...]
    coords_xy = np.array([[x, y] for y, x in coords_list_yx], dtype=np.float32)
    # Añadir columna de unos para coordenadas homogéneas
    coords_xy_h = np.hstack([coords_xy, np.ones((coords_xy.shape[0], 1))])

    try:
        # Aplicar transformación inversa (M_inv es 2x3)
        transformed_xy = (M_inv_np @ coords_xy_h.T).T # Resultado es [[x', y'], ...]
    except Exception as e:
        print(f"Error durante multiplicación matricial en transform_coordinates: {e}")
        return coords_list_yx # Devolver original en caso de error

    # Convertir de vuelta a lista de [y, x], redondeando y acotando
    img_h, img_w = resized_image_size # Usar tamaño global
    for x_orig, y_orig in transformed_xy:
        y_new = int(round(y_orig))
        x_new = int(round(x_orig))
        y_new_clipped = np.clip(y_new, 0, img_h - 1)
        x_new_clipped = np.clip(x_new, 0, img_w - 1)
        transformed_coords.append([y_new_clipped, x_new_clipped])

    return transformed_coords

def crop_image(image, top_left_yx, size_hw):
    """Recorta una región de la imagen PIL."""
    left = top_left_yx[1]
    upper = top_left_yx[0]
    right = left + size_hw[1] # width
    lower = upper + size_hw[0] # height
    return image.crop((left, upper, right, lower))

def image_to_vector(image):
    """Convierte una imagen PIL a un vector NumPy."""
    return np.array(image).flatten()

def vector_to_image(vector, size_hw):
    """Reconstruye una imagen PIL desde un vector NumPy (height, width)."""
    vector_uint8 = np.clip(vector, 0, 255).astype(np.uint8)
    return Image.fromarray(vector_uint8.reshape(size_hw))

def calculate_mse(image1, image2):
    """Calcula el Mean Squared Error entre dos imágenes PIL."""
    vector1 = image_to_vector(image1).astype(np.float64)
    vector2 = image_to_vector(image2).astype(np.float64)
    if vector1.shape != vector2.shape:
        # print(f"Error MSE: Shapes no coinciden - {vector1.shape} vs {vector2.shape}")
        return float('inf')
    return mean_squared_error(vector1, vector2)

def calculate_euclidean_distance(image1, image2):
    """Calcula la distancia euclidiana L2 entre dos imágenes PIL."""
    vector1 = image_to_vector(image1).astype(np.float64)
    vector2 = image_to_vector(image2).astype(np.float64)
    if vector1.shape != vector2.shape:
        # print(f"Error Euclidiana: Shapes no coinciden - {vector1.shape} vs {vector2.shape}")
        return float('inf')
    return np.linalg.norm(vector1 - vector2)

def apply_pca(image_vector, model_data):
    """Aplica la transformación PCA a un vector de imagen."""
    if model_data is None or 'pca' not in model_data or 'mean_face' not in model_data:
        print("Error: apply_pca recibió modelo inválido.")
        return None
        
    float_vector = image_vector.astype(float)
    mean_face_flat = model_data['mean_face'].flatten()
    pca_model = model_data['pca']

    if float_vector.shape != mean_face_flat.shape:
        # print(f"Error PCA Apply: Shape vector ({float_vector.shape}) != Shape media ({mean_face_flat.shape})")
        return None
        
    centered_vector = float_vector - mean_face_flat
    
    try:
        # Verificar n_features_in_ si está disponible (Scikit-learn >= 0.24)
        if hasattr(pca_model, 'n_features_in_') and centered_vector.shape[0] != pca_model.n_features_in_:
             # print(f"Error PCA Apply: Dimensión vector ({centered_vector.shape[0]}) != n_features_in_ ({pca_model.n_features_in_})")
             return None
        projected = pca_model.transform(centered_vector.reshape(1, -1))
        return projected[0]
    except ValueError as e:
        # print(f"Error en pca.transform: {e}.")
        # print(f"Vector shape: {centered_vector.shape}, n_features_in: {getattr(pca_model, 'n_features_in_', 'N/A')}")
        return None
    except Exception as e_gen:
         print(f"Error inesperado en apply_pca: {e_gen}")
         return None

def reconstruct_pca(projected_vector, model_data):
    """Reconstruye el vector de imagen desde su proyección PCA."""
    if projected_vector is None or model_data is None or 'pca' not in model_data or 'mean_face' not in model_data:
        # print("Error: reconstruct_pca recibió entrada inválida.")
        return None
        
    pca_model = model_data['pca']
    mean_face_flat = model_data['mean_face'].flatten()
    
    try:
        # Verificar n_components_ si está disponible
        if hasattr(pca_model, 'n_components_') and projected_vector.shape[0] != pca_model.n_components_:
            # print(f"Error PCA Reconstruct: Dimensión vector proyectado ({projected_vector.shape[0]}) != n_components_ ({pca_model.n_components_})")
            return None
        reconstructed_centered = pca_model.inverse_transform(projected_vector.reshape(1, -1))
        reconstructed = reconstructed_centered[0] + mean_face_flat
        return reconstructed
    except ValueError as e:
        # print(f"Error en pca.inverse_transform: {e}")
        return None
    except Exception as e_gen:
         print(f"Error inesperado en reconstruct_pca: {e_gen}")
         return None

def mark_coordinate(image, coordinate_yx, color=(255, 0, 0), radius=1):
    """Marca una coordenada (y, x) en la imagen PIL con un color y radio."""
    if coordinate_yx is None: return image # No marcar si no hay coordenada
    draw = ImageDraw.Draw(image)
    y, x = coordinate_yx
    # Asegurar que las coordenadas sean enteras para dibujar
    x_int, y_int = int(round(x)), int(round(y))
    bbox = [x_int - radius, y_int - radius, x_int + radius, y_int + radius]
    try:
        draw.ellipse(bbox, fill=color, outline=color)
    except Exception as e:
        print(f"Error dibujando marca en ({y_int}, {x_int}): {e}")
    return image

def process_region(image, search_coords_yx, centroid_local_yx, crop_size_hw, pca_model, metric='mse'):
    """Busca la mejor coincidencia PCA dentro de las search_coords_yx."""
    min_error = float('inf')
    best_coord_yx = None

    if not search_coords_yx or pca_model is None:
        # print(f"Advertencia: Coordenadas de búsqueda vacías o modelo PCA inválido en process_region.")
        return None, float('inf')

    img_width, img_height = image.size # PIL size es (width, height)

    for y_c, x_c in search_coords_yx:
        top_left_y = y_c - centroid_local_yx[0]
        top_left_x = x_c - centroid_local_yx[1]
        top_left_yx = (top_left_y, top_left_x)

        # Verificar si el crop está dentro de la imagen
        if 0 <= top_left_y and (top_left_y + crop_size_hw[0]) <= img_height and \
           0 <= top_left_x and (top_left_x + crop_size_hw[1]) <= img_width:
            try:
                cropped_region = crop_image(image, top_left_yx, crop_size_hw)
                # Doble check de tamaño por si crop falla en bordes
                if cropped_region.size != (crop_size_hw[1], crop_size_hw[0]):
                    continue

                cropped_vector = image_to_vector(cropped_region)

                projected_vector = apply_pca(cropped_vector, pca_model)
                if projected_vector is None: continue

                reconstructed_vector = reconstruct_pca(projected_vector, pca_model)
                if reconstructed_vector is None: continue

                reconstructed_image = vector_to_image(reconstructed_vector, crop_size_hw)

                if metric == 'mse':
                    error = calculate_mse(cropped_region, reconstructed_image)
                elif metric == 'euclidean':
                    error = calculate_euclidean_distance(cropped_region, reconstructed_image)
                else: # Fallback a MSE si la métrica no es válida
                    error = calculate_mse(cropped_region, reconstructed_image)

                if error < min_error:
                    min_error = error
                    best_coord_yx = (y_c, x_c) # Guardar la coordenada [y, x] del centro candidato

            except Exception as e:
                 # print(f"Excepción procesando coord ({y_c}, {x_c}): {e}")
                 continue # Saltar al siguiente punto si hay error

    # Devuelve None si no se encontró ninguna coincidencia válida
    if best_coord_yx is None:
         min_error = float('inf') # Asegurar que el error sea infinito si no hay coordenada

    return best_coord_yx, min_error


# --- Programa Principal ---

if __name__ == "__main__":
    print("--- Iniciando Predicción de Coordenadas (Múltiples Transformaciones) ---")

    # Carga de modelos y coordenadas de búsqueda (alineadas)
    print("Cargando modelos PCA y coordenadas de búsqueda...")
    # Usar nombres '_aligned' para claridad
    coord1_coords_aligned, coord2_coords_aligned = load_search_coordinates(json_path)
    pca_model_coord1 = load_pca_model(models_dir / "coord1_model.pkl") # Asegúrate que el nombre sea correcto
    pca_model_coord2 = load_pca_model(models_dir / "coord2_model.pkl") # Asegúrate que el nombre sea correcto

    # Verificar cargas
    if not coord1_coords_aligned or not coord2_coords_aligned:
        print("Error Crítico: No se pudieron cargar las coordenadas de búsqueda. Terminando.")
        exit()
    if pca_model_coord1 is None or pca_model_coord2 is None:
        print("Error Crítico: No se pudieron cargar los modelos PCA. Terminando.")
        exit()

    # --- Bucle Principal para Procesar Imágenes de Prueba ---
    results = {}
    print(f"\nProcesando imágenes en: {images_dir}")
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) # Ordenar para consistencia
    print(f"Se encontraron {len(image_files)} imágenes.")
    total_transforms_to_try = len(ANGLES_TO_TRY) * len(TRANSLATIONS_TO_TRY)
    if total_transforms_to_try == 0:
         print("Error: No hay transformaciones definidas en ANGLES_TO_TRY o TRANSLATIONS_TO_TRY. Terminando.")
         exit()
    print(f"Probando {len(ANGLES_TO_TRY)} ángulos y {len(TRANSLATIONS_TO_TRY)} traslaciones ({total_transforms_to_try} combinaciones por landmark).")

    for i, image_name_ext in enumerate(image_files):
        image_name = Path(image_name_ext).stem
        image_path = images_dir / image_name_ext
        print(f"\n[{i+1}/{len(image_files)}] Procesando imagen: {image_name_ext}")

        # Cargar y redimensionar imagen de prueba
        test_image = load_image(image_path)
        if test_image is None:
            print(f"  Error cargando imagen. Saltando.")
            continue
        # Asegurar redimensionado correcto
        if test_image.size != (resized_image_size[1], resized_image_size[0]):
             resized_test_image = resize_image(test_image, resized_image_size)
        else:
             resized_test_image = test_image # Ya tiene el tamaño correcto

        output_image = resized_test_image.convert('RGB') # Para dibujar marcas

        # --- Búsqueda con Múltiples Transformaciones ---
        overall_min_error_coord1_mse, best_location_coord1_mse, best_params_coord1_mse = float('inf'), None, None
        overall_min_error_coord1_euc, best_location_coord1_euc, best_params_coord1_euc = float('inf'), None, None
        overall_min_error_coord2_mse, best_location_coord2_mse, best_params_coord2_mse = float('inf'), None, None
        overall_min_error_coord2_euc, best_location_coord2_euc, best_params_coord2_euc = float('inf'), None, None

        transform_count = 0
        # Iterar sobre todas las transformaciones hipotéticas
        for angle in ANGLES_TO_TRY:
            for trans_xy in TRANSLATIONS_TO_TRY:
                transform_count += 1
                # Descomentar para progreso detallado:
                # print(f"  Probando tf {transform_count}/{total_transforms_to_try} (Angle: {angle}°, Trans: {trans_xy})...")

                M_delta = create_delta_transform(angle, trans_xy, IMG_CENTER)
                if M_delta is None: continue

                M_delta_inverse = invert_affine_transform(M_delta)
                if M_delta_inverse is None: continue

                coord1_coords_hypothesis = transform_coordinates(coord1_coords_aligned, M_delta_inverse)
                coord2_coords_hypothesis = transform_coordinates(coord2_coords_aligned, M_delta_inverse)

                # --- Procesar Coord1 ---
                best_coord1_hypo_mse, error1_hypo_mse = process_region(resized_test_image, coord1_coords_hypothesis, coord1_centroid_local, coord1_crop_size, pca_model_coord1, metric='mse')
                if error1_hypo_mse < overall_min_error_coord1_mse:
                    overall_min_error_coord1_mse = error1_hypo_mse
                    best_location_coord1_mse = best_coord1_hypo_mse
                    best_params_coord1_mse = {'angle': angle, 'translation': trans_xy}

                best_coord1_hypo_euc, error1_hypo_euc = process_region(resized_test_image, coord1_coords_hypothesis, coord1_centroid_local, coord1_crop_size, pca_model_coord1, metric='euclidean')
                if error1_hypo_euc < overall_min_error_coord1_euc:
                    overall_min_error_coord1_euc = error1_hypo_euc
                    best_location_coord1_euc = best_coord1_hypo_euc
                    best_params_coord1_euc = {'angle': angle, 'translation': trans_xy}

                # --- Procesar Coord2 ---
                best_coord2_hypo_mse, error2_hypo_mse = process_region(resized_test_image, coord2_coords_hypothesis, coord2_centroid_local, coord2_crop_size, pca_model_coord2, metric='mse')
                if error2_hypo_mse < overall_min_error_coord2_mse:
                    overall_min_error_coord2_mse = error2_hypo_mse
                    best_location_coord2_mse = best_coord2_hypo_mse
                    best_params_coord2_mse = {'angle': angle, 'translation': trans_xy}

                best_coord2_hypo_euc, error2_hypo_euc = process_region(resized_test_image, coord2_coords_hypothesis, coord2_centroid_local, coord2_crop_size, pca_model_coord2, metric='euclidean')
                if error2_hypo_euc < overall_min_error_coord2_euc:
                    overall_min_error_coord2_euc = error2_hypo_euc
                    best_location_coord2_euc = best_coord2_hypo_euc
                    best_params_coord2_euc = {'angle': angle, 'translation': trans_xy}

        # --- Fin del bucle de transformaciones ---
        print(f"  Búsqueda multi-transformación completada.")

        # Almacenar los MEJORES resultados globales encontrados
        results[image_name_ext] = {
            'coord1_mse': {'coordinate': best_location_coord1_mse, 'error': overall_min_error_coord1_mse, 'params': best_params_coord1_mse},
            'coord1_euclidean': {'coordinate': best_location_coord1_euc, 'error': overall_min_error_coord1_euc, 'params': best_params_coord1_euc},
            'coord2_mse': {'coordinate': best_location_coord2_mse, 'error': overall_min_error_coord2_mse, 'params': best_params_coord2_mse},
            'coord2_euclidean': {'coordinate': best_location_coord2_euc, 'error': overall_min_error_coord2_euc, 'params': best_params_coord2_euc},
        }
        # Imprimir un resumen de los mejores resultados encontrados
        print(f"  Resultados Finales: C1_MSE={best_location_coord1_mse} (Err: {overall_min_error_coord1_mse:.4f}), C2_MSE={best_location_coord2_mse} (Err: {overall_min_error_coord2_mse:.4f})")

        # Marcar las MEJORES coordenadas (basadas en MSE por simplicidad)
        output_image = mark_coordinate(output_image, best_location_coord1_mse, color=(0, 255, 0), radius=2) # Verde C1
        output_image = mark_coordinate(output_image, best_location_coord2_mse, color=(0, 0, 255), radius=2) # Azul C2

        # Guardar la imagen con las marcas finales
        output_image_path = output_viz_dir / f"{image_name}_matches.png" # Usar nombre consistente
        try:
            output_image.save(output_image_path)
            # print(f"  Imagen con marcas finales guardada en: {output_image_path}")
        except Exception as e:
            print(f"  Error guardando imagen {output_image_path}: {e}")

    # --- Fin del Bucle Principal de Imágenes ---

    # Guardar todos los resultados finales en un archivo JSON
    results_path = output_json_dir / "prediction_results.json" # Usar nombre consistente
    print(f"\nGuardando resultados finales en: {results_path}")
    try:
        with open(results_path, 'w') as f:
            # Serializador robusto para tipos NumPy y float infinito/NaN
            def default_serializer(obj):
                 if isinstance(obj, np.ndarray): return obj.tolist()
                 if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
                 elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                     if np.isinf(obj): return "inf" # Representar como string
                     if np.isnan(obj): return "nan" # Representar como string
                     return float(obj)
                 elif isinstance(obj, (np.bool_)): return bool(obj)
                 elif isinstance(obj, (np.void)): return None
                 # Permitir que json maneje tipos básicos o falle para otros
                 return json.JSONEncoder().default(obj)

            json.dump(results, f, indent=4, default=default_serializer)
        print("Resultados guardados correctamente.")
    except TypeError as e:
        print(f"Error de tipo al serializar JSON: {e}. Revisa los datos en 'results'.")
    except Exception as e:
        print(f"Error general guardando resultados JSON: {e}")

    print("\n--- Predicción de Coordenadas (Múltiples Transformaciones) Completada ---")