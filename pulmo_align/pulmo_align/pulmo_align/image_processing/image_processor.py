"""
Módulo para el procesamiento de imágenes pulmonares.

Este módulo proporciona clases y funciones para:
- Carga y redimensionamiento de imágenes
- Extracción de regiones de interés
- Procesamiento de imágenes recortadas
- Gestión de rutas de imágenes
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from .contrast_enhancer import ContrastEnhancer

class ImageProcessor:
    """
    Clase para el procesamiento de imágenes pulmonares.
    
    Esta clase maneja:
    - Carga y redimensionamiento de imágenes
    - Extracción de regiones de interés
    - Guardado de imágenes procesadas
    - Gestión de rutas de archivos
    - Mejoramiento de contraste mediante SAHS
    
    Attributes:
        base_path (Path): Ruta base para las imágenes del dataset
        output_base_path (Path): Ruta base para guardar las imágenes procesadas
        contrast_enhancer (ContrastEnhancer): Instancia para mejoramiento de contraste
    """
    
    def __init__(self, base_path: str = "COVID-19_Radiography_Dataset"):
        """
        Inicializa el procesador de imágenes.
        
        Args:
            base_path (str): Ruta base al dataset de imágenes
        """
        self.base_path = Path(base_path)
        self.output_base_path = Path("processed_images")
        self.contrast_enhancer = ContrastEnhancer()

    def get_image_path(self, index: int, indices_file: str) -> str:
        """
        Obtiene la ruta de una imagen basada en su índice.
        
        Args:
            index (int): Índice de la imagen
            indices_file (str): Ruta al archivo de índices
            
        Returns:
            str: Ruta completa a la imagen
            
        Raises:
            FileNotFoundError: Si no se encuentra el archivo de índices
            ValueError: Si el índice no existe en el archivo
        """
        try:
            data_indices = pd.read_csv(indices_file, header=None)
            row = data_indices[data_indices[0] == index].iloc[0]
            category = row[1]
            image_number = row[2]

            # Construir la ruta base según la categoría
            if category == 1:
                category_path = self.base_path / "COVID/images"
                image_name = f"COVID-{image_number}.png"
            elif category == 2:
                category_path = self.base_path / "Normal/images"
                image_name = f"Normal-{image_number}.png"
            elif category == 3:
                category_path = self.base_path / "Viral Pneumonia/images"
                image_name = f"Viral Pneumonia-{image_number}.png"
            else:
                raise ValueError(f"Categoría no válida: {category}")

            # Verificar si el archivo existe
            image_path = category_path / image_name
            if not image_path.exists():
                print(f"Advertencia: No se encontró la imagen {image_path}")
                # Intentar con formato de número diferente
                if len(str(image_number)) <= 4:
                    # Probar con formato de 4 dígitos
                    image_name_alt = image_name.replace(str(image_number), f"{image_number:04d}")
                    image_path = category_path / image_name_alt
                    if not image_path.exists():
                        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

            return str(image_path)

        except Exception as e:
            print(f"Error al obtener la ruta de la imagen {index}: {str(e)}")
            raise

    def load_and_resize_image(self, image_path: str, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
        """
        Carga, mejora el contraste y redimensiona una imagen.
        
        Args:
            image_path (str): Ruta a la imagen
            size (Tuple[int, int]): Dimensiones objetivo (ancho, alto)
            
        Returns:
            np.ndarray: Imagen procesada y redimensionada
            
        Raises:
            FileNotFoundError: Si no se encuentra la imagen
            ValueError: Si hay un error al procesar la imagen
        """
        try:
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
            
            # Aplicar mejora de contraste SAHS
            enhanced_image = self.contrast_enhancer.enhance_contrast_sahs(image)
            if enhanced_image is None:
                print(f"Advertencia: No se pudo mejorar el contraste de {image_path}")
                enhanced_image = image
            
            # Redimensionar imagen
            return cv2.resize(enhanced_image, size)
            
        except Exception as e:
            print(f"Error al cargar/procesar la imagen {image_path}: {str(e)}")
            raise

    def extract_region(self, 
                      image: np.ndarray,
                      center_x: int,
                      center_y: int,
                      width: int,
                      height: int,
                      intersection_x: int,
                      intersection_y: int,
                      new_x: int,
                      new_y: int) -> np.ndarray:
        """
        Extrae una región de interés de la imagen.
        
        Args:
            image (np.ndarray): Imagen fuente
            center_x (int): Coordenada x del centro
            center_y (int): Coordenada y del centro
            width (int): Ancho de la región
            height (int): Alto de la región
            intersection_x (int): Coordenada x del punto de intersección
            intersection_y (int): Coordenada y del punto de intersección
            new_x (int): Nueva coordenada x
            new_y (int): Nueva coordenada y
            
        Returns:
            np.ndarray: Región extraída
            
        Raises:
            ValueError: Si las dimensiones o coordenadas son inválidas
        """
        try:
            # Calculamos el desplazamiento
            dx = new_x - intersection_x
            dy = new_y - intersection_y

            # Ajustamos el centro
            new_center_x = center_x + dx
            new_center_y = center_y + dy

            # Calculamos las coordenadas para el recorte
            crop_start_x = max(new_center_x - (width // 2), 0)
            crop_end_x = min(new_center_x + (width // 2) + (width % 2), image.shape[1])
            crop_start_y = max(new_center_y - (height // 2), 0)
            crop_end_y = min(new_center_y + (height // 2) + (height % 2), image.shape[0])

            # Verificamos que las dimensiones sean válidas
            if crop_start_x >= crop_end_x or crop_start_y >= crop_end_y:
                raise ValueError(f"Dimensiones de recorte inválidas: x[{crop_start_x}:{crop_end_x}], y[{crop_start_y}:{crop_end_y}]")

            # Extraemos y redimensionamos la región
            cropped = image[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
            return cv2.resize(cropped, (width, height))

        except Exception as e:
            print(f"Error al extraer la región: {str(e)}")
            print(f"Parámetros: center({center_x},{center_y}), size({width},{height}), intersection({intersection_x},{intersection_y}), new({new_x},{new_y})")
            raise

    def save_cropped_image(self,
                          cropped_image: np.ndarray,
                          coord_name: str,
                          index: int) -> bool:
        """
        Guarda una imagen recortada.
        
        Args:
            cropped_image (np.ndarray): Imagen recortada
            coord_name (str): Nombre de la coordenada
            index (int): Índice de la imagen
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario
        """
        try:
            # Creamos el directorio si no existe
            output_dir = self.output_base_path / f'cropped_images_{coord_name}'
            output_dir.mkdir(parents=True, exist_ok=True)

            # Guardamos la imagen
            output_path = output_dir / f'cropped_image_index_{index}.png'
            success = cv2.imwrite(str(output_path), cropped_image)
            
            if not success:
                print(f"Error: No se pudo guardar la imagen en {output_path}")
            
            return success

        except Exception as e:
            print(f"Error al guardar la imagen {coord_name}_{index}: {str(e)}")
            return False

    def load_training_images(self, 
                           coord_name: str,
                           target_size: Tuple[int, int]) -> List[np.ndarray]:
        """
        Carga las imágenes de entrenamiento para una coordenada específica.
        Las imágenes ya tienen el SAHS aplicado desde el proceso de recorte.
        
        Args:
            coord_name (str): Nombre de la coordenada
            target_size (Tuple[int, int]): Dimensiones objetivo (ancho, alto)
            
        Returns:
            List[np.ndarray]: Lista de imágenes cargadas y procesadas
        """
        cropped_images = []
        try:
            for index in range(0, 99):
                cropped_path = self.output_base_path / f'cropped_images_{coord_name}' / f'cropped_image_index_{index}.png'
                if cropped_path.exists():
                    # Cargar imagen (ya tiene SAHS aplicado desde el recorte)
                    cropped = cv2.imread(str(cropped_path), cv2.IMREAD_GRAYSCALE)
                    if cropped is not None:
                        # Solo redimensionar si es necesario
                        cropped_resized = cv2.resize(cropped, target_size)
                        cropped_images.append(cropped_resized)
                else:
                    print(f"Advertencia: No se encontró {cropped_path}")
        except Exception as e:
            print(f"Error cargando imágenes para {coord_name}: {str(e)}")
        
        return cropped_images
