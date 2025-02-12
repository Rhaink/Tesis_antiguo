"""
Módulo para el procesamiento de imágenes pulmonares.

Este módulo proporciona clases y funciones para:
- Carga y redimensionamiento de imágenes
- Extracción de regiones de interés usando templates
- Procesamiento de imágenes recortadas
- Gestión de rutas de imágenes
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from .contrast_enhancer import ContrastEnhancer
from .template_processor import TemplateProcessor

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
    
    def __init__(self, base_path: str, visualization_dir: str = "visualization_results"):
        """
        Inicializa el procesador de imágenes.
        
        Args:
            base_path (str): Ruta base al dataset de imágenes
            visualization_dir (str): Directorio para guardar visualizaciones
        """
        self.base_path = Path(base_path)
        self.output_base_path = Path(base_path).parent / "processed_images"
        self.contrast_enhancer = ContrastEnhancer()
        self.template_processor = TemplateProcessor(visualization_dir)

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
                enhanced_image = image
            
            # Redimensionar imagen
            return cv2.resize(enhanced_image, size)
            
        except Exception as e:
            print(f"Error al cargar/procesar la imagen {image_path}: {str(e)}")
            raise

    def validate_coord_number(self, coord_num: int) -> None:
        """
        Valida que el número de coordenada sea 1 o 2.
        
        Args:
            coord_num (int): Número de coordenada a validar
            
        Raises:
            ValueError: Si el número de coordenada no es 1 o 2
        """
        if coord_num not in [1, 2]:
            raise ValueError(f"Solo se permiten los puntos 1 y 2. Recibido: {coord_num}")

    def extract_region(self, 
                      image: np.ndarray,
                      search_region: np.ndarray,
                      labeled_point: Tuple[int, int],
                      coord_num: int,
                      template_size: int = None) -> np.ndarray:
        """
        Extrae una región de interés de la imagen usando un template.
        
        Args:
            image (np.ndarray): Imagen fuente
            search_region (np.ndarray): Región de búsqueda binaria
            labeled_point (Tuple[int, int]): Punto etiquetado (x,y)
            coord_num (int): Número de coordenada
            template_size (int): Tamaño del template
            
        Returns:
            np.ndarray: Región extraída
            
        Raises:
            ValueError: Si las dimensiones o coordenadas son inválidas
        """
        try:
            # Validar número de coordenada
            self.validate_coord_number(coord_num)
            
            # Asegurar que la imagen tenga el tamaño correcto
            if image.shape[:2] != (64, 64):
                image = cv2.resize(image, (64, 64))
                
            # Asegurar que la región de búsqueda sea 64x64
            if search_region.shape != (64, 64):
                search_region = cv2.resize(search_region.astype(np.float32), (64, 64)) > 0.5
                
            # Preparar nombre de coordenada
            coord_name = f"coord{coord_num}"
            
            # Cargar datos pre-calculados del template
            template_data = self.template_processor.load_template_data(coord_name)
            if template_data is None:
                raise ValueError(f"No se encontraron datos pre-calculados para {coord_name}")
            
            # Extraer datos
            template_bounds = template_data["template_bounds"]
            width = template_bounds["width"]
            height = template_bounds["height"]
            min_x = template_bounds["min_x"]
            min_y = template_bounds["min_y"]
            intersection_point = (
                template_data["intersection_point"]["x"],
                template_data["intersection_point"]["y"]
            )
            
            # Crear template de recorte
            cutting_template = np.zeros((64, 64), dtype=np.uint8)
            cutting_template[0: height, 0: width] = 1
            
            # Obtener región recortada
            cropped = self.template_processor.crop_aligned_image(
                image, cutting_template, labeled_point, intersection_point
            )
            
            return cropped

        except Exception as e:
            print(f"Error al extraer la región para Coord{coord_num}: {str(e)}")
            print(f"Punto etiquetado: {labeled_point}")
            raise

    def validate_coord_name(self, coord_name: str) -> None:
        """
        Valida que el nombre de coordenada sea Coord1 o Coord2.
        
        Args:
            coord_name (str): Nombre de coordenada a validar
            
        Raises:
            ValueError: Si el nombre de coordenada no es válido
        """
        if coord_name not in ["Coord1", "Coord2"]:
            raise ValueError(f"Solo se permiten Coord1 y Coord2. Recibido: {coord_name}")

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
            # Validar nombre de coordenada
            self.validate_coord_name(coord_name)
            
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
