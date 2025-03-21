"""
Módulo para el procesamiento de imágenes pulmonares.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import os
from .contrast_enhancer import ContrastEnhancer
from .template_processor import TemplateProcessor

class ImageProcessor:
    """
    Clase para el procesamiento de imágenes pulmonares.
    
    Esta clase maneja:
    - Carga y redimensionamiento de imágenes
    - Mejoramiento de contraste mediante SAHS
    - Extracción de regiones usando templates
    - Guardado de imágenes procesadas
    """
    
    def __init__(self, base_path: str, template_data_path: str, output_dir: Optional[str] = None):
        """
        Inicializa el procesador de imágenes.
        
        Args:
            base_path: Ruta base donde se encuentran las imágenes
            template_data_path: Ruta al archivo JSON con datos de templates
            output_dir: Directorio opcional para guardar resultados
        """
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir) if output_dir else None
        self.contrast_enhancer = ContrastEnhancer()
        self.template_processor = TemplateProcessor(template_data_path)
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_and_resize_image(self, image_path: str, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
        """
        Carga y redimensiona una imagen.
        
        Args:
            image_path: Ruta de la imagen
            target_size: Tamaño objetivo (ancho, alto)
            
        Returns:
            Imagen procesada como array numpy
            
        Raises:
            FileNotFoundError: Si no se encuentra la imagen
            ValueError: Si la imagen no se puede cargar
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
            
        try:
            # Leer imagen
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
                
            # Aplicar mejora de contraste SAHS
            #enhanced_image = self.contrast_enhancer.enhance_contrast_sahs(image)
            #if enhanced_image is None:
            #    enhanced_image = image
            # Convertir a escala de grises si es necesario
            
            if len(image.shape) > 2:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image.copy()
        
            
            # Redimensionar imagen
            return cv2.resize(gray_image, size)
            
        except Exception as e:
            raise ValueError(f"Error procesando imagen {image_path}: {str(e)}")

    def load_training_images(self, coord_name: str) -> List[np.ndarray]:
        """
        Carga las imágenes de entrenamiento para un punto específico.
        
        Args:
            coord_name: Nombre del punto.
            
        Returns:
            Lista de imágenes procesadas
        """
        # Solo procesar coord1 y coord2
        if coord_name.lower() not in ['coord1', 'coord2']:
            raise ValueError(f"Solo se permiten coord1 y coord2. Recibido: {coord_name}")
            
        # Obtener dimensiones del template
        template_data = self.template_processor.load_template_data(coord_name)
        if template_data is None:
            raise ValueError(f"No se encontraron datos del template para {coord_name}")
            
        template_bounds = template_data["template_bounds"]
        template_size = (int(template_bounds["width"]), int(template_bounds["height"]))
            
        training_dir = Path("/home/donrobot/projects/Tesis/resultados/recorte/prueba_2/processed_images") / f"cropped_images_Coord{coord_name[-1]}"
        if not training_dir.exists():
            raise FileNotFoundError(f"No se encontró el directorio de entrenamiento: {training_dir}")
            
        images = []
        for img_path in training_dir.glob("*.png"):
            try:
                # Cargar imagen 
                image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError(f"No se pudo cargar la imagen: {img_path}")
                    
                # Verificar dimensiones
                if image.shape[:2] != (template_bounds["height"], template_bounds["width"]):
                    raise ValueError(
                        f"Dimensiones incorrectas en {img_path}: "
                        f"esperado {(template_bounds['height'], template_bounds['width'])}, "
                        f"obtenido {image.shape[:2]}"
                    )
                    
                images.append(image)
            except Exception as e:
                print(f"Error cargando {img_path}: {str(e)}")
                continue
                
        if not images:
            raise ValueError(f"No se encontraron imágenes válidas en {training_dir}")
            
        return images

    def save_processed_image(self, image: np.ndarray, filename: str) -> str:
        """
        Guarda una imagen procesada.
        
        Args:
            image: Imagen a guardar
            filename: Nombre del archivo
            
        Returns:
            Ruta donde se guardó la imagen
            
        Raises:
            ValueError: Si no se especificó directorio de salida
        """
        if not self.output_dir:
            raise ValueError("No se especificó directorio de salida")
            
        output_path = self.output_dir / filename
        
        try:
            cv2.imwrite(str(output_path), image)
            return str(output_path)
        except Exception as e:
            raise ValueError(f"Error guardando imagen {filename}: {str(e)}")

    def extract_region(self, 
                      image: np.ndarray,
                      x: int,
                      y: int,
                      width: int,
                      height: int) -> np.ndarray:
        """
        Extrae una región de una imagen.
        
        Args:
            image: Imagen fuente
            x: Coordenada x inicial
            y: Coordenada y inicial
            width: Ancho de la región
            height: Alto de la región
            
        Returns:
            Región extraída
            
        Raises:
            ValueError: Si las coordenadas son inválidas
        """
        if x < 0 or y < 0 or x + width > image.shape[1] or y + height > image.shape[0]:
            raise ValueError("Coordenadas fuera de los límites de la imagen")
            
        return image[y:y+height, x:x+width]

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normaliza una imagen al rango [0,1].
        
        Args:
            image: Imagen a normalizar
            
        Returns:
            Imagen normalizada
        """
        return image.astype(float) / 255.0

    def prepare_image_for_pca(self, image: np.ndarray) -> np.ndarray:
        """
        Prepara una imagen para análisis PCA.
        
        Args:
            image: Imagen a preparar
            
        Returns:
            Imagen preparada
        """
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Normalizar
        return self.normalize_image(image)
