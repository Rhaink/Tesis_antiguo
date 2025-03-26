"""
Utilidades para el procesamiento de imágenes.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

def clean_directory(directory: str) -> None:
    """
    Limpia un directorio eliminando todos los archivos.
    
    Args:
        directory: Ruta del directorio a limpiar
    """
    dir_path = Path(directory)
    if dir_path.exists():
        for file in dir_path.glob("*"):
            if file.is_file():
                file.unlink()

def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Asegura que una imagen esté en escala de grises.
    
    Args:
        image: Imagen a procesar
        
    Returns:
        Imagen en escala de grises
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normaliza una imagen al rango [0,1].
    
    Args:
        image: Imagen a normalizar
        
    Returns:
        Imagen normalizada
    """
    return image.astype(float) / 255.0

def resize_image(image: np.ndarray, 
                target_size: Optional[Tuple[int, int]] = None,
                scale_factor: Optional[float] = None) -> np.ndarray:
    """
    Redimensiona una imagen.
    
    Args:
        image: Imagen a redimensionar
        target_size: Tamaño objetivo (ancho, alto)
        scale_factor: Factor de escala
        
    Returns:
        Imagen redimensionada
        
    Raises:
        ValueError: Si no se especifica ni target_size ni scale_factor
    """
    if target_size is None and scale_factor is None:
        raise ValueError("Debe especificar target_size o scale_factor")
        
    if target_size:
        return cv2.resize(image, target_size)
        
    if scale_factor:
        new_size = (
            int(image.shape[1] * scale_factor),
            int(image.shape[0] * scale_factor)
        )
        return cv2.resize(image, new_size)

def extract_region(image: np.ndarray,
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

def apply_clahe(image: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Aplica ecualización de histograma adaptativa con contraste limitado (CLAHE).
    
    Args:
        image: Imagen a procesar
        clip_limit: Límite de contraste
        tile_grid_size: Tamaño de la cuadrícula
        
    Returns:
        Imagen procesada
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(ensure_grayscale(image))

def enhance_contrast(image: np.ndarray,
                    alpha: float = 1.5,
                    beta: float = 0) -> np.ndarray:
    """
    Mejora el contraste de una imagen.
    
    Args:
        image: Imagen a procesar
        alpha: Factor de ganancia
        beta: Sesgo
        
    Returns:
        Imagen procesada
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def denoise_image(image: np.ndarray,
                  h: float = 10.0,
                  template_window_size: int = 7,
                  search_window_size: int = 21) -> np.ndarray:
    """
    Aplica reducción de ruido a una imagen.
    
    Args:
        image: Imagen a procesar
        h: Parámetro de filtrado
        template_window_size: Tamaño de ventana de template
        search_window_size: Tamaño de ventana de búsqueda
        
    Returns:
        Imagen procesada
    """
    return cv2.fastNlMeansDenoising(
        ensure_grayscale(image),
        None,
        h,
        template_window_size,
        search_window_size
    )
