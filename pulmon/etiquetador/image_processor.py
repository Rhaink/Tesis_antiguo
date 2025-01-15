"""
Procesamiento de imágenes para el programa de etiquetado.
"""
import cv2
import numpy as np
from .config import VENTANA_ANCHO, VENTANA_ALTO

class ImageProcessor:
    """Clase para el procesamiento de imágenes."""
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Procesa una imagen para su visualización.
        
        Args:
            image: Imagen original.
            
        Returns:
            Imagen procesada.
        """
        # Redimensionar manteniendo proporción
        imagen_visualizacion = self.resize_aspect_ratio(image, width=VENTANA_ANCHO)
        
        # Aplicar CLAHE para mejorar contraste
        return self.apply_clahe(imagen_visualizacion)
    
    def resize_aspect_ratio(self, image: np.ndarray, width: int = None, height: int = None, 
                          inter: int = cv2.INTER_AREA) -> np.ndarray:
        """
        Redimensiona una imagen manteniendo su proporción.
        
        Args:
            image: Imagen a redimensionar.
            width: Ancho deseado.
            height: Alto deseado.
            inter: Método de interpolación.
            
        Returns:
            Imagen redimensionada.
        """
        dim = None
        (h, w) = image.shape[:2]
        
        if width is None and height is None:
            return image
            
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
            
        return cv2.resize(image, dim, interpolation=inter)
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica CLAHE para mejorar el contraste.
        
        Args:
            image: Imagen original.
            
        Returns:
            Imagen con contraste mejorado.
        """
        # Convertir a LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Aplicar CLAHE al canal L
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Combinar canales y convertir de vuelta a BGR
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    def calculate_line_thickness(self, image: np.ndarray) -> int:
        """
        Calcula el grosor de línea apropiado basado en el tamaño de la imagen.
        
        Args:
            image: Imagen procesada.
            
        Returns:
            Grosor de línea calculado.
        """
        altura, ancho = image.shape[:2]
        return max(1, min(altura, ancho) // 200)
