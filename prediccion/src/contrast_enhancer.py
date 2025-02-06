"""
Módulo para el mejoramiento de contraste en imágenes radiográficas.
Versión simplificada del proyecto original pulmo_align.
"""

import numpy as np
import cv2
from typing import Union

class ContrastEnhancer:
    """
    Clase que implementa algoritmos de mejoramiento de contraste.
    
    Esta clase proporciona métodos para:
    - Mejoramiento de contraste asimétrico (SAHS)
    - Normalización de imágenes radiográficas
    """
    
    @staticmethod
    def enhance_contrast_sahs(image: Union[np.ndarray, None]) -> Union[np.ndarray, None]:
        """
        Aplica el algoritmo SAHS para mejorar el contraste de la imagen.
        
        El algoritmo realiza un análisis estadístico asimétrico del histograma
        para determinar los límites de estiramiento basados en la media y
        desviación estándar de los grupos de píxeles por encima y debajo de la media.
        
        Args:
            image: Imagen de entrada en escala de grises
            
        Returns:
            Imagen con contraste mejorado
            
        Raises:
            ValueError: Si la imagen de entrada es None o tiene formato inválido
        """
        try:
            if image is None:
                raise ValueError("La imagen de entrada es None")
                
            # Convertir a escala de grises si es necesario
            if len(image.shape) > 2:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image.copy()
            
            # Calcular la media de los niveles de gris
            gray_mean = np.mean(gray_image)
            
            # Separar píxeles por encima y debajo de la media
            above_mean = gray_image[gray_image > gray_mean]
            below_or_equal_mean = gray_image[gray_image <= gray_mean]
            
            # Calcular límites usando desviación estándar asimétrica
            max_value = gray_mean
            min_value = gray_mean
            
            if above_mean.size > 0:
                # Factor 2.5 para el límite superior
                std_above = np.sqrt(np.mean((above_mean - gray_mean) ** 2))
                max_value = gray_mean + 2.5 * std_above
                
            if below_or_equal_mean.size > 0:
                # Factor 2.0 para el límite inferior
                std_below = np.sqrt(np.mean((below_or_equal_mean - gray_mean) ** 2))
                min_value = gray_mean - 2.0 * std_below
            
            # Normalizar al rango [0, 255]
            if max_value != min_value:
                enhanced_image = np.clip(
                    (255 / (max_value - min_value)) * (gray_image - min_value),
                    0, 255
                ).astype(np.uint8)
            else:
                enhanced_image = gray_image
                
            return enhanced_image
            
        except Exception as e:
            print(f"Error en enhance_contrast_sahs: {str(e)}")
            return None
