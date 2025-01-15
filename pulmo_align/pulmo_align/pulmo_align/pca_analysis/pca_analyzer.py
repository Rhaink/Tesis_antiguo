"""
Módulo para el análisis de componentes principales (PCA) en imágenes pulmonares.

Este módulo proporciona clases y funciones para:
- Aplicación de PCA a imágenes
- Cálculo de errores de reconstrucción
- Análisis de regiones de búsqueda
- Reconstrucción de imágenes
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Union
import cv2

class PCAAnalyzer:
    """
    Clase para realizar análisis PCA en imágenes pulmonares.
    
    Esta clase maneja:
    - Entrenamiento de modelos PCA
    - Cálculo de errores de reconstrucción
    - Análisis de regiones de búsqueda
    - Reconstrucción de imágenes
    
    Attributes:
        pca (PCA): Modelo PCA entrenado
        mean_face (np.ndarray): Imagen promedio del conjunto de entrenamiento
        eigenfaces (np.ndarray): Eigenfaces calculados
        n_components (int): Número de componentes principales
    """
    
    def __init__(self):
        """Inicializa el analizador PCA."""
        self.pca = None
        self.mean_face = None
        self.eigenfaces = None
        self.n_components = 0

    def train(self, images: List[np.ndarray], variance_threshold: float = 0.95) -> None:
        """
        Entrena el modelo PCA con un conjunto de imágenes.
        
        Args:
            images (List[np.ndarray]): Lista de imágenes de entrenamiento
            variance_threshold (float): Umbral de varianza explicada para seleccionar componentes
            
        Raises:
            ValueError: Si no hay imágenes de entrenamiento o son inválidas
        """
        if not images:
            raise ValueError("No hay imágenes de entrenamiento disponibles")
        
        # Preparar los datos
        X = np.array([img.astype(float) / 255.0 for img in images])
        X = X.reshape(len(images), -1)
        
        # Calcular número óptimo de componentes
        temp_pca = PCA()
        temp_pca.fit(X)
        cumulative_variance_ratio = np.cumsum(temp_pca.explained_variance_ratio_)
        self.n_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
        
        # Entrenar PCA final
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)
        
        # Guardar mean_face y eigenfaces
        self.mean_face = self.pca.mean_.reshape(images[0].shape)
        self.eigenfaces = self.pca.components_.reshape((self.n_components, *images[0].shape))

    def calculate_omega(self, test_image: np.ndarray) -> np.ndarray:
        """
        Calcula los coeficientes omega para una imagen de prueba.
        
        Args:
            test_image (np.ndarray): Imagen de prueba
            
        Returns:
            np.ndarray: Coeficientes omega
            
        Raises:
            ValueError: Si el modelo PCA no está entrenado
        """
        if self.pca is None:
            raise ValueError("El modelo PCA no está entrenado")
            
        test_image_vector = test_image.flatten().reshape(1, -1)
        return self.pca.transform(test_image_vector)

    def reconstruct_image(self, omega: np.ndarray) -> np.ndarray:
        """
        Reconstruye una imagen a partir de sus coeficientes omega.
        
        Args:
            omega (np.ndarray): Coeficientes omega
            
        Returns:
            np.ndarray: Imagen reconstruida
            
        Raises:
            ValueError: Si el modelo PCA no está entrenado
        """
        if self.pca is None:
            raise ValueError("El modelo PCA no está entrenado")
            
        return self.pca.inverse_transform(omega)

    def calculate_error(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calcula el error entre una imagen original y su reconstrucción.
        
        Args:
            original (np.ndarray): Imagen original
            reconstructed (np.ndarray): Imagen reconstruida
            
        Returns:
            float: Error de reconstrucción (norma L2)
        """
        return np.linalg.norm(original.flatten() - reconstructed.flatten())

    def analyze_search_region(self, 
                            image: np.ndarray,
                            search_coordinates: List[Tuple[int, int]],
                            template_width: int,
                            template_height: int,
                            intersection_x: int,
                            intersection_y: int) -> Tuple[float, Tuple[int, int], List[float]]:
        """
        Analiza una región de búsqueda para encontrar la mejor coincidencia.
        
        Args:
            image (np.ndarray): Imagen a analizar
            search_coordinates (List[Tuple[int, int]]): Lista de coordenadas de búsqueda
            template_width (int): Ancho del template
            template_height (int): Alto del template
            intersection_x (int): Coordenada x del punto de intersección
            intersection_y (int): Coordenada y del punto de intersección
            
        Returns:
            Tuple[float, Tuple[int, int], List[float]]: 
                - Error mínimo encontrado
                - Coordenadas del error mínimo
                - Lista de errores para todas las coordenadas
                
        Raises:
            ValueError: Si el modelo PCA no está entrenado o los parámetros son inválidos
        """
        if self.pca is None:
            raise ValueError("El modelo PCA no está entrenado")

        min_error = float('inf')
        min_error_coords = None
        errors = []

        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        for coord in search_coordinates:
            template_y, template_x = coord
            start_x = template_x - intersection_x
            start_y = template_y - intersection_y

            # Verificar límites
            if (start_x < 0 or start_y < 0 or 
                start_x + template_width > image_gray.shape[1] or 
                start_y + template_height > image_gray.shape[0]):
                errors.append(float('inf'))
                continue

            try:
                # Extraer y procesar región
                cropped_region = image_gray[
                    start_y:start_y+template_height,
                    start_x:start_x+template_width
                ]
                
                if cropped_region.size == 0:
                    errors.append(float('inf'))
                    continue

                # Redimensionar si es necesario
                if cropped_region.shape != (template_height, template_width):
                    cropped_region = cv2.resize(cropped_region, 
                                              (template_width, template_height))
                
                # Normalizar
                cropped_region = cropped_region.astype(float) / 255.0

                # Calcular error
                omega = self.calculate_omega(cropped_region)
                reconstructed = self.reconstruct_image(omega)
                error = self.calculate_error(cropped_region, reconstructed)
                
                errors.append(error)

                if error < min_error:
                    min_error = error
                    min_error_coords = (template_x, template_y)

            except Exception as e:
                print(f"Error en coordenada ({template_x}, {template_y}): {str(e)}")
                errors.append(float('inf'))

        if min_error_coords is None:
            raise ValueError("No se encontró ninguna coordenada válida")

        return min_error, min_error_coords, errors

    def get_model_info(self) -> Dict:
        """
        Obtiene información sobre el modelo PCA entrenado.
        
        Returns:
            Dict: Diccionario con información del modelo
        """
        if self.pca is None:
            return {"trained": False}
            
        return {
            "trained": True,
            "n_components": self.n_components,
            "explained_variance_ratio": self.pca.explained_variance_ratio_.sum(),
            "mean_face_shape": self.mean_face.shape if self.mean_face is not None else None,
            "n_eigenfaces": len(self.eigenfaces) if self.eigenfaces is not None else 0
        }
