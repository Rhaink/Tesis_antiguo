"""
Módulo para el análisis de componentes principales (PCA) en imágenes pulmonares.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .template_processor import TemplateProcessor
import cv2
import pickle
from pathlib import Path

class PCAAnalyzer:
    """
    Clase para realizar análisis PCA en imágenes pulmonares.
    
    Esta clase maneja:
    - Entrenamiento de modelos PCA
    - Cálculo de errores de reconstrucción
    - Análisis de regiones de búsqueda
    - Reconstrucción de imágenes
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa el analizador PCA.
        
        Args:
            model_path: Ruta opcional para cargar un modelo pre-entrenado
        """
        self.pca = None
        self.mean_face = None
        self.eigenfaces = None
        self.n_components = 0
        
        if model_path:
            self.load_model(model_path)

    def train(self, images: List[np.ndarray], variance_threshold: float = 0.95) -> None:
        """
        Entrena el modelo PCA con un conjunto de imágenes.
        
        Args:
            images: Lista de imágenes de entrenamiento
            variance_threshold: Umbral de varianza explicada para seleccionar componentes
            
        Raises:
            ValueError: Si no hay imágenes de entrenamiento o son inválidas
        """
        if not images:
            raise ValueError("No hay imágenes de entrenamiento disponibles")
        
        # Preparar los datos
        X = np.array([img.astype(float) for img in images])
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

    def save_model(self, model_path: str) -> None:
        """
        Guarda el modelo PCA entrenado.
        
        Args:
            model_path: Ruta donde guardar el modelo
            
        Raises:
            ValueError: Si el modelo no está entrenado
        """
        if self.pca is None:
            raise ValueError("El modelo PCA no está entrenado")
            
        model_data = {
            'pca': self.pca,
            'mean_face': self.mean_face,
            'eigenfaces': self.eigenfaces,
            'n_components': self.n_components
        }
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, model_path: str) -> None:
        """
        Carga un modelo PCA pre-entrenado.
        
        Args:
            model_path: Ruta del modelo a cargar
            
        Raises:
            FileNotFoundError: Si no se encuentra el archivo del modelo
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
            
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.pca = model_data['pca']
        self.mean_face = model_data['mean_face']
        self.eigenfaces = model_data['eigenfaces']
        self.n_components = model_data['n_components']

    def analyze_search_region(self, 
                            image: np.ndarray,
                            search_coordinates: List[Tuple[int, int]],
                            template_processor: 'TemplateProcessor',
                            template_data: Dict) -> Tuple[float, Tuple[int, int], List[float]]:
        """
        Analiza una región de búsqueda para encontrar la mejor coincidencia.
        
        El proceso:
        1. Para cada punto en la región de búsqueda:
           - Mueve el template completo usando el punto de intersección como ancla
           - Extrae la región usando las dimensiones originales del template
           - Calcula el error de reconstrucción PCA
        2. Encuentra el punto donde el error de reconstrucción es mínimo
        
        Args:
            image: Imagen a analizar
            search_coordinates: Lista de coordenadas de búsqueda (y, x)
            template_processor: Instancia de TemplateProcessor para extraer regiones
            template_data: Datos del template incluyendo dimensiones y punto de intersección
            
        Returns:
            Tuple con:
            - Error mínimo encontrado
            - Coordenadas (y, x) del error mínimo
            - Lista de errores para todas las coordenadas
            
        Raises:
            ValueError: Si el modelo no está entrenado o no se encuentran coordenadas válidas
        """
        if self.pca is None:
            raise ValueError("El modelo PCA no está entrenado")

        min_error = float('inf')
        min_error_coords = None
        errors = []

        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Obtener dimensiones del template para validación
        template_bounds = template_data["template_bounds"]
        template_width = int(template_bounds["width"])
        template_height = int(template_bounds["height"])

        for coord in search_coordinates:
            try:
                # Extraer región usando el template
                cropped_region = template_processor.extract_region(
                    image=image,
                    template_data=template_data,
                    search_point=coord
                )
                
                # Verificar dimensiones del recorte
                if cropped_region.shape != (template_height, template_width):
                    raise ValueError(f"Dimensiones incorrectas del recorte: {cropped_region.shape}")
                
                # Normalizar
                cropped_region = cropped_region.astype(float)

                # Calcular error de reconstrucción
                omega = self.calculate_omega(cropped_region)
                reconstructed = self.reconstruct_image(omega)
                error = self.calculate_error(cropped_region, reconstructed)
                
                errors.append(error)

                # Actualizar mínimo si corresponde
                if error < min_error:
                    min_error = error
                    min_error_coords = coord

            except ValueError as ve:
                # Error esperado (fuera de límites o dimensiones incorrectas)
                errors.append(float('inf'))
            except Exception as e:
                # Error inesperado
                print(f"Error inesperado en coordenada {coord}: {str(e)}")
                errors.append(float('inf'))

        if min_error_coords is None:
            raise ValueError("No se encontró ninguna coordenada válida en la región de búsqueda")

        return min_error, min_error_coords, errors

    def calculate_omega(self, test_image: np.ndarray) -> np.ndarray:
        """
        Calcula los coeficientes omega para una imagen de prueba.
        
        Args:
            test_image: Imagen de prueba
            
        Returns:
            Coeficientes omega
        """
        if self.pca is None:
            raise ValueError("El modelo PCA no está entrenado")
            
        test_image_vector = test_image.flatten().reshape(1, -1)
        return self.pca.transform(test_image_vector)

    def reconstruct_image(self, omega: np.ndarray) -> np.ndarray:
        """
        Reconstruye una imagen a partir de sus coeficientes omega.
        
        Args:
            omega: Coeficientes omega
            
        Returns:
            Imagen reconstruida
        """
        if self.pca is None:
            raise ValueError("El modelo PCA no está entrenado")
            
        return self.pca.inverse_transform(omega)

    def calculate_error(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calcula el error entre una imagen original y su reconstrucción.
        
        Args:
            original: Imagen original
            reconstructed: Imagen reconstruida
            
        Returns:
            Error de reconstrucción (norma L2)
        """
        return np.linalg.norm(original.flatten() - reconstructed.flatten())

    def get_model_info(self) -> Dict:
        """
        Obtiene información sobre el modelo PCA entrenado.
        
        Returns:
            Diccionario con información del modelo
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
