"""
Módulo para el procesamiento de templates y recorte de imágenes.
Versión simplificada del proyecto original pulmo_align.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class TemplateProcessor:
    """
    Clase para el procesamiento de templates y recorte de imágenes.
    
    Esta clase maneja:
    - Carga de datos pre-calculados de templates
    - Creación de templates de recorte
    - Alineación y recorte de imágenes
    """
    
    def __init__(self, template_data_path: str):
        """
        Inicializa el procesador de templates.
        
        Args:
            template_data_path: Ruta al archivo JSON con datos pre-calculados
        """
        self.template_data_path = Path(template_data_path)
        self.template_data = self._load_template_data_file()
        
    def _load_template_data_file(self) -> Dict:
        """
        Carga el archivo JSON con los datos pre-calculados de los templates.
        
        Returns:
            Dict con los datos de todos los templates
        """
        try:
            if not self.template_data_path.exists():
                raise FileNotFoundError(f"No se encontró el archivo de datos en {self.template_data_path}")
                
            with open(self.template_data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error cargando archivo de datos pre-calculados: {str(e)}")
            raise
    
    def validate_coord_name(self, coord_name: str) -> None:
        """
        Valida que el nombre de coordenada tenga el formato correcto.
        
        Args:
            coord_name: Nombre de la coordenada a validar
            
        Raises:
            ValueError: Si el nombre de coordenada no es válido
        """
        if not coord_name.lower().startswith("coord"):
            raise ValueError(f"El nombre debe empezar con 'coord'. Recibido: {coord_name}")
            
        try:
            coord_num = int(coord_name[5:])
            if coord_num < 1 or coord_num > 15:
                raise ValueError
        except ValueError:
            raise ValueError(f"Número de coordenada inválido en: {coord_name}")

    def load_template_data(self, coord_name: str) -> Optional[Dict]:
        """
        Obtiene los datos pre-calculados para una coordenada específica.
        
        Args:
            coord_name: Nombre de la coordenada (e.g., "coord1")
            
        Returns:
            Dict con los datos del template o None si no existe
            
        Raises:
            ValueError: Si el nombre de coordenada no es válido
        """
        self.validate_coord_name(coord_name)
        return self.template_data.get(coord_name.lower())
    
    def create_cutting_template(self, 
                              template_bounds: Dict,
                              template_size: int = 64) -> np.ndarray:
        """
        Crea el template de recorte basado en los límites especificados.
        
        Args:
            template_bounds: Diccionario con los límites del template
            template_size: Tamaño del template
            
        Returns:
            Matriz binaria con el template de recorte
            
        Raises:
            ValueError: Si las dimensiones son inválidas
        """
        template = np.zeros((template_size, template_size))
        
        width = template_bounds["width"]
        height = template_bounds["height"]
        
        # Validar dimensiones
        if width <= 0 or height <= 0:
            raise ValueError(f"Dimensiones inválidas: {width}x{height}")
            
        if width > template_size or height > template_size:
            raise ValueError(f"Template excede límites: {width}x{height}")
        
        # Crear template comenzando desde (0,0)
        template[0:height, 0:width] = 1
        
        return template
    
    def crop_aligned_image(self,
                          image: np.ndarray,
                          template: np.ndarray,
                          labeled_point: Tuple[int, int],
                          intersection_point: Tuple[int, int]) -> np.ndarray:
        """
        Recorta la imagen usando el template alineado con el punto etiquetado.
        
        Args:
            image: Imagen original
            template: Template de recorte
            labeled_point: Coordenadas del punto etiquetado
            intersection_point: Coordenadas del punto de intersección
            
        Returns:
            Imagen recortada del tamaño del template de recorte
        """
        # Obtener dimensiones del template
        non_zero = np.nonzero(template)
        min_y, max_y = non_zero[0].min(), non_zero[0].max()
        min_x, max_x = non_zero[1].min(), non_zero[1].max()
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        
        # Calcular desplazamiento
        dx = labeled_point[0] - (min_x + intersection_point[0])
        dy = labeled_point[1] - (min_y + intersection_point[1])
        
        # Calcular coordenadas finales con límites
        final_min_x = np.clip(min_x + dx, 0, 64 - width)
        final_min_y = np.clip(min_y + dy, 0, 64 - height)
        final_max_x = final_min_x + width
        final_max_y = final_min_y + height
        
        # Recortar y retornar
        return image[final_min_y:final_max_y, final_min_x:final_max_x]

    def extract_region(self,
                      image: np.ndarray,
                      template_data: Dict,
                      search_point: Tuple[int, int]) -> np.ndarray:
        """
        Extrae una región de la imagen usando el template y punto de búsqueda.
        
        El proceso:
        1. El template mantiene sus dimensiones originales (ej: 46x45)
        2. El punto de intersección mantiene su posición relativa dentro del template (ej: 24,0)
        3. El template se mueve para que su punto de intersección coincida con cada punto de búsqueda
        4. Se extrae la región correspondiente
        
        Args:
            image: Imagen original
            template_data: Datos del template
            search_point: Punto de búsqueda actual (y, x)
            
        Returns:
            Región extraída del tamaño del template
            
        Raises:
            ValueError: Si el template queda fuera de los límites de la imagen
        """
        # Obtener dimensiones del template
        template_width = int(template_data["template_bounds"]["width"])   # ej: 46
        template_height = int(template_data["template_bounds"]["height"]) # ej: 45
        
        # Obtener punto de intersección (posición fija dentro del template)
        intersection_x = int(template_data["intersection_point"]["x"])    # ej: 24
        intersection_y = int(template_data["intersection_point"]["y"])    # ej: 0
        
        # El punto de búsqueda (search_point) está en formato (y, x)
        search_y, search_x = search_point
        
        # Calcular la esquina superior izquierda del template
        # Cuando el punto de intersección coincide con el punto de búsqueda
        template_start_x = search_x - intersection_x
        template_start_y = search_y - intersection_y
        
        # Verificar límites
        if (template_start_x < 0 or 
            template_start_y < 0 or 
            template_start_x + template_width > image.shape[1] or 
            template_start_y + template_height > image.shape[0]):
            raise ValueError(f"Template fuera de límites en punto ({search_x}, {search_y})")
        
        # Extraer región
        region = image[template_start_y:template_start_y + template_height,
                      template_start_x:template_start_x + template_width]
        
        # Verificar dimensiones de la región extraída
        if region.shape != (template_height, template_width):
            raise ValueError(
                f"Dimensiones incorrectas: {region.shape} vs esperado ({template_height}, {template_width})"
            )
        
        return region
