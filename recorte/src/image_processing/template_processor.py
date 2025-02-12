"""
Módulo para el procesamiento de templates y visualizaciones de recorte de imágenes.

Este módulo proporciona funciones para:
- Cálculo de distancias de template
- Creación de templates de recorte
- Gestión de visualizaciones
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Tuple, Dict, Optional
"""
Módulo para el procesamiento de templates y visualizaciones de recorte de imágenes.

Este módulo proporciona funciones para:
- Cálculo de distancias de template
- Creación de templates de recorte
- Visualización de cada paso del proceso
"""


class TemplateProcessor:
    """
    Clase para el procesamiento de templates y visualizaciones.
    
    Esta clase maneja:
    - Cálculo de distancias desde regiones de búsqueda
    - Creación de templates de recorte
    - Generación de visualizaciones para cada paso
    """
    
    def __init__(self, visualization_dir: str = "visualization_results"):
        """
        Inicializa el procesador de templates.
        
        Args:
            visualization_dir: Directorio donde se guardarán las visualizaciones
        """
        self.visualization_dir = Path(visualization_dir)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        
        # Ruta al archivo de datos pre-calculados
        self.template_data_path = Path(__file__).parent.parent.parent.parent / "tools" / "template_analysis" / "template_analysis_results.json"
        self.template_data = self._load_template_data_file()
        
    def _load_template_data_file(self) -> Dict:
        """
        Carga el archivo JSON con los datos pre-calculados de los templates.
        
        Returns:
            Dict con los datos de todos los templates
        """
        try:
            if not self.template_data_path.exists():
                print(f"Advertencia: No se encontró el archivo de datos pre-calculados en {self.template_data_path}")
                return {}
                
            with open(self.template_data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error cargando archivo de datos pre-calculados: {str(e)}")
            return {}
    
    def validate_coord_name(self, coord_name: str) -> None:
        """
        Valida que el nombre de coordenada sea coord1 o coord2.
        
        Args:
            coord_name: Nombre de la coordenada a validar
            
        Raises:
            ValueError: Si el nombre de coordenada no es válido
        """
        if coord_name.lower() not in ["coord1", "coord2"]:
            raise ValueError(f"Solo se permiten coord1 y coord2. Recibido: {coord_name}")

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
        return self.template_data.get(coord_name)

    def validate_intersection_point(self, x: int, y: int, a: int, b: int, c: int, d: int) -> bool:
        """
        Valida que el punto de intersección está dentro de los límites válidos.
        
        Args:
            x: Coordenada x del punto de intersección (d)
            y: Coordenada y del punto de intersección (c)
            a,b,c,d: Distancias calculadas
            
        Returns:
            bool: True si el punto es válido, False en caso contrario
        """
        # El punto (d,c) siempre es válido si las distancias son válidas
        return True

    def calculate_template_distances(self, 
                                  search_region: np.ndarray, 
                                  template_size: int = 64) -> Tuple[int, int, int, int]:
        """
        Calcula las distancias a,b,c,d desde la región de búsqueda al template original.
        
        Args:
            search_region: Matriz binaria con la región de búsqueda
            template_size: Tamaño del template original
            
        Returns:
            Tuple con las distancias (a,b,c,d)
            
        Raises:
            ValueError: Si las dimensiones son inválidas o la región está vacía
        """
        # Validar tamaño del template
        if template_size != 64:
            raise ValueError("El tamaño del template debe ser 64x64")
            
        # Validar dimensiones de la región de búsqueda
        if search_region.shape != (64, 64):
            raise ValueError("La región de búsqueda debe ser 64x64")
            
        # Obtener límites de la región de búsqueda
        non_zero = np.nonzero(search_region)
        if len(non_zero[0]) == 0:
            raise ValueError("Región de búsqueda vacía")
            
        min_y, max_y = non_zero[0].min(), non_zero[0].max()
        min_x, max_x = non_zero[1].min(), non_zero[1].max()
        
        # Calcular distancias con límites
        a = min_y  # Distancia desde el borde superior
        d = min_x  # Distancia desde el borde izquierdo
        
        # Calcular ancho y alto de la región
        region_width = max_x - min_x + 1
        region_height = max_y - min_y + 1
        
        # Calcular distancias al final
        b = 63 - max_x  # Distancia al borde derecho
        c = 63 - max_y  # Distancia al borde inferior
        
        # Validar distancias
        if a + c >= 64:
            raise ValueError(f"Las distancias verticales a({a})+c({c}) suman más que el tamaño del template")
        if b + d >= 64:
            raise ValueError(f"Las distancias horizontales b({b})+d({d}) suman más que el tamaño del template")
            
        return a, b, c, d

    def create_cutting_template(self, 
                              a: int, 
                              b: int, 
                              c: int, 
                              d: int, 
                              template_size: int = 64) -> np.ndarray:
        """
        Crea el template de recorte basado en las distancias calculadas.
        
        Args:
            a: Distancia al borde superior
            b: Distancia al borde derecho
            c: Distancia al borde inferior
            d: Distancia al borde izquierdo
            template_size: Tamaño del template
            
        Returns:
            Matriz binaria con el template de recorte
            
        Raises:
            ValueError: Si las dimensiones son inválidas
        """
        template = np.zeros((template_size, template_size))
        
        # Calcular dimensiones del cuadrilátero
        height = c + a  # Suma de distancias verticales
        width = b + d   # Suma de distancias horizontales
        
        # Validar dimensiones básicas
        if height <= 0:
            raise ValueError(f"Altura inválida: {height} (a={a}, c={c})")
        if width <= 0:
            raise ValueError(f"Ancho inválido: {width} (b={b}, d={d})")
        
        # Validar y ajustar coordenadas
        if a < 0:
            a = 0
        if d < 0:
            d = 0
            
        # Ajustar dimensiones si exceden límites
        if a + height > template_size:
            total = a + c
            if total > 0:
                ratio_a = a / total
                ratio_c = c / total
                new_height = template_size - a
                c = int(new_height * ratio_c)
                height = c + a
        
        if d + width > template_size:
            total = b + d
            if total > 0:
                ratio_d = d / total
                ratio_b = b / total
                new_width = template_size - d
                b = int(new_width * ratio_b)
                width = b + d
        
        # Verificar dimensiones finales
        if a >= template_size or d >= template_size:
            raise ValueError(f"Coordenadas fuera de rango: a={a}, d={d}")
        if height <= 0 or width <= 0:
            raise ValueError(f"Dimensiones inválidas después de ajuste: {width}x{height}")
        if a + height > template_size or d + width > template_size:
            raise ValueError(f"Template excede límites después de ajuste: ({d},{a}) + {width}x{height}")
        
        # Crear template
        template[a:a+height, d:d+width] = 1
        
        return template

    def transform_intersection_point(self,
                                   local_point: Tuple[int, int],
                                   template: np.ndarray) -> Tuple[int, int]:
        """
        Transforma el punto de intersección del sistema local al sistema 64x64.
        
        Args:
            local_point: Punto (d,a) en coordenadas del template recortado
            template: Template completo 64x64
            
        Returns:
            Punto transformado al sistema 64x64
        """
        # Obtener límites del template
        non_zero = np.nonzero(template)
        min_y = non_zero[0].min()  # Offset vertical
        min_x = non_zero[1].min()  # Offset horizontal
        
        # Transformar sumando offsets
        x = min_x + local_point[0]
        y = min_y + local_point[1]
        
        return (x, y)

    def validate_template_bounds(self, template, labeled_point, intersection_point):
        """
        Valida que el template mantenga sus dimensiones dentro de los límites.
        
        Args:
            template: Template de recorte
            labeled_point: Coordenadas del punto etiquetado
            intersection_point: Coordenadas del punto de intersección
            
        Returns:
            Dict con información de dimensiones y límites validados
            
        Raises:
            ValueError: Si las dimensiones o coordenadas son inválidas
        """
        # Obtener dimensiones del template
        non_zero = np.nonzero(template)
        min_y, max_y = non_zero[0].min(), non_zero[0].max()
        min_x, max_x = non_zero[1].min(), non_zero[1].max()
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        
        # Verificar dimensiones originales
        if width > 64 or height > 64:
            raise ValueError(f"Template original excede límites: {width}x{height}")
        
        # Verificar punto de intersección
        if not (0 <= intersection_point[0] < 64 and 0 <= intersection_point[1] < 64):
            raise ValueError(f"Punto de intersección fuera de límites: {intersection_point}")
        
        # Verificar punto etiquetado
        if not (0 <= labeled_point[0] < 64 and 0 <= labeled_point[1] < 64):
            raise ValueError(f"Punto etiquetado fuera de límites: {labeled_point}")
        
        return {
            'width': width,
            'height': height,
            'original_bounds': {
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y
            }
        }
        
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
            intersection_point: Coordenadas del punto de intersección local (d,a)
            
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
