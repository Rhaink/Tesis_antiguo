"""
Módulo para el procesamiento de templates y visualizaciones de recorte de imágenes.

Este módulo proporciona funciones para:
- Cálculo de distancias de template (sistema 0-based)
- Creación de templates de recorte
- Gestión de visualizaciones
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Tuple, Dict, Optional

class TemplateProcessor:
    """
    Clase para el procesamiento de templates y visualizaciones.
    
    Esta clase maneja:
    - Cálculo de distancias desde regiones de búsqueda (0-based)
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
        self.template_data_path = Path(__file__).parent.parent.parent.parent / "resultados" / "analisis_regiones" / "prueba_2" /  "template_analysis_results.json" 
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
    
    def _validate_coordinates(self, x: int, y: int, context: str = "") -> None:
        """
        Valida que las coordenadas estén en el rango 0-63.
        
        Args:
            x (int): Coordenada x
            y (int): Coordenada y
            context (str): Contexto para el mensaje de error
            
        Raises:
            ValueError: Si las coordenadas están fuera de rango
        """
        if not (0 <= x <= 63 and 0 <= y <= 63):
            raise ValueError(
                f"Coordenadas fuera de rango (0-63) {context}: ({x}, {y})"
            )
            
    def _validate_point(self, point: Tuple[int, int], context: str = "") -> None:
        """
        Valida que un punto (x,y) esté en el rango 0-63.
        
        Args:
            point: Tuple con coordenadas (x,y)
            context: Contexto para el mensaje de error
            
        Raises:
            ValueError: Si el punto está fuera de rango
        """
        self._validate_coordinates(point[0], point[1], context)
    
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
        Todas las coordenadas se manejan en sistema 0-based (0-63).
        
        Args:
            x: Coordenada x del punto de intersección
            y: Coordenada y del punto de intersección
            a,b,c,d: Distancias calculadas en sistema 0-based
            
        Returns:
            bool: True si el punto es válido
            
        Raises:
            ValueError: Si el punto está fuera de rango
        """
        self._validate_coordinates(x, y, "punto de intersección")
        return True

    def calculate_template_distances(self, 
                                  search_region: np.ndarray, 
                                  template_size: int = 64) -> Tuple[int, int, int, int]:
        """
        Calcula las distancias a,b,c,d desde la región de búsqueda al template original.
        Todas las distancias se calculan en sistema 0-based (0-63).
        
        Args:
            search_region: Matriz binaria con la región de búsqueda (0-based)
            template_size: Tamaño del template original
            
        Returns:
            Tuple[int, int, int, int]: Distancias (a,b,c,d) en sistema 0-based
            
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
        
        # Validar que los límites están en rango 0-63
        for val, name in [(min_y, "min_y"), (max_y, "max_y"), 
                         (min_x, "min_x"), (max_x, "max_x")]:
            if not (0 <= val <= 63):
                raise ValueError(f"Límite {name}={val} fuera del rango 0-63")
        
        # Calcular distancias (todo en sistema 0-based)
        a = min_y      # Distancia desde el borde superior (0-based)
        d = min_x      # Distancia desde el borde izquierdo (0-based)
        b = 63 - max_x # Distancia al borde derecho (0-based)
        c = 63 - max_y # Distancia al borde inferior (0-based)
        
        # Validar que las distancias son válidas
        if a + c >= 64:
            raise ValueError(f"Las distancias verticales a({a})+c({c}) exceden el rango 0-63")
        if b + d >= 64:
            raise ValueError(f"Las distancias horizontales b({b})+d({d}) exceden el rango 0-63")
            
        return a, b, c, d

    def create_cutting_template(self, 
                              a: int, 
                              b: int, 
                              c: int, 
                              d: int, 
                              template_size: int = 64) -> np.ndarray:
        """
        Crea el template de recorte basado en las distancias calculadas.
        Todas las distancias deben estar en sistema 0-based (0-63).
        
        Args:
            a: Distancia al borde superior (0-based)
            b: Distancia al borde derecho (0-based)
            c: Distancia al borde inferior (0-based)
            d: Distancia al borde izquierdo (0-based)
            template_size: Tamaño del template
            
        Returns:
            np.ndarray: Matriz binaria con el template de recorte
            
        Raises:
            ValueError: Si las dimensiones son inválidas
        """
        # Validar que las distancias están en rango
        for val, name in [(a, "a"), (b, "b"), (c, "c"), (d, "d")]:
            if not (0 <= val <= 63):
                raise ValueError(f"Distancia {name}={val} fuera del rango 0-63")
        
        template = np.zeros((template_size, template_size))
        
        # Calcular dimensiones del cuadrilátero (en sistema 0-based)
        height = c + a  # Suma de distancias verticales
        width = b + d   # Suma de distancias horizontales
        
        # Validar dimensiones básicas
        if height <= 0:
            raise ValueError(f"Altura inválida: {height} (a={a}, c={c})")
        if width <= 0:
            raise ValueError(f"Ancho inválido: {width} (b={b}, d={d})")
        
        # Validar dimensiones finales
        if a >= template_size or d >= template_size:
            raise ValueError(f"Coordenadas fuera de rango: a={a}, d={d}")
        if height <= 0 or width <= 0:
            raise ValueError(f"Dimensiones inválidas: {width}x{height}")
        if a + height > template_size or d + width > template_size:
            raise ValueError(f"Template excede límites: ({d},{a}) + {width}x{height}")
        
        # Crear template (coordenadas 0-based)
        template[a:a+height, d:d+width] = 1
        
        return template

    def transform_intersection_point(self,
                                   local_point: Tuple[int, int],
                                   template: np.ndarray) -> Tuple[int, int]:
        """
        Transforma el punto de intersección del sistema local al sistema 64x64 (0-based).
        
        Args:
            local_point: Punto (x,y) en coordenadas del template (0-based)
            template: Template completo 64x64 (0-based)
            
        Returns:
            Tuple[int, int]: Punto transformado al sistema 64x64 (0-based)
            
        Raises:
            ValueError: Si las coordenadas están fuera de rango
        """
        # Obtener límites del template
        non_zero = np.nonzero(template)
        if len(non_zero[0]) == 0:
            raise ValueError("Template vacío")
            
        min_y = non_zero[0].min()  # Offset vertical (0-based)
        min_x = non_zero[1].min()  # Offset horizontal (0-based)
        
        # Transformar sumando offsets
        x = min_x + local_point[0]  # Mantiene sistema 0-based
        y = min_y + local_point[1]  # Mantiene sistema 0-based
        
        # Validar rango 0-63
        self._validate_coordinates(x, y, "punto transformado")
        
        return (x, y)

    def validate_template_bounds(self, template, labeled_point, intersection_point):
        """
        Valida que el template mantenga sus dimensiones dentro de los límites.
        Todas las coordenadas se validan en sistema 0-based (0-63).
        
        Args:
            template: Template de recorte (matriz 64x64)
            labeled_point: Coordenadas (x,y) del punto etiquetado (0-based)
            intersection_point: Coordenadas (x,y) del punto de intersección (0-based)
            
        Returns:
            Dict: Información de dimensiones y límites validados
            
        Raises:
            ValueError: Si las dimensiones o coordenadas son inválidas
        """
        # Validar puntos en rango 0-63
        self._validate_point(labeled_point, "punto etiquetado")
        self._validate_point(intersection_point, "punto de intersección")
        
        # Obtener dimensiones del template
        non_zero = np.nonzero(template)
        min_y, max_y = non_zero[0].min(), non_zero[0].max()
        min_x, max_x = non_zero[1].min(), non_zero[1].max()
        
        # Validar límites en rango 0-63
        for val, name in [(min_y, "min_y"), (max_y, "max_y"),
                         (min_x, "min_x"), (max_x, "max_x")]:
            if not (0 <= val <= 63):
                raise ValueError(f"Límite {name}={val} fuera del rango 0-63")
        
        # Calcular dimensiones
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        
        # Verificar dimensiones
        if width > 64 or height > 64:
            raise ValueError(f"Template excede límites: {width}x{height}")
        
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
        Todas las coordenadas se manejan en sistema 0-based (0-63).
        
        Args:
            image: Imagen original
            template: Template de recorte (matriz 64x64)
            labeled_point: Coordenadas (x,y) del punto etiquetado (0-based)
            intersection_point: Coordenadas (x,y) de intersección (0-based)
            
        Returns:
            np.ndarray: Imagen recortada del tamaño del template
            
        Raises:
            ValueError: Si las coordenadas están fuera de rango
        """
        # Validar puntos en rango 0-63
        self._validate_point(labeled_point, "punto etiquetado")
        self._validate_point(intersection_point, "punto de intersección")
        
        # Obtener dimensiones del template
        non_zero = np.nonzero(template)
        min_y, max_y = non_zero[0].min(), non_zero[0].max()
        min_x, max_x = non_zero[1].min(), non_zero[1].max()
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        
        # Calcular desplazamiento preservando sistema 0-based
        dx = labeled_point[0] - (min_x + intersection_point[0])
        dy = labeled_point[1] - (min_y + intersection_point[1])
        
        # Calcular coordenadas finales con límites
        final_min_x = np.clip(min_x + dx, 0, 63 - width + 1)
        final_min_y = np.clip(min_y + dy, 0, 63 - height + 1)
        final_max_x = final_min_x + width
        final_max_y = final_min_y + height
        
        # Validar coordenadas finales
        if not (0 <= final_min_x <= 63 and 0 <= final_min_y <= 63 and
                0 <= final_max_x <= 64 and 0 <= final_max_y <= 64):
            raise ValueError(
                f"Coordenadas de recorte fuera de rango: "
                f"({final_min_x},{final_min_y}) -> ({final_max_x},{final_max_y})"
            )
        
        # Recortar y retornar
        return image[final_min_y:final_max_y, final_min_x:final_max_x]
