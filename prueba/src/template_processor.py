"""
Módulo para el procesamiento de templates y recorte de imágenes.
Implementa la lógica exacta del módulo de entrenamiento para mantener consistencia.
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
    - Creación y validación de templates de recorte
    - Extracción de regiones usando la misma geometría que el entrenamiento
    """
    
    def __init__(self, template_data_path: str):
        """
        Inicializa el procesador de templates.
        
        Args:
            template_data_path: Ruta al archivo JSON con datos pre-calculados
        """
        self.template_data_path = Path(template_data_path)
        self.template_data = self._load_template_data_file()
        
    def _validate_coordinates(self, y: int, x: int, context: str = "") -> None:
        """
        Valida que las coordenadas estén en el rango 0-63.
        
        Args:
            y: Coordenada y
            x: Coordenada x
            context: Contexto para el mensaje de error
            
        Raises:
            ValueError: Si las coordenadas están fuera de rango
        """
        if not (0 <= y <= 63 and 0 <= x <= 63):
            raise ValueError(
                f"Coordenadas fuera de rango (0-63) {context}: ({y}, {x})"
            )
        
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
    
    def _validate_point(self, point: Tuple[int, int], context: str = "") -> None:
        """
        Valida que un punto (y,x) esté en el rango 0-63.
        
        Args:
            point: Tuple con coordenadas (y,x)
            context: Contexto para el mensaje de error
            
        Raises:
            ValueError: Si el punto está fuera de rango
        """
        self._validate_coordinates(point[0], point[1], context)

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
    
    def crop_aligned_image(self,
                          image: np.ndarray,
                          template: np.ndarray,
                          labeled_point: Tuple[int, int],
                          intersection_point: Tuple[int, int]) -> np.ndarray: 
        """
        Recorta la imagen usando el template alineado con el punto etiquetado.
        
        IMPORTANTE: Todas las coordenadas están en formato (y,x)
        para mantener consistencia con el orden de indexación de numpy arrays [y,x].
        
        Args:
            image: Imagen original
            template: Template de recorte
            labeled_point: Coordenadas (y,x) del punto etiquetado
            intersection_point: Coordenadas (y,x) del punto de intersección

        Returns:
            Imagen recortada del tamaño del template
            
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
        # labeled_point y intersection_point están en formato (y,x)
        dy = labeled_point[0] - (min_y + intersection_point[0])  # desplazamiento en y
        dx = labeled_point[1] - (min_x + intersection_point[1])  # desplazamiento en x
        
        # Calcular coordenadas finales con límites
        # Usar height para límite y, width para límite x
        final_min_y = np.clip(min_y + dy, 0, 63 - height + 1)
        final_min_x = np.clip(min_x + dx, 0, 63 - width + 1)
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

    def extract_region(self,
                      image: np.ndarray,
                      template_data: Dict,
                      search_point: Tuple[int, int]) -> np.ndarray: 
        """
        Extrae una región de la imagen usando la misma geometría que en entrenamiento.
        
        El proceso:
        1. Obtiene las dimensiones del template y el punto de intersección
        2. Calcula la posición donde el punto de intersección del template
           se alinea con el punto de búsqueda actual
        3. Extrae la región usando esa posición y las dimensiones del template
        
        IMPORTANTE: Todo se maneja en formato (y,x) para mantener consistencia:
        - search_point viene como (y,x) desde CoordinateManager
        - intersection_point del JSON viene como (y,x)
        - numpy arrays usan [y,x] para indexación
        
        Args:
            image: Imagen original (64x64)
            template_data: Datos del template
            search_point: Punto de búsqueda actual (y,x)
            
        Returns:
            Región extraída del tamaño del template (height x width)
            
        Raises:
            ValueError: Si las coordenadas están fuera de rango
        """
        try:
            # Validar dimensiones de la imagen
            if image.shape != (64, 64):
                raise ValueError(f"La imagen debe ser 64x64, recibido: {image.shape}")
                
            # Obtener datos del template
            template_bounds = template_data["template_bounds"]
            intersection_point = template_data["intersection_point"]
            
            # Obtener dimensiones del template (height=filas, width=columnas)
            height = int(template_bounds["height"])
            width = int(template_bounds["width"])
            
            # El template se coloca de forma que su punto de intersección
            # coincide con el punto de búsqueda actual
            
            # Obtener y validar coordenadas
            search_y, search_x = search_point
            self._validate_point((search_y, search_x), "punto de búsqueda")
            
            # Las coordenadas del punto de intersección son relativas al template
            # y ya están validadas al cargar el template_data
            intersection_y = int(intersection_point["y"])
            intersection_x = int(intersection_point["x"])
            
            # Para obtener el punto de inicio del template:
            # 1. El punto de búsqueda es donde debe estar el punto de intersección
            # 2. Retrocedemos desde ese punto según la posición de intersección en el template
            start_y = search_y - intersection_y
            start_x = search_x - intersection_x
            
            # Validar que el template esté dentro de los límites de la imagen
            # No deberíamos llegar aquí si las coordenadas de búsqueda son válidas
            if start_y < 0 or start_x < 0 or start_y + height > 64 or start_x + width > 64:
                print(f"\nDetalles de coordenadas:")
                print(f"Punto de búsqueda: ({search_y}, {search_x})")
                print(f"Punto de intersección: ({intersection_y}, {intersection_x})")
                print(f"Dimensiones del template: {height}x{width}")
                print(f"Inicio calculado: ({start_y}, {start_x})")
                raise ValueError(f"Template fuera de límites en ({start_y}, {start_x})")
            
            # Extraer la región alineada
            region = image[start_y:start_y + height, start_x:start_x + width]
            
            # Verificar dimensiones del resultado
            if region.shape != (height, width):
                raise ValueError(
                    f"Dimensiones incorrectas en región extraída: "
                    f"esperado ({height}, {width}), "
                    f"obtenido {region.shape}"
                )
            
            return region
            
        except Exception as e:
            print(f"\nError detallado: {str(e)}")
            raise ValueError(f"Error extrayendo región en y:{search_y}, x:{search_x}: {str(e)}")
