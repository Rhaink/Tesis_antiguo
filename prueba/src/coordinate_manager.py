"""
Módulo para la gestión de coordenadas de búsqueda.
"""

import json
from typing import Dict, List, Tuple
from pathlib import Path

class CoordinateManager:
    """
    Clase para gestionar las coordenadas de búsqueda de puntos anatómicos.
    
    Esta clase maneja:
    - Carga de coordenadas desde archivo JSON
    - Validación de datos de coordenadas
    - Acceso a coordenadas específicas
    """
    
    def __init__(self):
        """Inicializa el gestor de coordenadas."""
        self.template_data = {}  # Datos completos del JSON
        self.search_coordinates = {}  # Coordenadas de búsqueda generadas

    def read_search_coordinates(self, json_path: str) -> None:
        """
        Lee las coordenadas de búsqueda desde un archivo JSON.
        Todo se maneja en formato (y,x)
        
        Args:
            json_path: Ruta al archivo JSON con las coordenadas
            
        Raises:
            FileNotFoundError: Si no se encuentra el archivo
            ValueError: Si el formato del archivo es inválido
        """
        if not Path(json_path).exists():
            raise FileNotFoundError(f"No se encontró el archivo {json_path}")
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Validar estructura del JSON
            if not isinstance(data, dict):
                raise ValueError("El archivo JSON debe contener un objeto")
                
            # Procesar todas las coordenadas disponibles
            for coord_name, coord_info in data.items():
                # Extraer información del template y región
                template_bounds = coord_info.get('template_bounds', {})
                region_bounds = coord_info.get('region_bounds', {})
                intersection = coord_info.get('intersection_point', {})
                
                # Validar estructura completa
                required_template_keys = {'min_x', 'max_x', 'min_y', 'max_y', 'width', 'height'}
                required_region_keys = {'sup', 'inf', 'left', 'right', 'width', 'height'}
                required_intersection_keys = {'x', 'y'}
                
                if not all(key in template_bounds for key in required_template_keys):
                    raise ValueError(f"Faltan campos en template_bounds para {coord_name}")
                    
                if not all(key in region_bounds for key in required_region_keys):
                    raise ValueError(f"Faltan campos en region_bounds para {coord_name}")
                    
                if not all(key in intersection for key in required_intersection_keys):
                    raise ValueError(f"Faltan campos en intersection_point para {coord_name}")
                
                # Obtener dimensiones del template
                template_height = int(template_bounds["height"])
                template_width = int(template_bounds["width"])
                
                # Obtener punto de intersección
                intersection_y = int(intersection["y"])
                intersection_x = int(intersection["x"])
                
                # El punto de búsqueda puede estar en cualquier lugar donde:
                # 1. El template quede dentro de la imagen cuando su punto de intersección
                #    se alinee con el punto de búsqueda
                # 2. El punto de búsqueda esté dentro de la región definida
                
                # Calcular los límites de la región de búsqueda
                # Permitir que el punto de intersección se mueva por toda la región
                # mientras el template permanezca dentro de la imagen
                safe_left = region_bounds['left']
                safe_right = region_bounds['right']
                safe_sup = region_bounds['sup']
                safe_inf = region_bounds['inf']
                
                print(f"\nRegión de búsqueda para {coord_name}:")
                print(f"Región original: X({region_bounds['left']},{region_bounds['right']}) Y({region_bounds['sup']},{region_bounds['inf']})")
                print(f"Template: {template_width}x{template_height}")
                print(f"Punto de intersección: ({intersection_x}, {intersection_y})")
                
                # Generar coordenadas dentro de la región de búsqueda original
                search_coords = []
                for y in range(region_bounds['sup'], region_bounds['inf'] + 1):
                    for x in range(region_bounds['left'], region_bounds['right'] + 1):
                        # Para cada coordenada (y,x), verificar que al alinear
                        # el punto de intersección del template con ella,
                        # el template completo quede dentro de la imagen
                        template_y = y - intersection_y  # Borde superior del template
                        template_x = x - intersection_x  # Borde izquierdo del template
                        
                        # Verificar que el template completo quede dentro de los límites
                        if (template_y >= 0 and template_x >= 0 and  # No negativo
                            template_y + template_height <= 64 and    # No excede abajo
                            template_x + template_width <= 64):       # No excede derecha
                            search_coords.append((y, x))
                
                # Guardar datos
                self.template_data[coord_name] = coord_info
                self.search_coordinates[coord_name] = search_coords
                
                print(f"Coordenadas de búsqueda generadas: {len(search_coords)}")
                
            if not self.template_data:
                raise ValueError("No se encontraron coordenadas válidas")
                
        except json.JSONDecodeError:
            raise ValueError("El archivo JSON no es válido")
        except Exception as e:
            raise ValueError(f"Error al leer coordenadas: {str(e)}")

    def get_search_coordinates(self, coord_name: str) -> List[Tuple[int, int]]:
        """
        Obtiene las coordenadas de búsqueda para un punto específico.
        
        Args:
            coord_name: Nombre del punto anatómico
            
        Returns:
            Lista de coordenadas de búsqueda
            
        Raises:
            KeyError: Si no se encuentra el punto especificado
        """
        if coord_name not in self.search_coordinates:
            raise KeyError(f"No se encontraron coordenadas para {coord_name}")
            
        return self.search_coordinates[coord_name]

    def get_template_bounds(self, coord_name: str) -> Dict:
        """
        Obtiene los límites del template para un punto específico.
        
        Args:
            coord_name: Nombre del punto anatómico
            
        Returns:
            Diccionario con los límites del template incluyendo:
            - min_x, max_x: límites horizontales
            - min_y, max_y: límites verticales
            - width, height: dimensiones del template
            
        Raises:
            KeyError: Si no se encuentra el punto especificado
        """
        if coord_name not in self.template_data:
            raise KeyError(f"No se encontraron datos para {coord_name}")
            
        return self.template_data[coord_name]["template_bounds"]

    def get_region_bounds(self, coord_name: str) -> Dict:
        """
        Obtiene los límites de la región de búsqueda para un punto específico.
        
        Args:
            coord_name: Nombre del punto anatómico
            
        Returns:
            Diccionario con los límites de la región incluyendo:
            - sup, inf: límites verticales
            - left, right: límites horizontales
            - width, height: dimensiones de la región
            
        Raises:
            KeyError: Si no se encuentra el punto especificado
        """
        if coord_name not in self.template_data:
            raise KeyError(f"No se encontraron datos para {coord_name}")
            
        return self.template_data[coord_name]["region_bounds"]

    def get_intersection_point(self, coord_name: str) -> Tuple[int, int]:
        """
        Obtiene el punto de intersección para un punto específico.
        
        Args:
            coord_name: Nombre del punto anatómico
            
        Returns:
            Tupla con coordenadas (y, x) del punto de intersección.
            IMPORTANTE: Se convierte de formato JSON (x,y) a formato numpy (y,x)
            para mantener consistencia con el sistema de coordenadas del proyecto.
            
        Raises:
            KeyError: Si no se encuentra el punto especificado
        """
        if coord_name not in self.template_data:
            raise KeyError(f"No se encontraron datos para {coord_name}")
            
        intersection = self.template_data[coord_name]["intersection_point"]
        # Convertir de (x,y) a (y,x) para mantener consistencia
        return (int(intersection["y"]), int(intersection["x"]))

    def get_template_size(self, coord_name: str) -> Tuple[int, int]:
        """
        Obtiene el tamaño del template para un punto específico.
        
        Args:
            coord_name: Nombre del punto anatómico
            
        Returns:
            Tupla con (ancho, alto) del template
            
        Raises:
            KeyError: Si no se encuentra el punto especificado
        """
        template_bounds = self.get_template_bounds(coord_name)
        return (int(template_bounds["width"]), int(template_bounds["height"]))

    def get_all_coordinate_names(self) -> List[str]:
        """
        Obtiene la lista de nombres de todos los puntos de interés.
        
        Returns:
            Lista de nombres de puntos de interés
        """
        return list(self.template_data.keys())

    def validate_coordinates(self) -> bool:
        """
        Valida que todos los datos de coordenadas sean consistentes.
        
        Returns:
            True si los datos son válidos, False en caso contrario
        """
        try:
            for coord_name in self.template_data:
                # Verificar que existan coordenadas de búsqueda
                if coord_name not in self.search_coordinates:
                    return False
                    
                if not self.search_coordinates[coord_name]:
                    return False
                
                # Verificar datos del template
                template_bounds = self.get_template_bounds(coord_name)
                if not all(template_bounds[key] > 0 for key in ['width', 'height']):
                    return False
                
                # Verificar punto de intersección
                intersection = self.get_intersection_point(coord_name)
                if not all(x >= 0 for x in intersection):
                    return False
                
                # Verificar región de búsqueda
                region_bounds = self.get_region_bounds(coord_name)
                if not all(region_bounds[key] >= 0 for key in ['sup', 'inf', 'left', 'right']):
                    return False
                    
            return True
            
        except Exception:
            return False
