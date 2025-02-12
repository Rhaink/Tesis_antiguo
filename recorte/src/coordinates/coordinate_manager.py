"""
Módulo para el manejo y procesamiento de coordenadas en imágenes pulmonares.

Este módulo proporciona clases y funciones para:
- Lectura de coordenadas desde archivos CSV y JSON
- Gestión de coordenadas de puntos de referencia
- Procesamiento de coordenadas de búsqueda
"""

import csv
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class CoordinateManager:
    """
    Clase para gestionar las coordenadas de puntos de referencia y búsqueda en imágenes pulmonares.
    
    Esta clase maneja:
    - Lectura de coordenadas desde archivos CSV
    - Lectura de coordenadas de búsqueda desde archivos JSON
    - Procesamiento y validación de coordenadas
    - Gestión de configuraciones de coordenadas
    """
    
    def __init__(self):
        """Inicializa el gestor de coordenadas con la configuración predeterminada."""
        self.coord_data = {}
        self.coordinates = {}
        self.search_coordinates = {}
        
    def _calculate_region_bounds(self, coord_points):
        """
        Calcula los límites de una región basado en las coordenadas de búsqueda.
        
        Args:
            coord_points (List[List[int]]): Lista de coordenadas [x,y]
            
        Returns:
            Dict: Diccionario con los límites calculados (sup, inf, left, right, width, height)
        """
        if not coord_points:
            return None
            
        # Extraer x,y de los puntos
        x_coords = [p[0] for p in coord_points]
        y_coords = [p[1] for p in coord_points]
        
        # Calcular límites
        left = min(x_coords)
        right = max(x_coords)
        sup = min(y_coords)
        inf = max(y_coords)
        
        # Calcular dimensiones
        width = right - left + 1
        height = inf - sup + 1
        
        return {
            "sup": sup,
            "inf": inf,
            "left": left,
            "right": right,
            "width": width,
            "height": height
        }

    def read_coordinates(self, filename: str) -> None:
        """
        Lee las coordenadas desde un archivo CSV.
        
        Args:
            filename (str): Ruta al archivo CSV con las coordenadas
            
        Raises:
            FileNotFoundError: Si no se encuentra el archivo
            ValueError: Si el formato del archivo es inválido
        """
        try:
            coordinates = {}
            with open(filename, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    # Convertir el índice a entero
                    index = int(row[0])
                    # Inicializar el diccionario para este índice
                    coordinates[index] = {}
                    
                    # MODIFICACIÓN: Procesar solo las coordenadas 1 y 2
                    for i in range(2):
                        # Calcular las posiciones en el row para cada coordenada
                        x_pos = 1 + (i * 2)  # Posición para x
                        y_pos = 2 + (i * 2)  # Posición para y
                        
                        # Verificar que tenemos suficientes datos
                        if x_pos < len(row) and y_pos < len(row):
                            coord_name = f"Coord{i+1}"
                            x = int(row[x_pos]) if row[x_pos] else 0
                            y = int(row[y_pos]) if row[y_pos] else 0
                            coordinates[index][coord_name] = (x, y)
                        else:
                            print(f"Advertencia: Datos faltantes para índice {index}, coordenada {i+1}")
                            coordinates[index][f"Coord{i+1}"] = (0, 0)
            
            self.coordinates = coordinates
        except Exception as e:
            raise ValueError(f"Error al leer el archivo de coordenadas: {str(e)}")

    def read_search_coordinates(self, filename: str) -> None:
        """
        Lee las coordenadas de búsqueda desde un archivo JSON.
        
        Args:
            filename (str): Ruta al archivo JSON con las coordenadas de búsqueda
            
        Raises:
            FileNotFoundError: Si no se encuentra el archivo
            ValueError: Si el formato del archivo es inválido
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                coordinates = {}
                self.coord_data = {}  # Reiniciar coord_data
                
                # MODIFICACIÓN: Procesar solo coord1 y coord2
                for i in range(1, 3):
                    coord_name = f"coord{i}"
                    if coord_name in data:
                        coordinates[coord_name] = data[coord_name]
                        # Calcular límites para esta coordenada
                        region_bounds = self._calculate_region_bounds(data[coord_name])
                        if region_bounds:
                            self.coord_data[f"Coord{i}"] = region_bounds
                    else:
                        print(f"Advertencia: {coord_name} no encontrada en el archivo")
                        coordinates[coord_name] = []
                
                self.search_coordinates = coordinates
        except Exception as e:
            raise ValueError(f"Error al leer el archivo de coordenadas de búsqueda: {str(e)}")

    def get_coordinate_config(self, coord_name: str) -> Optional[Dict]:
        """
        Obtiene la configuración para una coordenada específica.
        
        Args:
            coord_name (str): Nombre de la coordenada (e.g., "Coord1")
            
        Returns:
            Dict: Configuración de la coordenada o None si no existe
        """
        return self.coord_data.get(coord_name)

    def get_search_coordinates(self, coord_name: str) -> List[Tuple[int, int]]:
        """
        Obtiene las coordenadas de búsqueda para una coordenada específica.
        
        Args:
            coord_name (str): Nombre de la coordenada (e.g., "Coord1")
            
        Returns:
            List[Tuple[int, int]]: Lista de coordenadas de búsqueda
        """
        key = coord_name.lower()
        return self.search_coordinates.get(key, [])

    def get_image_coordinates(self, index: int) -> Optional[Dict]:
        """
        Obtiene las coordenadas para una imagen específica.
        
        Args:
            index (int): Índice de la imagen
            
        Returns:
            Dict: Coordenadas de la imagen o None si no existe
        """
        return self.coordinates.get(index)

    @staticmethod
    def calculate_center(sup: int, inf: int, left: int, right: int) -> Tuple[int, int]:
        """
        Calcula el centro de una región basada en sus límites.
        
        Args:
            sup (int): Límite superior
            inf (int): Límite inferior
            left (int): Límite izquierdo
            right (int): Límite derecho
            
        Returns:
            Tuple[int, int]: Coordenadas del centro (x, y)
        """
        center_x = (left + right) // 2
        center_y = (sup + inf) // 2
        return center_x, center_y

    @staticmethod
    def calculate_intersection(sup: int, inf: int, left: int, right: int) -> Tuple[int, int]:
        """
        Calcula el punto de intersección de una región.
        
        Args:
            sup (int): Límite superior
            inf (int): Límite inferior
            left (int): Límite izquierdo
            right (int): Límite derecho
            
        Returns:
            Tuple[int, int]: Coordenadas del punto de intersección (x, y)
        """
        return left, sup
