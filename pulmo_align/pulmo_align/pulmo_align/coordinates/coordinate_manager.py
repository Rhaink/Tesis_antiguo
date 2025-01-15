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
    
    Attributes:
        coord_data (Dict): Diccionario con la configuración de coordenadas para cada punto
        coordinates (Dict): Diccionario con las coordenadas leídas del archivo CSV
        search_coordinates (Dict): Diccionario con las coordenadas de búsqueda
    """
    
    def __init__(self):
        """Inicializa el gestor de coordenadas con la configuración predeterminada."""
        self.coord_data = {
            "Coord1": {"sup": 0, "inf": 47, "left": 22, "right": 25, "width": 47, "height": 47},
            "Coord2": {"sup": 38, "inf": 1, "left": 26, "right": 25, "width": 51, "height": 39},
            "Coord3": {"sup": 11, "inf": 38, "left": 3, "right": 42, "width": 45, "height": 49},
            "Coord4": {"sup": 10, "inf": 37, "left": 39, "right": 3, "width": 42, "height": 47},
            "Coord5": {"sup": 20, "inf": 28, "left": 1, "right": 45, "width": 46, "height": 48},
            "Coord6": {"sup": 22, "inf": 27, "left": 41, "right": 1, "width": 42, "height": 49},
            "Coord7": {"sup": 28, "inf": 16, "left": 0, "right": 46, "width": 46, "height": 44},
            "Coord8": {"sup": 30, "inf": 14, "left": 42, "right": 0, "width": 42, "height": 44},
            "Coord9": {"sup": 12, "inf": 38, "left": 23, "right": 25, "width": 48, "height": 50},
            "Coord10": {"sup": 22, "inf": 28, "left": 24, "right": 25, "width": 49, "height": 50},
            "Coord11": {"sup": 30, "inf": 15, "left": 25, "right": 25, "width": 50, "height": 45},
            "Coord12": {"sup": 1, "inf": 47, "left": 6, "right": 35, "width": 41, "height": 48},
            "Coord13": {"sup": 2, "inf": 47, "left": 31, "right": 9, "width": 40, "height": 49},
            "Coord14": {"sup": 36, "inf": 2, "left": 0, "right": 47, "width": 47, "height": 38},
            "Coord15": {"sup": 38, "inf": 0, "left": 44, "right": 0, "width": 44, "height": 38}
        }
        self.coordinates = {}
        self.search_coordinates = {}

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
                    
                    # Procesar las 15 coordenadas (2 números por coordenada)
                    for i in range(15):
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
                for i in range(1, 16):
                    coord_name = f"coord{i}"
                    if coord_name in data:
                        coordinates[coord_name] = data[coord_name]
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
