"""
Módulo para el manejo y procesamiento de coordenadas en imágenes pulmonares.

Este módulo proporciona clases y funciones para:
- Lectura de coordenadas desde archivos CSV y JSON
- Gestión de coordenadas de puntos de referencia (sistema 0-based)
- Procesamiento de coordenadas de búsqueda
- Validación de coordenadas en rango 0-63
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

    def _calculate_region_bounds(self, coord_points):
        """
        Calcula los límites de una región basado en las coordenadas de búsqueda.
        Todas las coordenadas se manejan en sistema 0-based (0-63).
        
        Args:
            coord_points (List[List[int]]): Lista de coordenadas [y,x] en formato 0-based
            
        Returns:
            Dict: Diccionario con los límites calculados (sup, inf, left, right, width, height)
            
        Raises:
            ValueError: Si alguna coordenada está fuera del rango 0-63
        """
        if not coord_points:
            return None
            
        # Extraer y,x de los puntos (cambiado orden para consistencia)
        y_coords = [p[0] for p in coord_points]
        x_coords = [p[1] for p in coord_points]
        
        # Validar rango 0-63
        if any(y < 0 or y > 63 for y in y_coords) or any(x < 0 or x > 63 for x in x_coords):
            raise ValueError("Coordenadas fuera del rango válido 0-63")
        
        # Calcular límites
        sup = min(y_coords)    # y mínima (0-based)
        inf = max(y_coords)    # y máxima (0-based)
        left = min(x_coords)   # x mínima (0-based)
        right = max(x_coords)  # x máxima (0-based)
        
        # Calcular dimensiones (incluyendo puntos extremos)
        width = right - left + 1   # Número total de puntos en X
        height = inf - sup + 1     # Número total de puntos en Y
        
        return {
            "sup": sup,     # y mínima (0-based)
            "inf": inf,     # y máxima (0-based)
            "left": left,   # x mínima (0-based)
            "right": right, # x máxima (0-based)
            "width": width, # Número total de puntos en X
            "height": height # Número total de puntos en Y
        }

    def read_coordinates(self, filename: str) -> None:
        try:
            coordinates = {}
            with open(filename, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    try:
                        index = int(row[0]) # Asumimos que el índice sí es entero
                    except ValueError:
                        print(f"Advertencia: Omitiendo fila con índice no entero: {row[0]}")
                        continue # Saltar esta fila si el índice no es válido

                    coordinates[index] = {}
                    for i in range(15):
                        x_pos = 1 + (i * 2)
                        y_pos = 2 + (i * 2)

                        if x_pos < len(row) and y_pos < len(row):
                            coord_name = f"Coord{i+1}"
                            try:
                                # LEER COMO FLOAT
                                x_str = row[x_pos]
                                y_str = row[y_pos]
                                x = float(x_str) if x_str else 0.0
                                y = float(y_str) if y_str else 0.0

                                # Guardar como float
                                coordinates[index][coord_name] = (x, y)

                                # QUITAR LA VALIDACIÓN AQUÍ (o modificar _validate_coordinates)
                                # self._validate_coordinates(int(x), int(y), f"en índice {index}") # Comentado/Eliminado

                            except ValueError:
                                print(f"Advertencia: Valor no numérico para índice {index}, coord {i+1}. Fila: {row}")
                                coordinates[index][coord_name] = (0.0, 0.0) # Valor por defecto o manejar error
                        else:
                            # Mantener advertencia si faltan columnas
                            print(f"Advertencia: Datos faltantes para índice {index}, coordenada {i+1}")
                            coordinates[index][f"Coord{i+1}"] = (0.0, 0.0)

            self.coordinates = coordinates
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo de coordenadas: {filename}")
        except Exception as e:
            # Captura genérica para otros posibles errores de lectura/procesamiento
            raise ValueError(f"Error al leer o procesar el archivo de coordenadas '{filename}': {str(e)}")

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
                
                # Procesar coord1 hasta coord15
                for i in range(1, 16):
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
        El punto de intersección se define como la esquina superior izquierda (left, sup)
        de la región delimitada por los límites proporcionados.
        
        Args:
            sup (int): Límite superior (y mínima, 0-based)
            inf (int): Límite inferior (y máxima, 0-based)
            left (int): Límite izquierdo (x mínima, 0-based)
            right (int): Límite derecho (x máxima, 0-based)
            
        Returns:
            Tuple[int, int]: Coordenadas del punto de intersección (x, y) en formato 0-based
            
        Raises:
            ValueError: Si las coordenadas están fuera del rango 0-63 o los límites son inválidos
        """
        # Validar rango 0-63
        if not all(0 <= val <= 63 for val in [sup, inf, left, right]):
            raise ValueError("Límites fuera del rango válido 0-63")
            
        # Validar que los límites son coherentes
        if sup > inf:
            raise ValueError("El límite superior no puede ser mayor que el inferior")
        if left > right:
            raise ValueError("El límite izquierdo no puede ser mayor que el derecho")
            
        return left, sup  # Retorna coordenadas (x, y) en formato 0-based
