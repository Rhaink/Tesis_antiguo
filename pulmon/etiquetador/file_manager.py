"""
Manejo de archivos para el programa de etiquetado.
"""
import csv
from typing import List
from .models import ImageAnnotation

class FileManager:
    """Clase para el manejo de archivos."""
    
    def __init__(self, output_file: str):
        """
        Inicializa el manejador de archivos.
        
        Args:
            output_file: Ruta del archivo CSV de salida.
        """
        self.output_file = output_file
        self.current_index = 0
        self.current_image = ""
    
    def set_current_image(self, index: int, image_path: str) -> None:
        """
        Establece la imagen actual.
        
        Args:
            index: Índice de la imagen.
            image_path: Ruta de la imagen.
        """
        self.current_index = index
        self.current_image = image_path.split('/')[-1].split('.')[0]
    
    def save_annotation(self, annotation: ImageAnnotation) -> None:
        """
        Guarda una anotación en el archivo CSV.
        
        Args:
            annotation: Anotación a guardar.
        """
        if not annotation.are_all_points_defined():
            return
            
        # Escalar coordenadas a resolución 64x64
        height = max(point.y for point in annotation.points if point is not None)
        scale_factor = 64 / height
        scaled_points = []
        for point in annotation.points:
            scaled_x = int(point.x * scale_factor)
            scaled_y = int(point.y * scale_factor)
            scaled_points.extend([scaled_x, scaled_y])
            
        # Escribir al archivo CSV
        with open(self.output_file, 'a', newline='') as archivo_csv:
            writer = csv.writer(archivo_csv, delimiter=',')
            row = [self.current_index] + scaled_points + [self.current_image]
            writer.writerow(row)
