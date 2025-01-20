"""
Manejo de archivos para el programa de etiquetado.
"""
import csv
from typing import List
from .models import ImageAnnotation, Point

class FileManager:
    """Clase para el manejo de archivos."""
    
    def __init__(self, base_output_file: str):
        """
        Inicializa el manejador de archivos.
        
        Args:
            base_output_file: Ruta base del archivo CSV de salida (sin extensión).
        """
        self.base_output_file = base_output_file
        self.current_index = 0
        self.current_image = ""
        self.display_resolution = 640  # Resolución de visualización
        self.output_resolutions = [64, 128, 256]  # Resoluciones de salida
    
    def set_current_image(self, index: int, image_path: str) -> None:
        """
        Establece la imagen actual.
        
        Args:
            index: Índice de la imagen.
            image_path: Ruta de la imagen.
        """
        self.current_index = index
        self.current_image = image_path.split('/')[-1].split('.')[0]
    
    def _scale_to_resolution(self, points: List[Point], target_resolution: int) -> List[int]:
        """
        Escala las coordenadas desde la resolución de visualización a la resolución objetivo
        usando un método de alta precisión que preserva las relaciones espaciales.
        
        Args:
            points: Lista de puntos a escalar.
            target_resolution: Resolución objetivo.
            
        Returns:
            Lista de coordenadas escaladas.
        """
        # Calcular factores de escala con mayor precisión
        scale_factor = float(target_resolution) / float(self.display_resolution)
        
        # Encontrar el centro de masa de los puntos para mejor preservación de relaciones
        sum_x = sum(point.x for point in points if point is not None)
        sum_y = sum(point.y for point in points if point is not None)
        num_points = len([p for p in points if p is not None])
        center_x = sum_x / num_points
        center_y = sum_y / num_points
        
        scaled_points = []
        for point in points:
            # Calcular distancia desde el centro
            dx = point.x - center_x
            dy = point.y - center_y
            
            # Escalar coordenadas relativas al centro
            scaled_dx = dx * scale_factor
            scaled_dy = dy * scale_factor
            
            # Calcular nuevas coordenadas absolutas
            new_x = (center_x * scale_factor) + scaled_dx
            new_y = (center_y * scale_factor) + scaled_dy
            
            # Redondear y limitar al rango válido con alta precisión
            scaled_x = int(round(new_x))
            scaled_y = int(round(new_y))
            
            # Asegurar que las coordenadas estén dentro del rango
            scaled_x = max(0, min(scaled_x, target_resolution - 1))
            scaled_y = max(0, min(scaled_y, target_resolution - 1))
            
            scaled_points.extend([scaled_x, scaled_y])
        
        return scaled_points

    def _get_output_filename(self, resolution: int) -> str:
        """
        Genera el nombre de archivo para una resolución específica.
        
        Args:
            resolution: Resolución objetivo.
            
        Returns:
            Ruta completa del archivo.
        """
        base_name = self.base_output_file.rsplit('.', 1)[0]
        return f"{base_name}_{resolution}x{resolution}.csv"

    def save_annotation(self, annotation: ImageAnnotation) -> None:
        """
        Guarda una anotación en múltiples resoluciones.
        
        Args:
            annotation: Anotación a guardar.
        """
        if not annotation.are_all_points_defined():
            return
            
        # Guardar en cada resolución objetivo
        for resolution in self.output_resolutions:
            output_file = self._get_output_filename(resolution)
            scaled_points = self._scale_to_resolution(annotation.points, resolution)
            
            # Escribir al archivo CSV
            with open(output_file, 'a', newline='') as archivo_csv:
                writer = csv.writer(archivo_csv, delimiter=',')
                row = [self.current_index] + scaled_points + [self.current_image]
                writer.writerow(row)
