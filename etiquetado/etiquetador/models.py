"""
Clases base para el manejo de puntos y líneas en el programa de etiquetado.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import numpy as np

class Point:
    """Representa un punto en coordenadas 2D."""
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    
    def to_tuple(self) -> Tuple[int, int]:
        """Convierte el punto a una tupla de coordenadas."""
        return (self.x, self.y)
    
    def scale(self, factor: float) -> 'Point':
        """Escala las coordenadas del punto por un factor dado."""
        return Point(
            x=int(self.x * factor),
            y=int(self.y * factor)
        )
    
    def move_horizontal(self, dx: int, reference_point: 'Point', pendiente: float) -> None:
        """
        Mueve el punto horizontalmente y recalcula su coordenada Y usando la pendiente.
        
        Args:
            dx: Desplazamiento horizontal.
            reference_point: Punto de referencia para el cálculo.
            pendiente: Pendiente para recalcular Y.
        """
        self.x += dx
        if pendiente != float('inf'):
            self.y = int(pendiente * (self.x - reference_point.x) + reference_point.y)

class Line:
    """Representa una línea entre dos puntos."""
    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end
        self.pendiente = self._calculate_pendiente()
        self.pendiente_perpendicular = self._calculate_pendiente_perpendicular()
        
    def _calculate_pendiente(self) -> float:
        """Calcula la pendiente de la línea."""
        if self.start.x != self.end.x:
            return (self.start.y - self.end.y) / (self.start.x - self.end.x)
        return float('inf')
    
    def _calculate_pendiente_perpendicular(self) -> float:
        """Calcula la pendiente perpendicular a la línea."""
        if self.pendiente == 0:
            return float('inf')
        elif self.pendiente == float('inf'):
            return 0
        return -1 / self.pendiente
    
    def get_point_at_fraction(self, fraction: float) -> Point:
        """Obtiene un punto a una fracción específica de la línea."""
        x = int(self.start.x + (self.end.x - self.start.x) * fraction)
        y = int(self.start.y + (self.end.y - self.start.y) * fraction)
        return Point(x, y)
    
    def get_perpendicular_points(self, base_point: Point, distance: int) -> Tuple[Point, Point]:
        """
        Calcula dos puntos a una distancia perpendicular de un punto base.
        
        Args:
            base_point: Punto base sobre la línea.
            distance: Distancia perpendicular.
            
        Returns:
            Tupla de dos puntos perpendiculares.
        """
        if self.start.x != self.end.x:
            x1 = base_point.x - distance
            x2 = base_point.x + distance
            y1 = int(self.pendiente_perpendicular * (x1 - base_point.x) + base_point.y)
            y2 = int(self.pendiente_perpendicular * (x2 - base_point.x) + base_point.y)
        else:
            x1 = base_point.x - distance
            x2 = base_point.x + distance
            y1 = base_point.y
            y2 = base_point.y
        return Point(x1, y1), Point(x2, y2)

class ImageAnnotation:
    """Maneja las anotaciones de puntos y líneas en la imagen."""
    def __init__(self):
        self.points: List[Optional[Point]] = [None] * 15
        self.main_line: Optional[Line] = None
        self.perpendicular_lines: List[Optional[Line]] = []
        self.intermediate_points = {}
    
    def set_point(self, index: int, point: Point) -> None:
        """Establece un punto en el índice especificado."""
        if 0 <= index < len(self.points):
            self.points[index] = point
    
    def get_point(self, index: int) -> Optional[Point]:
        """Obtiene el punto en el índice especificado."""
        if 0 <= index < len(self.points):
            return self.points[index]
        return None
    
    def move_point(self, index: int, dx: int) -> None:
        """
        Mueve un punto horizontalmente y recalcula su posición.
        Sigue la lógica exacta del programa original.
        
        Args:
            index: Índice del punto a mover.
            dx: Desplazamiento horizontal.
        """
        if not (0 <= index < len(self.points)) or self.points[index] is None:
            return
            
        point = self.points[index]
        
        # Determinar punto de referencia y pendiente para el cálculo
        if 2 <= index <= 3:  # Puntos 3 y 4
            ref_point = self.intermediate_points['cuarto1']
            pendiente = self.main_line.pendiente_perpendicular
        elif 4 <= index <= 5:  # Puntos 5 y 6
            ref_point = self.intermediate_points['cuarto2']
            pendiente = self.main_line.pendiente_perpendicular
        elif 6 <= index <= 7:  # Puntos 7 y 8
            ref_point = self.intermediate_points['cuarto3']
            pendiente = self.main_line.pendiente_perpendicular
        elif 11 <= index <= 12:  # Puntos 12 y 13
            ref_point = self.points[0]  # inicio
            pendiente = self.main_line.pendiente_perpendicular
        elif 13 <= index <= 14:  # Puntos 14 y 15
            ref_point = self.points[1]  # final
            pendiente = self.main_line.pendiente_perpendicular
        else:
            return
            
        point.move_horizontal(dx, ref_point, pendiente)
        self.points[index] = point
    
    def are_all_points_defined(self) -> bool:
        """Verifica si todos los puntos están definidos."""
        return all(point is not None for point in self.points)
    
    def calculate_main_line(self) -> None:
        """Calcula la línea principal entre los primeros dos puntos."""
        if self.points[0] is not None and self.points[1] is not None:
            self.main_line = Line(self.points[0], self.points[1])
    
    def calculate_all_points(self, y: int) -> None:
        """
        Calcula todos los puntos basados en los puntos iniciales.
        """
        if self.points[0] is None or self.points[1] is None:
            return
            
        inicio = self.points[0]
        final = self.points[1]
        
        # Calcular puntos intermedios
        if inicio.x != final.x:
            # Calcular pendiente y pendiente perpendicular
            self.calculate_main_line()
            pendiente = self.main_line.pendiente
            pendiente_perpendicular = self.main_line.pendiente_perpendicular
            
            # Puntos intermedios
            medio = Point(
                int((inicio.x + final.x)/2),
                int((inicio.y + final.y)/2)
            )
            
            # Puntos de cuartos
            cuarto1 = Point(
                int(inicio.x + (final.x - inicio.x) / 4),
                int(inicio.y + (final.y - inicio.y) / 4)
            )
            cuarto2 = Point(
                int(inicio.x + 2 * (final.x - inicio.x) / 4),
                int(inicio.y + 2 * (final.y - inicio.y) / 4)
            )
            cuarto3 = Point(
                int(inicio.x + 3 * (final.x - inicio.x) / 4),
                int(inicio.y + 3 * (final.y - inicio.y) / 4)
            )
            
            # Guardar puntos intermedios
            self.intermediate_points = {
                'medio': medio,
                'cuarto1': cuarto1,
                'cuarto2': cuarto2,
                'cuarto3': cuarto3
            }
            
            # Calcular líneas perpendiculares
            def create_perpendicular_line(base_point: Point, distance: int) -> Tuple[Point, Point]:
                if pendiente == float('inf'):
                    return Point(base_point.x - distance, base_point.y), Point(base_point.x + distance, base_point.y)
                elif pendiente == 0:
                    return Point(base_point.x, base_point.y - distance), Point(base_point.x, base_point.y + distance)
                else:
                    x1 = base_point.x - distance
                    x2 = base_point.x + distance
                    y1 = int(pendiente_perpendicular * (x1 - base_point.x) + base_point.y)
                    y2 = int(pendiente_perpendicular * (x2 - base_point.x) + base_point.y)
                    return Point(x1, y1), Point(x2, y2)
            
            # Crear líneas perpendiculares
            p3, p4 = create_perpendicular_line(cuarto1, 100)
            p5, p6 = create_perpendicular_line(cuarto2, 100)
            p7, p8 = create_perpendicular_line(cuarto3, 100)
            p12, p13 = create_perpendicular_line(inicio, 80)
            p14, p15 = create_perpendicular_line(final, 100)
            
            # Asignar puntos
            self.points[2] = p3
            self.points[3] = p4
            self.points[4] = p5
            self.points[5] = p6
            self.points[6] = p7
            self.points[7] = p8
            self.points[8] = cuarto1  # punto central en cuarto1
            self.points[9] = cuarto2  # punto central en cuarto2
            self.points[10] = cuarto3  # punto central en cuarto3
            self.points[11] = p12
            self.points[12] = p13
            self.points[13] = p14
            self.points[14] = p15
            
        else:
            # Caso especial cuando la línea es vertical
            # Calcular puntos intermedios
            cuarto1 = Point(inicio.x, int(inicio.y + (final.y - inicio.y) / 4))
            cuarto2 = Point(inicio.x, int(inicio.y + 2 * (final.y - inicio.y) / 4))
            cuarto3 = Point(inicio.x, int(inicio.y + 3 * (final.y - inicio.y) / 4))
            
            # Guardar puntos intermedios
            self.intermediate_points = {
                'medio': Point(inicio.x, int((inicio.y + final.y)/2)),
                'cuarto1': cuarto1,
                'cuarto2': cuarto2,
                'cuarto3': cuarto3
            }
            
            # Crear línea principal vertical
            self.main_line = Line(inicio, final)
            
            # Asignar puntos
            self.points[2] = Point(16, int(cuarto1.y))
            self.points[3] = Point(48, int(cuarto1.y))
            self.points[4] = Point(0, int(cuarto2.y))
            self.points[5] = Point(64, int(cuarto2.y))
            self.points[6] = Point(0, int(cuarto3.y))
            self.points[7] = Point(64, int(cuarto3.y))
            self.points[8] = Point(0, int(cuarto1.y))
            self.points[9] = Point(64, int(cuarto2.y))
            self.points[10] = Point(0, int(cuarto3.y))
            self.points[11] = Point(64, int(inicio.y))
            self.points[12] = Point(0, int(inicio.y))
            self.points[13] = Point(64, int(final.y))
            self.points[14] = Point(64, int(final.y))
