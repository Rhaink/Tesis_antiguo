"""
Módulo para el manejo de la interfaz gráfica del programa de etiquetado.
"""
from typing import Tuple, Optional, Callable
import cv2
import numpy as np
from .config import (
    VENTANA_NOMBRE,
    VENTANA_ANCHO,
    VENTANA_ALTO,
    COLOR_LINEA,
    COLOR_PUNTO,
    COLOR_LINEA_CENTRAL,
    MENSAJES,
    MENU_TEXTO
)
from .models import Point, Line, ImageAnnotation

class GUIManager:
    """Clase para el manejo de la interfaz gráfica."""
    
    def __init__(self):
        """Inicializa el gestor de interfaz gráfica."""
        self.window_name = VENTANA_NOMBRE
        self.image = None
        self.clon = None
        self.grosor_linea = 1
        self.radio_punto = 2
        self.annotation = ImageAnnotation()
        self._setup_window()
    
    def _setup_window(self) -> None:
        """Configura la ventana principal."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    
    def set_image(self, image: np.ndarray, grosor_linea: int) -> None:
        """
        Establece la imagen actual y sus propiedades.
        
        Args:
            image: Imagen a mostrar.
            grosor_linea: Grosor de línea calculado.
        """
        self.original_image = image.copy()
        self.grosor_linea = grosor_linea
        self.radio_punto = grosor_linea * 2
        
        # Configurar ventana y mostrar imagen
        cv2.resizeWindow(self.window_name, VENTANA_ANCHO, VENTANA_ALTO)
        self.image = self.original_image.copy()
        self.clon = self.image.copy()
        
        # Dibujar línea central
        height, width = self.image.shape[:2]
        cv2.line(self.image, (width//2, 0), (width//2, height), 
                COLOR_LINEA_CENTRAL, self.grosor_linea)
        cv2.imshow(self.window_name, self.image)
    
    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """
        Callback para eventos del mouse.
        
        Args:
            event: Tipo de evento del mouse.
            x: Coordenada X del mouse.
            y: Coordenada Y del mouse.
            flags: Flags del evento.
            param: Parámetros adicionales.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Primer click
            point = Point(x, y)
            self.annotation.set_point(0, point)
            self._draw_point(point)
            print('--------------')
            print(MENSAJES['PRIMER_CLICK'])
            print('coordenadas:', [x, y])
            
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Segundo click
            if self.annotation.get_point(0) is not None:
                point = Point(x, y)
                self.annotation.set_point(1, point)
                self._draw_point(point)
                self._draw_main_line()
                print('--------------')
                print(MENSAJES['SEGUNDO_CLICK'])
                print('coordenadas:', [x, y])
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Tercer click - Calcula todos los puntos
            if self.annotation.get_point(0) is not None and self.annotation.get_point(1) is not None:
                self.annotation.calculate_all_points(y)
                self._draw_all_elements()
                print('--------------')
                print(MENSAJES['TERCER_CLICK'])
    
    def _draw_point(self, point: Point) -> None:
        """Dibuja un punto en la imagen."""
        cv2.circle(self.image, point.to_tuple(), self.radio_punto, COLOR_PUNTO, -1)
    
    def _draw_line(self, start: Point, end: Point) -> None:
        """Dibuja una línea entre dos puntos."""
        cv2.line(self.image, start.to_tuple(), end.to_tuple(), COLOR_LINEA, self.grosor_linea)
    
    def _draw_main_line(self) -> None:
        """Dibuja la línea principal entre los primeros dos puntos."""
        p1 = self.annotation.get_point(0)
        p2 = self.annotation.get_point(1)
        if p1 is not None and p2 is not None:
            self._draw_line(p1, p2)
    
    def _calculate_perpendicular_points(self) -> None:
        """Calcula los puntos perpendiculares basados en la línea principal."""
        if self.annotation.main_line is None:
            return
            
        # Calcular puntos intermedios y perpendiculares
        self.annotation.calculate_perpendicular_lines()
        
        # Asignar puntos calculados
        if self.annotation.perpendicular_lines:
            for i, line in enumerate(self.annotation.perpendicular_lines):
                base_idx = (i * 2) + 2
                self.annotation.set_point(base_idx, line.start)
                self.annotation.set_point(base_idx + 1, line.end)
    
    def _draw_all_elements(self) -> None:
        """Dibuja todos los elementos en la imagen."""
        self.image = self.clon.copy()
        
        # Dibujar línea central usando dimensiones de la imagen
        height, width = self.image.shape[:2]
        cv2.line(self.image, (width//2, 0), (width//2, height), 
                COLOR_LINEA_CENTRAL, self.grosor_linea)
        
        # Dibujar línea principal
        if self.annotation.points[0] is not None and self.annotation.points[1] is not None:
            self._draw_line(self.annotation.points[0], self.annotation.points[1])
        
        # Dibujar líneas perpendiculares
        pairs = [(2,3), (4,5), (6,7), (11,12), (13,14)]  # Pares de puntos para líneas
        for start_idx, end_idx in pairs:
            if (start_idx < len(self.annotation.points) and 
                end_idx < len(self.annotation.points) and
                self.annotation.points[start_idx] is not None and 
                self.annotation.points[end_idx] is not None):
                self._draw_line(self.annotation.points[start_idx], 
                              self.annotation.points[end_idx])
        
        # Dibujar todos los puntos, incluyendo los puntos centrales
        for i, point in enumerate(self.annotation.points):
            if point is not None:
                self._draw_point(point)
                # Dibujar puntos centrales con un color diferente
                if i in [8, 9, 10]:  # puntos 9, 10, 11
                    cv2.circle(self.image, point.to_tuple(), 
                             self.radio_punto + 1, COLOR_LINEA_CENTRAL, 1)
    
    def show_preview(self) -> None:
        """Muestra una previsualización con los puntos numerados."""
        if not self.annotation.are_all_points_defined():
            print(MENSAJES['DEFINIR_COORDENADAS'])
            return
            
        preview = self.image.copy()
        for i, point in enumerate(self.annotation.points):
            if point is not None:
                cv2.circle(preview, point.to_tuple(), self.radio_punto, COLOR_PUNTO, -1)
                cv2.putText(preview, str(i+1), 
                          (point.x+5, point.y-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Previsualización", preview)
        cv2.waitKey(0)
        cv2.destroyWindow("Previsualización")
    
    def clear_image(self) -> None:
        """Limpia la imagen a su estado original."""
        self.image = self.clon.copy()
        height, width = self.image.shape[:2]
        cv2.line(self.image, (width//2, 0), (width//2, height), 
                COLOR_LINEA_CENTRAL, self.grosor_linea)
    
    def show_menu(self) -> None:
        """Muestra el menú de ayuda."""
        print(MENU_TEXTO)
    
    def update_display(self) -> None:
        """Actualiza la visualización de la ventana."""
        try:
            cv2.imshow(self.window_name, self.image)
        except:
            pass
    
    def close_windows(self) -> None:
        """Cierra todas las ventanas."""
        cv2.destroyAllWindows()
    
    def get_annotation(self) -> ImageAnnotation:
        """Retorna la anotación actual."""
        return self.annotation
