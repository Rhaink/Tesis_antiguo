"""
Punto de entrada principal del programa de etiquetado.
"""
import cv2
import numpy as np
from typing import List, Optional
from .config import VENTANA_NOMBRE, VENTANA_ANCHO, VENTANA_ALTO
from .gui_manager import GUIManager
from .event_handler import EventHandler
from .file_manager import FileManager
from .image_processor import ImageProcessor

class ImageAnnotator:
    """Clase principal del programa de etiquetado."""
    
    def __init__(self, image_paths: List[str], output_file: str):
        """
        Inicializa el programa de etiquetado.
        
        Args:
            image_paths: Lista de rutas de imágenes a procesar.
            output_file: Ruta del archivo de salida CSV.
        """
        self.image_paths = image_paths
        self.current_image_index = 0
        
        # Inicializar componentes
        self.gui = GUIManager()
        self.event_handler = EventHandler(self.gui)
        self.file_manager = FileManager(output_file)
        self.image_processor = ImageProcessor()
        
        # Configurar callback del mouse
        cv2.setMouseCallback(VENTANA_NOMBRE, self.gui.mouse_callback)
    
    def run(self) -> None:
        """Ejecuta el bucle principal del programa."""
        while self.current_image_index < len(self.image_paths):
            # Cargar y procesar imagen actual
            path = self.image_paths[self.current_image_index]
            print("Imagen actual:", path)
            print()
            
            # Establecer imagen actual en el manejador de archivos
            self.file_manager.set_current_image(self.current_image_index, path)
            
            # Procesar imagen
            imagen_original = cv2.imread(path)
            if imagen_original is None:
                print(f"Error al cargar la imagen: {path}")
                self.current_image_index += 1
                continue
            
            # Reiniciar anotación para nueva imagen
            self.gui = GUIManager()
            cv2.setMouseCallback(VENTANA_NOMBRE, self.gui.mouse_callback)
            self.event_handler = EventHandler(self.gui)
                
            # Procesar imagen para visualización
            imagen_procesada = self.image_processor.process_image(imagen_original)
            grosor_linea = self.image_processor.calculate_line_thickness(imagen_procesada)
            
            # Configurar GUI con la nueva imagen
            self.gui.set_image(imagen_procesada, grosor_linea)
            
            # Mostrar menú
            self.gui.show_menu()
            
            # Bucle de eventos para la imagen actual
            while True:
                self.gui.update_display()
                key = cv2.waitKey(1) & 0xFF
                
                if key == 255:  # No key pressed
                    continue
                    
                # Manejar evento de teclado
                result = self.event_handler.handle_keyboard_event(key)
                
                if result == 'break':
                    # Guardar anotación si es necesario
                    annotation = self.gui.get_annotation()
                    if annotation.are_all_points_defined():
                        self.file_manager.save_annotation(annotation)
                    self.current_image_index += 1
                    break
                elif result == 'exit':
                    print("Programa Terminado")
                    return
                    
        print("Programa Terminado")
        self.gui.close_windows()
