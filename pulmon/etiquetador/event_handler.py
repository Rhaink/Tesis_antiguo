"""
Manejo de eventos del programa de etiquetado.
"""
from typing import Optional
import cv2
from .config import TECLAS, MENSAJES
from .models import Point
from .gui_manager import GUIManager

class EventHandler:
    """Clase para el manejo de eventos del programa."""
    
    def __init__(self, gui_manager: GUIManager):
        """
        Inicializa el manejador de eventos.
        
        Args:
            gui_manager: Instancia del manejador de interfaz gráfica.
        """
        self.gui = gui_manager
        self.annotation = gui_manager.get_annotation()
        
    def handle_keyboard_event(self, key: int) -> str:
        """
        Maneja los eventos de teclado.
        
        Args:
            key: Código de la tecla presionada.
            
        Returns:
            Comando a ejecutar ('continue', 'break', 'exit')
        """
        key_char = chr(key & 0xFF)
        
        # Movimientos de puntos
        if key_char in [TECLAS['MOVER_IZQ_3'], TECLAS['MOVER_DER_3']]:
            self._mover_punto(2, -1 if key_char == TECLAS['MOVER_IZQ_3'] else 1)
        elif key_char in [TECLAS['MOVER_IZQ_4'], TECLAS['MOVER_DER_4']]:
            self._mover_punto(3, -1 if key_char == TECLAS['MOVER_IZQ_4'] else 1)
        elif key_char in [TECLAS['MOVER_IZQ_5'], TECLAS['MOVER_DER_5']]:
            self._mover_punto(4, -1 if key_char == TECLAS['MOVER_IZQ_5'] else 1)
        elif key_char in [TECLAS['MOVER_IZQ_6'], TECLAS['MOVER_DER_6']]:
            self._mover_punto(5, -1 if key_char == TECLAS['MOVER_IZQ_6'] else 1)
        elif key_char in [TECLAS['MOVER_IZQ_7'], TECLAS['MOVER_DER_7']]:
            self._mover_punto(6, -1 if key_char == TECLAS['MOVER_IZQ_7'] else 1)
        elif key_char in [TECLAS['MOVER_IZQ_8'], TECLAS['MOVER_DER_8']]:
            self._mover_punto(7, -1 if key_char == TECLAS['MOVER_IZQ_8'] else 1)
        elif key_char in [TECLAS['MOVER_IZQ_12'], TECLAS['MOVER_DER_12']]:
            self._mover_punto(11, -1 if key_char == TECLAS['MOVER_IZQ_12'] else 1)
        elif key_char in [TECLAS['MOVER_IZQ_13'], TECLAS['MOVER_DER_13']]:
            self._mover_punto(12, -1 if key_char == TECLAS['MOVER_IZQ_13'] else 1)
        elif key_char in [TECLAS['MOVER_IZQ_14'], TECLAS['MOVER_DER_14']]:
            self._mover_punto(13, -1 if key_char == TECLAS['MOVER_IZQ_14'] else 1)
        elif key_char in [TECLAS['MOVER_IZQ_15'], TECLAS['MOVER_DER_15']]:
            self._mover_punto(14, -1 if key_char == TECLAS['MOVER_IZQ_15'] else 1)
            
        # Acciones especiales
        elif key_char == TECLAS['PREVISUALIZAR']:
            self.gui.show_preview()
        elif key_char == TECLAS['LIMPIAR']:
            self.gui.clear_image()
        elif key_char == TECLAS['GUARDAR']:
            print('--------------')
            print(MENSAJES['REGISTRO_GUARDADO'])
            return 'break'
        elif key_char == TECLAS['SIGUIENTE']:
            return 'break'
        elif key_char == TECLAS['TERMINAR']:
            self.gui.close_windows()
            return 'exit'
            
        return 'continue'
    
    def _mover_punto(self, indice: int, direccion: int) -> None:
        """
        Mueve un punto en una dirección específica.
        
        Args:
            indice: Índice del punto a mover.
            direccion: Dirección del movimiento (-1: izquierda, 1: derecha).
        """
        self.annotation.move_point(indice, direccion)
        self.gui._draw_all_elements()
        self.gui.update_display()
