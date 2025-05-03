"""
Paquete para el programa de etiquetado de imágenes médicas.
"""
from .main import ImageAnnotator
from .models import Point, Line, ImageAnnotation
from .gui_manager import GUIManager
from .event_handler import EventHandler
from .file_manager import FileManager
from .image_processor import ImageProcessor

__all__ = [
    'ImageAnnotator',
    'Point',
    'Line',
    'ImageAnnotation',
    'GUIManager',
    'EventHandler',
    'FileManager',
    'ImageProcessor'
]
