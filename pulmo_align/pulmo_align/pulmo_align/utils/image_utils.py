"""
Utilidades para el manejo de imágenes en PulmoAlign Viewer.
"""

from PIL import Image, ImageTk
from pathlib import Path
from typing import Tuple, Optional, Dict
import gc

def load_and_resize_image(image_path: str, display_width: int = 900) -> Tuple[ImageTk.PhotoImage, int, int]:
    """
    Carga una imagen y la redimensiona manteniendo su proporción.
    
    Args:
        image_path: Ruta a la imagen
        display_width: Ancho deseado para la visualización
        
    Returns:
        Tuple[ImageTk.PhotoImage, int, int]: La imagen procesada, ancho y alto finales
    """
    img = Image.open(image_path)
    ratio = display_width / img.width
    display_height = int(img.height * ratio)
    
    img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    
    return photo, display_width, display_height

def collect_images_from_directory(base_dir: Path, pattern: str = "*.png") -> list:
    """
    Recolecta todas las imágenes de un directorio que coincidan con el patrón.
    
    Args:
        base_dir: Directorio base donde buscar
        pattern: Patrón de búsqueda (por defecto: "*.png")
        
    Returns:
        list: Lista de diccionarios con información de las imágenes encontradas
    """
    images = []
    
    if base_dir.exists():
        for img_path in base_dir.glob(pattern):
            images.append({
                'path': str(img_path),
                'title': img_path.name
            })
    
    return sorted(images, key=lambda x: x['title'])

def clean_directory(directory: Path, pattern: str = "*.png") -> None:
    """
    Limpia un directorio eliminando los archivos que coincidan con el patrón.
    
    Args:
        directory: Directorio a limpiar
        pattern: Patrón de archivos a eliminar (por defecto: "*.png")
    """
    if directory.exists():
        for file in directory.glob(pattern):
            try:
                file.unlink()
            except:
                pass
        gc.collect()

def collect_processed_images(base_dir: Path) -> list:
    """
    Recolecta imágenes procesadas organizadas por coordenada.
    
    Args:
        base_dir: Directorio base donde buscar
        
    Returns:
        list: Lista de diccionarios con información de las imágenes encontradas
    """
    images = []
    
    if base_dir.exists():
        # Buscar en cada directorio de coordenadas
        for coord_dir in base_dir.glob("cropped_images_Coord*"):
            coord_name = coord_dir.name.replace("cropped_images_", "")
            
            # Recolectar todas las imágenes en este directorio
            for img_path in coord_dir.glob("*.png"):
                images.append({
                    'path': str(img_path),
                    'title': f"{coord_name} - {img_path.name}"
                })
    
    # Ordenar las imágenes por nombre de coordenada y nombre de archivo
    return sorted(images, key=lambda x: (x['title'].split(' - ')[0], x['title'].split(' - ')[1]))
