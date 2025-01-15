"""
Utilidades para el manejo de archivos en PulmoAlign Viewer.
"""

from pathlib import Path
from typing import List, Tuple
from tkinter import filedialog

def browse_file(title: str, filetypes: List[Tuple[str, str]], initial_dir: str = ".") -> str:
    """
    Abre un diálogo para seleccionar un archivo.
    
    Args:
        title: Título de la ventana de diálogo
        filetypes: Lista de tuplas con tipos de archivo permitidos
        initial_dir: Directorio inicial para la búsqueda
        
    Returns:
        str: Ruta del archivo seleccionado o cadena vacía si se cancela
    """
    filename = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes + [("All files", "*.*")],
        initialdir=initial_dir
    )
    return filename

def ensure_directory(path: Path) -> None:
    """
    Asegura que un directorio exista, creándolo si es necesario.
    
    Args:
        path: Ruta del directorio a verificar/crear
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def get_project_root() -> Path:
    """
    Obtiene la ruta raíz del proyecto.
    
    Returns:
        Path: Ruta raíz del proyecto
    """
    return Path(__file__).parent.parent

def validate_paths(*paths: Path) -> None:
    """
    Valida que existan las rutas especificadas.
    
    Args:
        *paths: Rutas a validar
        
    Raises:
        FileNotFoundError: Si alguna ruta no existe
    """
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"No se encontró la ruta: {path}")

def get_file_extension(filename: str) -> str:
    """
    Obtiene la extensión de un archivo.
    
    Args:
        filename: Nombre del archivo
        
    Returns:
        str: Extensión del archivo en minúsculas
    """
    return Path(filename).suffix.lower()

def is_image_file(filename: str) -> bool:
    """
    Verifica si un archivo es una imagen basándose en su extensión.
    
    Args:
        filename: Nombre del archivo
        
    Returns:
        bool: True si es un archivo de imagen, False en caso contrario
    """
    valid_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
    return get_file_extension(filename) in valid_extensions
