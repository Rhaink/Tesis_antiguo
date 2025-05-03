"""
Utilidades para el manejo de archivos.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

def ensure_directory(directory: str) -> None:
    """
    Asegura que un directorio exista, creándolo si es necesario.
    
    Args:
        directory: Ruta del directorio
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

def list_files(directory: str,
               pattern: str = "*",
               recursive: bool = False) -> List[str]:
    """
    Lista archivos en un directorio.
    
    Args:
        directory: Ruta del directorio
        pattern: Patrón de búsqueda (glob)
        recursive: Si se debe buscar recursivamente
        
    Returns:
        Lista de rutas de archivos
        
    Raises:
        FileNotFoundError: Si no existe el directorio
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"No se encontró el directorio: {directory}")
        
    if recursive:
        return [str(f) for f in dir_path.rglob(pattern) if f.is_file()]
    else:
        return [str(f) for f in dir_path.glob(pattern) if f.is_file()]

def get_file_info(file_path: str) -> Tuple[str, str, str]:
    """
    Obtiene información de un archivo.
    
    Args:
        file_path: Ruta del archivo
        
    Returns:
        Tupla con (nombre, extensión, directorio)
        
    Raises:
        FileNotFoundError: Si no existe el archivo
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
        
    return (path.stem, path.suffix, str(path.parent))

def is_image_file(file_path: str) -> bool:
    """
    Verifica si un archivo es una imagen.
    
    Args:
        file_path: Ruta del archivo
        
    Returns:
        True si es una imagen, False en caso contrario
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
    return Path(file_path).suffix.lower() in image_extensions

def get_unique_filename(directory: str, base_name: str, extension: str) -> str:
    """
    Genera un nombre de archivo único en un directorio.
    
    Args:
        directory: Directorio destino
        base_name: Nombre base del archivo
        extension: Extensión del archivo
        
    Returns:
        Nombre de archivo único
    """
    counter = 1
    dir_path = Path(directory)
    
    # Asegurar que la extensión comience con punto
    if not extension.startswith('.'):
        extension = f".{extension}"
    
    file_path = dir_path / f"{base_name}{extension}"
    while file_path.exists():
        file_path = dir_path / f"{base_name}_{counter}{extension}"
        counter += 1
    
    return str(file_path)

def safe_delete_file(file_path: str) -> bool:
    """
    Elimina un archivo de forma segura.
    
    Args:
        file_path: Ruta del archivo
        
    Returns:
        True si se eliminó correctamente, False en caso contrario
    """
    try:
        path = Path(file_path)
        if path.exists() and path.is_file():
            path.unlink()
            return True
    except Exception:
        pass
    return False

def get_file_size(file_path: str) -> Optional[int]:
    """
    Obtiene el tamaño de un archivo en bytes.
    
    Args:
        file_path: Ruta del archivo
        
    Returns:
        Tamaño en bytes o None si no existe
    """
    try:
        return Path(file_path).stat().st_size
    except Exception:
        return None

def get_latest_file(directory: str, pattern: str = "*") -> Optional[str]:
    """
    Obtiene el archivo más reciente en un directorio.
    
    Args:
        directory: Ruta del directorio
        pattern: Patrón de búsqueda (glob)
        
    Returns:
        Ruta del archivo más reciente o None si no hay archivos
    """
    try:
        files = list_files(directory, pattern)
        if not files:
            return None
            
        return max(files, key=lambda x: Path(x).stat().st_mtime)
    except Exception:
        return None

def copy_file_safe(source: str, destination: str, overwrite: bool = False) -> bool:
    """
    Copia un archivo de forma segura.
    
    Args:
        source: Ruta del archivo origen
        destination: Ruta del archivo destino
        overwrite: Si se debe sobrescribir si existe
        
    Returns:
        True si se copió correctamente, False en caso contrario
    """
    try:
        src_path = Path(source)
        dst_path = Path(destination)
        
        if not src_path.exists():
            return False
            
        if dst_path.exists() and not overwrite:
            return False
            
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_bytes(src_path.read_bytes())
        return True
        
    except Exception:
        return False
