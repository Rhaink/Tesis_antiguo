"""
Componente de selección de archivos para PulmoAlign Viewer.
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Tuple, Optional, Callable
from pathlib import Path

from ..utils.file_utils import browse_file

class FileSelector:
    """Componente reutilizable para selección de archivos."""
    
    def __init__(self, 
                 parent: ttk.Frame,
                 label_text: str,
                 file_types: List[Tuple[str, str]],
                 initial_dir: str = ".",
                 initial_value: str = "",
                 on_change: Optional[Callable[[str], None]] = None):
        """
        Inicializa un nuevo selector de archivos.
        
        Args:
            parent: Widget padre donde se creará el selector
            label_text: Texto para la etiqueta del selector
            file_types: Lista de tuplas con tipos de archivo permitidos
            initial_dir: Directorio inicial para la búsqueda
            initial_value: Valor inicial para el campo de archivo
            on_change: Callback opcional para cuando cambia el archivo seleccionado
        """
        self.parent = parent
        self.file_types = file_types
        self.initial_dir = initial_dir
        self.on_change = on_change
        
        # Variables
        self.file_path = tk.StringVar(value=initial_value)
        self.file_path.trace_add('write', self._on_path_change)
        
        # Crear widgets
        self.setup_gui(label_text)
        
    def setup_gui(self, label_text: str) -> None:
        """
        Configura la interfaz gráfica del selector.
        
        Args:
            label_text: Texto para la etiqueta del selector
        """
        # Frame principal
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill='x', padx=5, pady=2)
        
        # Etiqueta
        ttk.Label(self.frame, text=label_text).pack(side='left', padx=5)
        
        # Campo de entrada
        self.entry = ttk.Entry(self.frame, textvariable=self.file_path)
        self.entry.pack(side='left', fill='x', expand=True, padx=5)
        
        # Botón examinar
        ttk.Button(
            self.frame,
            text="Examinar",
            command=self._browse
        ).pack(side='left', padx=5)
    
    def _browse(self) -> None:
        """Abre el diálogo de selección de archivo."""
        filename = browse_file(
            title=f"Seleccionar archivo",
            filetypes=self.file_types,
            initial_dir=self.initial_dir
        )
        if filename:
            self.file_path.set(filename)
    
    def _on_path_change(self, *args) -> None:
        """Maneja el cambio en la ruta del archivo."""
        if self.on_change:
            self.on_change(self.file_path.get())
    
    def get_path(self) -> str:
        """
        Obtiene la ruta del archivo seleccionado.
        
        Returns:
            str: Ruta del archivo
        """
        return self.file_path.get()
    
    def set_path(self, path: str) -> None:
        """
        Establece la ruta del archivo.
        
        Args:
            path: Nueva ruta del archivo
        """
        self.file_path.set(path)
    
    def is_valid(self) -> bool:
        """
        Verifica si la ruta actual es válida.
        
        Returns:
            bool: True si la ruta existe, False en caso contrario
        """
        return Path(self.get_path()).exists()
    
    def clear(self) -> None:
        """Limpia la selección actual."""
        self.file_path.set("")
