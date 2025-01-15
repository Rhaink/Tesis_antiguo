"""
Componente de visualización de imágenes para PulmoAlign Viewer.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from typing import List, Dict, Optional, Callable
from pathlib import Path
import gc

from pulmo_align.utils.image_utils import load_and_resize_image

class ImageViewer:
    """Componente reutilizable para visualización de imágenes."""
    
    def __init__(self, parent: ttk.Frame, display_width: int = 900):
        """
        Inicializa un nuevo visualizador de imágenes.
        
        Args:
            parent: Widget padre donde se creará el visualizador
            display_width: Ancho de visualización para las imágenes
        """
        self.parent = parent
        self.display_width = display_width
        
        # Variables de estado
        self.current_images: List[Dict] = []
        self.current_index: int = 0
        self.photo_references: List[ImageTk.PhotoImage] = []
        
        # Crear widgets
        self.setup_gui()
        
    def setup_gui(self) -> None:
        """Configura la interfaz gráfica del visualizador."""
        # Frame principal
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Frame para la imagen actual
        self.image_frame = ttk.Frame(self.frame)
        self.image_frame.pack(fill='both', expand=True)
        
        # Frame para navegación
        self.nav_frame = ttk.Frame(self.frame)
        self.nav_frame.pack(fill='x', padx=5, pady=5)
        
        # Botones de navegación
        ttk.Button(self.nav_frame, text="←", command=self.prev_image).pack(side='left', padx=5)
        self.counter_label = ttk.Label(self.nav_frame, text="0/0")
        self.counter_label.pack(side='left', padx=5)
        ttk.Button(self.nav_frame, text="→", command=self.next_image).pack(side='left', padx=5)
    
    def load_images(self, images: List[Dict]) -> None:
        """
        Carga una lista de imágenes para visualización.
        
        Args:
            images: Lista de diccionarios con 'path' y 'title' de cada imagen
        """
        self.current_images = images
        self.current_index = 0
        self.photo_references.clear()
        gc.collect()
        
        if images:
            self.show_current_image()
        else:
            self.show_no_images_message()
        
        self.update_counter()
    
    def show_current_image(self) -> None:
        """Muestra la imagen actual."""
        # Limpiar frame
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        if not self.current_images:
            self.show_no_images_message()
            return
        
        # Obtener imagen actual
        current = self.current_images[self.current_index]
        
        try:
            # Asegurarse de que la ruta sea absoluta
            image_path = Path(current['path']).resolve()
            
            # Verificar que el archivo existe
            if not image_path.exists():
                raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
            
            # Cargar y redimensionar imagen
            img = Image.open(str(image_path))
            ratio = self.display_width / img.width
            display_height = int(img.height * ratio)
            
            img = img.resize((self.display_width, display_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Mantener referencia
            self.photo_references = [photo]
            
            # Mostrar título
            ttk.Label(self.image_frame, text=current['title']).pack()
            
            # Mostrar imagen
            label = ttk.Label(self.image_frame, image=photo)
            label.image = photo
            label.pack(padx=5, pady=5)
            
        except Exception as e:
            ttk.Label(self.image_frame, text=f"Error al cargar imagen: {str(e)}").pack()
        
        self.update_counter()
    
    def show_no_images_message(self) -> None:
        """Muestra un mensaje cuando no hay imágenes disponibles."""
        ttk.Label(self.image_frame, text="No hay imágenes disponibles").pack()
    
    def prev_image(self) -> None:
        """Navega a la imagen anterior."""
        if self.current_images:
            self.current_index = (self.current_index - 1) % len(self.current_images)
            self.show_current_image()
    
    def next_image(self) -> None:
        """Navega a la siguiente imagen."""
        if self.current_images:
            self.current_index = (self.current_index + 1) % len(self.current_images)
            self.show_current_image()
    
    def update_counter(self) -> None:
        """Actualiza el contador de imágenes."""
        if self.current_images:
            self.counter_label.config(
                text=f"Imagen {self.current_index + 1} de {len(self.current_images)}"
            )
        else:
            self.counter_label.config(text="0/0")
    
    def clear(self) -> None:
        """Limpia el visualizador."""
        self.current_images = []
        self.current_index = 0
        self.photo_references.clear()
        gc.collect()
        
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        self.show_no_images_message()
        self.update_counter()
    
    def get_current_image(self) -> Optional[Dict]:
        """
        Obtiene la información de la imagen actual.
        
        Returns:
            Optional[Dict]: Diccionario con información de la imagen actual o None
        """
        if self.current_images and 0 <= self.current_index < len(self.current_images):
            return self.current_images[self.current_index]
        return None
