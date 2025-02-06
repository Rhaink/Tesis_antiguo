"""
Clase base abstracta para las pestañas de PulmoAlign Viewer.
"""

import tkinter as tk
from tkinter import ttk
from abc import ABC, abstractmethod

class TabBase(ABC):
    """Clase base abstracta para todas las pestañas de la aplicación."""
    
    def __init__(self, parent):
        """
        Inicializa una nueva pestaña.
        
        Args:
            parent: El widget padre (normalmente un ttk.Notebook)
        """
        self.parent = parent
        self.frame = ttk.Frame(parent)
        
        # Variables comunes
        self.current_images = []
        self.current_image_index = 0
        self.photo_references = []
        
        # Configuración inicial
        self.setup_gui()
        # Mover initialize_components al final para asegurar que la GUI esté lista
        self.frame.after(100, self.initialize_components)
    
    @abstractmethod
    def setup_gui(self):
        """
        Configura la interfaz gráfica de la pestaña.
        Debe ser implementado por las clases hijas.
        """
        pass
    
    @abstractmethod
    def initialize_components(self):
        """
        Inicializa los componentes necesarios para la pestaña.
        Debe ser implementado por las clases hijas.
        """
        pass
    
    def log_message(self, message: str):
        """
        Registra un mensaje en el panel de log.
        
        Args:
            message: El mensaje a registrar
        """
        if hasattr(self, 'log_text'):
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.frame.update()
    
    def setup_log_panel(self, parent: ttk.Frame, height: int = 5) -> ttk.Frame:
        """
        Configura un panel de log estándar.
        
        Args:
            parent: El widget padre donde se creará el panel
            height: Altura del panel en líneas de texto
            
        Returns:
            ttk.Frame: El frame que contiene el panel de log
        """
        log_frame = ttk.LabelFrame(parent, text="Log del Proceso")
        log_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Scrollbar para el log
        log_scroll = ttk.Scrollbar(log_frame)
        log_scroll.pack(side='right', fill='y')
        
        # Text widget para el log
        self.log_text = tk.Text(log_frame, height=height, yscrollcommand=log_scroll.set)
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
        log_scroll.config(command=self.log_text.yview)
        
        return log_frame
    
    def setup_image_navigation(self, parent: ttk.Frame) -> ttk.Frame:
        """
        Configura un panel de navegación estándar para imágenes.
        
        Args:
            parent: El widget padre donde se creará el panel
            
        Returns:
            ttk.Frame: El frame que contiene los controles de navegación
        """
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(fill='x', padx=5, pady=5)
        
        # Botones de navegación
        ttk.Button(nav_frame, text="←", command=self.prev_image).pack(side='left', padx=5)
        self.image_counter = ttk.Label(nav_frame, text="0/0")
        self.image_counter.pack(side='left', padx=5)
        ttk.Button(nav_frame, text="→", command=self.next_image).pack(side='left', padx=5)
        
        return nav_frame
    
    def prev_image(self):
        """Navega a la imagen anterior."""
        if self.current_images:
            self.current_image_index = (self.current_image_index - 1) % len(self.current_images)
            self.show_current_image()
    
    def next_image(self):
        """Navega a la siguiente imagen."""
        if self.current_images:
            self.current_image_index = (self.current_image_index + 1) % len(self.current_images)
            self.show_current_image()
    
    def update_image_counter(self):
        """Actualiza el contador de imágenes."""
        if hasattr(self, 'image_counter'):
            if self.current_images:
                self.image_counter.config(
                    text=f"Imagen {self.current_image_index + 1} de {len(self.current_images)}"
                )
            else:
                self.image_counter.config(text="0/0")
    
    @abstractmethod
    def show_current_image(self):
        """
        Muestra la imagen actual.
        Debe ser implementado por las clases hijas.
        """
        pass
