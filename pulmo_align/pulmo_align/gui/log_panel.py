"""
Componente de panel de log para PulmoAlign Viewer.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional

class LogPanel:
    """Componente reutilizable para mostrar logs."""
    
    def __init__(self, parent: ttk.Frame, height: int = 5, title: str = "Log del Proceso"):
        """
        Inicializa un nuevo panel de log.
        
        Args:
            parent: Widget padre donde se creará el panel
            height: Altura del panel en líneas de texto
            title: Título del panel
        """
        self.parent = parent
        self.frame = self._create_frame(title)
        self.log_text = self._create_text_widget(height)
        self.pending_messages = []
        
        # Programar el primer procesamiento de mensajes pendientes
        self.frame.after(100, self._process_pending_messages)
        
    def _create_frame(self, title: str) -> ttk.Frame:
        """
        Crea el frame principal del panel.
        
        Args:
            title: Título del panel
            
        Returns:
            ttk.Frame: Frame configurado
        """
        frame = ttk.LabelFrame(self.parent, text=title)
        frame.pack(fill='both', expand=True, padx=5, pady=5)
        return frame
        
    def _create_text_widget(self, height: int) -> tk.Text:
        """
        Crea el widget de texto con scrollbar.
        
        Args:
            height: Altura del widget en líneas de texto
            
        Returns:
            tk.Text: Widget de texto configurado
        """
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.frame)
        scrollbar.pack(side='right', fill='y')
        
        # Text widget
        text = tk.Text(self.frame, height=height, yscrollcommand=scrollbar.set)
        text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Configurar scrollbar
        scrollbar.config(command=text.yview)
        
        return text
    
    def _process_pending_messages(self):
        """Procesa los mensajes pendientes."""
        try:
            while self.pending_messages:
                message, level = self.pending_messages.pop(0)
                self._write_message(message, level)
            
            # Programar el siguiente procesamiento
            self.frame.after(100, self._process_pending_messages)
        except Exception:
            # Si hay un error, intentar nuevamente más tarde
            self.frame.after(100, self._process_pending_messages)
    
    def _write_message(self, message: str, level: Optional[str] = None):
        """
        Escribe un mensaje en el widget de texto.
        
        Args:
            message: Mensaje a escribir
            level: Nivel del mensaje (info, warning, error)
        """
        try:
            # Agregar prefijo según el nivel
            prefix = ""
            if level == "warning":
                prefix = "[ADVERTENCIA] "
            elif level == "error":
                prefix = "[ERROR] "
                
            # Insertar mensaje
            self.log_text.insert(tk.END, prefix + message + "\n")
            self.log_text.see(tk.END)
            self.frame.update()
        except Exception:
            # Si hay un error al escribir, agregar el mensaje a la cola
            self.pending_messages.append((message, level))
    
    def log(self, message: str, level: Optional[str] = None) -> None:
        """
        Registra un mensaje en el panel.
        
        Args:
            message: Mensaje a registrar
            level: Nivel del mensaje (info, warning, error)
        """
        try:
            self._write_message(message, level)
        except Exception:
            # Si hay un error, agregar el mensaje a la cola
            self.pending_messages.append((message, level))
    
    def clear(self) -> None:
        """Limpia todos los mensajes del panel."""
        try:
            self.log_text.delete(1.0, tk.END)
            self.pending_messages.clear()
        except Exception:
            pass
    
    def get_text(self) -> str:
        """
        Obtiene todo el texto del panel.
        
        Returns:
            str: Contenido completo del panel
        """
        try:
            return self.log_text.get(1.0, tk.END)
        except Exception:
            return ""
    
    def save_to_file(self, filename: str) -> None:
        """
        Guarda el contenido del log en un archivo.
        
        Args:
            filename: Nombre del archivo donde guardar
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.get_text())
        except Exception:
            pass
