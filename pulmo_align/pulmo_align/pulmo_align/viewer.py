"""
Clase principal de PulmoAlign Viewer.
"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path
import sys

# Agregar el directorio raíz al path para importar el paquete
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent  # Directorio raíz del proyecto

from pulmo_align.coordinates.coordinate_manager import CoordinateManager
from pulmo_align.image_processing.image_processor import ImageProcessor
from pulmo_align.visualization.visualizer import Visualizer
from pulmo_align.visualization.combined_visualizer import CombinedVisualizer

from pulmo_align.gui.pca_tab import PCATab
from pulmo_align.gui.extract_tab import ExtractTab

class PulmoAlignViewer:
    """Clase principal de la aplicación PulmoAlign Viewer."""
    
    def __init__(self, root):
        """
        Inicializa la aplicación.
        
        Args:
            root: Ventana principal de Tkinter
        """
        self.root = root
        self.root.title("PulmoAlign Viewer")
        self.root.geometry("1000x800")
        
        # Inicializar componentes compartidos
        self.initialize_shared_components()
        
        # Configurar interfaz
        self.setup_gui()
    
    def initialize_shared_components(self):
        """Inicializa los componentes compartidos entre pestañas."""
        # Gestores y procesadores
        self.coord_manager = CoordinateManager()
        self.image_processor = ImageProcessor(
            base_path=str(PROJECT_ROOT / "COVID-19_Radiography_Dataset")
        )
        
        # Visualizadores
        self.visualizer = Visualizer()
        self.combined_visualizer = CombinedVisualizer()
    
    def setup_gui(self):
        """Configura la interfaz gráfica principal."""
        # Crear notebook para pestañas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)
        
        # Crear pestaña PCA
        self.pca_tab = PCATab(
            parent=self.notebook,
            coord_manager=self.coord_manager,
            image_processor=self.image_processor,
            visualizer=self.visualizer,
            combined_visualizer=self.combined_visualizer,
            project_root=PROJECT_ROOT
        )
        self.notebook.add(self.pca_tab.frame, text='PCA Analysis')
        
        # Crear pestaña Extract
        self.extract_tab = ExtractTab(
            parent=self.notebook,
            coord_manager=self.coord_manager,
            image_processor=self.image_processor,
            project_root=PROJECT_ROOT
        )
        self.notebook.add(self.extract_tab.frame, text='Extract Images')

def main():
    """Función principal de la aplicación."""
    root = tk.Tk()
    app = PulmoAlignViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
