"""
Implementación de la pestaña de análisis PCA.
"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path
import gc
from PIL import Image, ImageTk

from pulmo_align.gui.base_tab import TabBase
from pulmo_align.gui.log_panel import LogPanel
from pulmo_align.gui.image_viewer import ImageViewer
from pulmo_align.utils.image_utils import clean_directory
from pulmo_align.utils.file_utils import browse_file, is_image_file
from pulmo_align.pca_analysis.pca_analyzer import PCAAnalyzer  # Importar PCAAnalyzer directamente

class PCATab(TabBase):
    """Pestaña para análisis PCA de imágenes pulmonares."""
    
    def __init__(self, parent, coord_manager, image_processor, visualizer, combined_visualizer, project_root):
        """
        Inicializa la pestaña PCA.
        
        Args:
            parent: Widget padre
            coord_manager: Gestor de coordenadas
            image_processor: Procesador de imágenes
            visualizer: Visualizador de resultados
            combined_visualizer: Visualizador de resultados combinados
            project_root: Ruta raíz del proyecto
        """
        self.coord_manager = coord_manager
        self.image_processor = image_processor
        self.visualizer = visualizer
        self.combined_visualizer = combined_visualizer
        self.project_root = project_root
        
        # Inicializar modelos PCA
        self.pca_models = {}
        
        super().__init__(parent)
    
    def setup_gui(self):
        """Configura la interfaz gráfica de la pestaña."""
        # Frame principal con dos paneles
        self.main_frame = ttk.PanedWindow(self.frame, orient='vertical')
        self.main_frame.pack(fill='both', expand=True)

        # Panel superior para controles y log
        self.top_panel = ttk.Frame(self.main_frame)
        self.main_frame.add(self.top_panel, weight=1)

        # Panel inferior para imágenes
        self.bottom_panel = ttk.Frame(self.main_frame)
        self.main_frame.add(self.bottom_panel, weight=3)

        # Frame para controles
        self.control_frame = ttk.Frame(self.top_panel)
        self.control_frame.pack(fill='x', padx=5, pady=5)

        # Botón para seleccionar imagen
        ttk.Button(
            self.control_frame,
            text="Seleccionar Imagen de Prueba",
            command=self.select_and_analyze_image
        ).pack(side='left', padx=5)

        # Label para mostrar imagen seleccionada
        self.image_label = ttk.Label(self.control_frame, text="No se ha seleccionado imagen")
        self.image_label.pack(side='left', padx=5)

        # Panel de log
        self.log_panel = LogPanel(self.top_panel)

        # Visualizador de imágenes
        self.image_viewer = ImageViewer(self.bottom_panel)
    
    def initialize_components(self):
        """Inicializa los componentes necesarios."""
        try:
            self.log_message("Inicializando componentes...")
            
            # Cargar coordenadas de búsqueda
            self.coord_manager.read_search_coordinates(
                str(self.project_root.parent / "all_search_coordinates.json")
            )
            self.log_message("Coordenadas de búsqueda cargadas")

            # Inicializar modelos PCA
            for coord_name, config in self.coord_manager.coord_data.items():
                self.log_message(f"Inicializando PCA para {coord_name}...")
                
                # Cargar imágenes de entrenamiento
                training_images = self.image_processor.load_training_images(
                    coord_name=coord_name,
                    target_size=(config['width'], config['height'])
                )
                
                if training_images:
                    # Inicializar y entrenar PCA
                    pca = PCAAnalyzer()  # Usar PCAAnalyzer directamente
                    pca.train(training_images)
                    self.pca_models[coord_name] = pca
                    
                    model_info = pca.get_model_info()
                    self.log_message(f"PCA {coord_name}: {model_info['n_components']} componentes")

            self.log_message("Inicialización completada")
            
        except Exception as e:
            self.log_message(f"Error en inicialización: {str(e)}")
    
    def select_and_analyze_image(self):
        """Selecciona y analiza una imagen."""
        filetypes = [
            ('Imágenes', '*.png;*.jpg;*.jpeg'),
            ('Todos los archivos', '*.*')
        ]
        
        initial_dir = str(self.project_root.parent / "COVID-19_Radiography_Dataset")
        if not Path(initial_dir).exists():
            initial_dir = "."

        filepath = browse_file(
            "Seleccionar imagen de prueba",
            filetypes,
            initial_dir
        )
        
        if filepath:
            self.image_label.config(text=f"Imagen seleccionada: {Path(filepath).name}")
            self.analyze_image(filepath)
    
    def analyze_image(self, image_path):
        """
        Analiza una imagen usando PCA.
        
        Args:
            image_path: Ruta de la imagen a analizar
        """
        try:
            # Limpiar visualizaciones anteriores
            self.image_viewer.clear()
            
            # Limpiar directorio de visualizaciones
            viz_dir = self.project_root.parent / "visualization_results"
            clean_directory(viz_dir)

            self.log_message(f"\nAnalizando imagen: {image_path}")
            
            # Cargar y procesar imagen
            image = self.image_processor.load_and_resize_image(image_path)
            results = {}
            
            # Analizar cada coordenada
            for coord_name, config in self.coord_manager.coord_data.items():
                if coord_name not in self.pca_models:
                    self.log_message(f"Saltando {coord_name}: no hay modelo PCA")
                    continue
                    
                try:
                    self.log_message(f"\nAnalizando {coord_name}...")
                    
                    # Obtener coordenadas de búsqueda
                    search_coordinates = self.coord_manager.get_search_coordinates(coord_name)
                    self.log_message(f"Coordenadas de búsqueda: {len(search_coordinates)}")
                    
                    # Analizar región de búsqueda
                    min_error, min_error_coords, errors = self.pca_models[coord_name].analyze_search_region(
                        image=image,
                        search_coordinates=search_coordinates,
                        template_width=config['width'],
                        template_height=config['height'],
                        intersection_x=config['left'],
                        intersection_y=config['sup']
                    )

                    min_error_step = errors.index(min(errors)) + 1

                    self.log_message(f"Error mínimo: {min_error:.4f} en coordenadas: {min_error_coords}")
                    self.log_message(f"Encontrado en el paso: {min_error_step}")

                    results[coord_name] = {
                        'min_error': min_error,
                        'min_error_coords': min_error_coords,
                        'min_error_step': min_error_step,
                        'errors': errors,
                        'search_coordinates': search_coordinates
                    }
                    
                    # Visualizar distribución de errores
                    self.visualizer.plot_error_distribution(
                        errors=errors,
                        coord_name=coord_name,
                        save=True
                    )
                    
                    # Visualizar camino de búsqueda
                    self.visualizer.plot_search_path(
                        search_coordinates=search_coordinates,
                        min_error_coords=min_error_coords,
                        coord_name=coord_name,
                        save=True
                    )
                    
                except Exception as e:
                    self.log_message(f"Error procesando {coord_name}: {str(e)}")
                    continue

            # Visualizar resultados finales
            if results:
                self.log_message("\nGenerando visualizaciones...")
                self.visualizer.visualize_results(
                    image=image,
                    coord_config=self.coord_manager.coord_data,
                    results=results,
                    pca_models=self.pca_models,
                    save=True
                )
                
                # Generar visualización combinada
                self.log_message("\nGenerando visualización combinada...")
                self.combined_visualizer.visualize_combined_results(
                    image=image,
                    results=results,
                    save=True
                )
                
                # Generar visualización de Coord1 y Coord2
                self.log_message("\nGenerando visualización de Coord1 y Coord2...")
                self.combined_visualizer.visualize_coord1_coord2(
                    image=image,
                    results=results,
                    save=True
                )

                # Recolectar y mostrar imágenes
                self.collect_result_images()
            else:
                self.log_message("No se obtuvieron resultados para analizar")

            self.log_message("Análisis completado")

        except Exception as e:
            self.log_message(f"Error en análisis: {str(e)}")
    
    def collect_result_images(self):
        """Recolecta y muestra las imágenes de resultados."""
        images = []
        
        # 1. Primero Coord1 y Coord2
        viz_dir = self.project_root.parent / "visualization_results"
        if (viz_dir / "coord1_coord2_results.png").exists():
            images.append({
                'path': str(viz_dir / "coord1_coord2_results.png"),
                'title': "Resultados Coord1 y Coord2"
            })

        # 2. Visualización combinada
        if (viz_dir / "combined_results.png").exists():
            images.append({
                'path': str(viz_dir / "combined_results.png"),
                'title': "Resultados Combinados"
            })

        # 3. Resultados individuales
        for i in range(1, 16):
            coord_name = f"Coord{i}"
            result_path = viz_dir / f"{coord_name}_results.png"
            if result_path.exists():
                images.append({
                    'path': str(result_path),
                    'title': f"Resultados {coord_name}"
                })

        # Cargar imágenes en el visualizador
        self.image_viewer.load_images(images)
    
    def show_current_image(self):
        """Muestra la imagen actual."""
        self.image_viewer.show_current_image()
    
    def log_message(self, message):
        """
        Registra un mensaje en el panel de log.
        
        Args:
            message: Mensaje a registrar
        """
        self.log_panel.log(message)
