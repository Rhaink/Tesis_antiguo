"""
Implementación de la pestaña de extracción de imágenes.
"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path
from PIL import Image, ImageTk
import gc
import re
from typing import Dict, List

from pulmo_align.gui.base_tab import TabBase
from pulmo_align.gui.log_panel import LogPanel
from pulmo_align.gui.image_viewer import ImageViewer
from pulmo_align.gui.file_selector import FileSelector
from pulmo_align.utils.image_utils import collect_processed_images

def natural_sort_key(s: str) -> list:
    """
    Función auxiliar para ordenamiento natural de strings con números.
    Convierte "99" en el número 99 para comparación.
    
    Args:
        s: String a convertir
        
    Returns:
        list: Lista de partes convertidas para ordenamiento
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

class ExtractTab(TabBase):
    """Pestaña para extracción de imágenes pulmonares."""
    
    def __init__(self, parent, coord_manager, image_processor, project_root):
        """
        Inicializa la pestaña de extracción.
        
        Args:
            parent: Widget padre
            coord_manager: Gestor de coordenadas
            image_processor: Procesador de imágenes
            project_root: Ruta raíz del proyecto
        """
        self.coord_manager = coord_manager
        self.image_processor = image_processor
        self.project_root = project_root
        self.photo_references = []  # Para mantener referencias a las imágenes
        self.all_images: Dict[str, List[Dict]] = {}  # Imágenes organizadas por coordenada
        super().__init__(parent)
    
    def setup_gui(self):
        """Configura la interfaz gráfica de la pestaña."""
        # Frame principal con dos paneles
        self.main_frame = ttk.PanedWindow(self.frame, orient='vertical')
        self.main_frame.pack(fill='both', expand=True)

        # Panel superior para controles y log
        self.top_panel = ttk.Frame(self.main_frame)
        self.main_frame.add(self.top_panel, weight=1)

        # Panel inferior para resultados
        self.bottom_panel = ttk.Frame(self.main_frame)
        self.main_frame.add(self.bottom_panel, weight=3)

        # Frame para controles
        self.control_frame = ttk.Frame(self.top_panel)
        self.control_frame.pack(fill='x', padx=5, pady=5)

        # Frame para archivos de entrada
        self.files_frame = ttk.LabelFrame(self.control_frame, text="Archivos de Entrada")
        self.files_frame.pack(fill='x', padx=5, pady=5)

        # Selector de archivo de coordenadas
        self.coord_selector = FileSelector(
            parent=self.files_frame,
            label_text="Archivo de Coordenadas:",
            file_types=[("CSV files", "*.csv")],
            initial_value=str(self.project_root / "coordenadas.csv")
        )

        # Selector de archivo de índices
        self.indices_selector = FileSelector(
            parent=self.files_frame,
            label_text="Archivo de Índices:",
            file_types=[("CSV files", "*.csv")],
            initial_value=str(self.project_root / "indices.csv")
        )

        # Botón para iniciar extracción
        ttk.Button(
            self.control_frame,
            text="Iniciar Extracción",
            command=self.start_extraction
        ).pack(pady=10)

        # Panel de log
        self.log_panel = LogPanel(self.top_panel)

        # Frame para visualización de imágenes
        self.image_viewer_frame = ttk.Frame(self.bottom_panel)
        self.image_viewer_frame.pack(fill='both', expand=True)

        # Frame para selector de coordenadas
        coord_select_frame = ttk.Frame(self.image_viewer_frame)
        coord_select_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(coord_select_frame, text="Coordenada:").pack(side='left', padx=5)
        self.coord_combobox = ttk.Combobox(coord_select_frame, state='readonly')
        self.coord_combobox.pack(side='left', padx=5)
        self.coord_combobox.bind('<<ComboboxSelected>>', self.on_coord_selected)

        # Frame para la imagen actual
        self.current_image_frame = ttk.Frame(self.image_viewer_frame)
        self.current_image_frame.pack(fill='both', expand=True)

        # Botones de navegación
        nav_frame = ttk.Frame(self.image_viewer_frame)
        nav_frame.pack(fill='x', padx=5, pady=5)
        ttk.Button(nav_frame, text="←", command=self.prev_image).pack(side='left', padx=5)
        self.image_counter = ttk.Label(nav_frame, text="0/0")
        self.image_counter.pack(side='left', padx=5)
        ttk.Button(nav_frame, text="→", command=self.next_image).pack(side='left', padx=5)

        # Cargar imágenes existentes si las hay
        self.collect_extracted_images()
    
    def initialize_components(self):
        """Inicializa los componentes necesarios."""
        # No se requiere inicialización especial para esta pestaña
        pass
    
    def on_coord_selected(self, event):
        """Maneja el cambio de coordenada seleccionada."""
        coord = self.coord_combobox.get()
        if coord in self.all_images:
            self.current_images = self.all_images[coord]
            self.current_image_index = 0
            self.show_current_image()
    
    def start_extraction(self):
        """Inicia el proceso de extracción de imágenes."""
        try:
            # Limpiar visualizador
            self.current_images = []
            self.current_image_index = 0
            self.photo_references.clear()
            self.all_images.clear()
            gc.collect()
            
            self.log_message("\nIniciando proceso de extracción...")
            self.log_message(f"Archivo de coordenadas: {self.coord_selector.get_path()}")
            self.log_message(f"Archivo de índices: {self.indices_selector.get_path()}")

            # Validar archivos
            if not self.coord_selector.is_valid():
                self.log_message("Error: Archivo de coordenadas no encontrado", "error")
                return
            
            if not self.indices_selector.is_valid():
                self.log_message("Error: Archivo de índices no encontrado", "error")
                return

            # Procesar imágenes
            results = self.process_images(
                self.coord_selector.get_path(),
                self.indices_selector.get_path()
            )

            # Recolectar y mostrar imágenes extraídas
            self.collect_extracted_images()

            self.log_message("\nProceso completado")
            
            # Mostrar resumen final
            total_processed = sum(result['processed'] for result in results.values())
            total_failed = sum(result['failed'] for result in results.values())
            
            self.log_message("\nResumen final:")
            self.log_message(f"Total de imágenes procesadas exitosamente: {total_processed}")
            self.log_message(f"Total de imágenes fallidas: {total_failed}")
            self.log_message(f"Total de operaciones: {total_processed + total_failed}")

        except Exception as e:
            self.log_message(f"\nError durante la extracción: {str(e)}", "error")
    
    def process_images(self, coordinates_file: str, indices_file: str) -> dict:
        """
        Procesa las imágenes y extrae las regiones de interés.
        
        Args:
            coordinates_file: Ruta al archivo de coordenadas
            indices_file: Ruta al archivo de índices
            
        Returns:
            dict: Resultados del procesamiento
        """
        results = {coord_name: {'processed': 0, 'failed': 0, 'errors': []} 
                  for coord_name in self.coord_manager.coord_data.keys()}
        
        try:
            self.log_message("\nLeyendo coordenadas...")
            self.coord_manager.read_coordinates(coordinates_file)
            
            total_images = len(self.coord_manager.coordinates)
            self.log_message(f"\nTotal de imágenes a procesar: {total_images}")
            
            # Procesamos cada coordenada
            for coord_name, config in self.coord_manager.coord_data.items():
                self.log_message(f"\nProcesando {coord_name}...")
                
                # Calculamos centro e intersección
                center_x, center_y = self.coord_manager.calculate_center(
                    config["sup"], config["inf"],
                    config["left"], config["right"]
                )
                intersection_x, intersection_y = self.coord_manager.calculate_intersection(
                    config["sup"], config["inf"],
                    config["left"], config["right"]
                )

                self.log_message(f"Centro: ({center_x}, {center_y})")
                self.log_message(f"Intersección: ({intersection_x}, {intersection_y})")
                self.log_message(f"Dimensiones: {config['width']}x{config['height']}")

                # Procesamos cada imagen
                for index, coords in self.coord_manager.coordinates.items():
                    try:
                        # Obtener ruta de la imagen
                        image_path = self.image_processor.get_image_path(index, indices_file)
                        
                        # Cargar y redimensionar imagen
                        image = self.image_processor.load_and_resize_image(image_path)
                        
                        # Obtener nuevas coordenadas
                        new_x, new_y = coords[coord_name]
                        
                        # Extraer región
                        cropped_image = self.image_processor.extract_region(
                            image=image,
                            center_x=center_x,
                            center_y=center_y,
                            width=config["width"],
                            height=config["height"],
                            intersection_x=intersection_x,
                            intersection_y=intersection_y,
                            new_x=new_x,
                            new_y=new_y
                        )
                        
                        # Guardar imagen recortada
                        success = self.image_processor.save_cropped_image(
                            cropped_image=cropped_image,
                            coord_name=coord_name,
                            index=index
                        )
                        
                        if success:
                            results[coord_name]['processed'] += 1
                        else:
                            results[coord_name]['failed'] += 1
                            results[coord_name]['errors'].append(f"Error al guardar imagen {index}")
                        
                        # Actualizar cada 10 imágenes
                        if index % 10 == 0:
                            self.log_message(f"Procesadas: {index}/{total_images}")
                            self.frame.update()
                        
                    except Exception as e:
                        self.log_message(f"\nError procesando imagen {index} para {coord_name}: {str(e)}")
                        results[coord_name]['failed'] += 1
                        results[coord_name]['errors'].append(f"Error en imagen {index}: {str(e)}")

                # Mostrar resultados para esta coordenada
                self.log_message(f"\nResultados para {coord_name}:")
                self.log_message(f"  Procesadas exitosamente: {results[coord_name]['processed']}")
                self.log_message(f"  Fallidas: {results[coord_name]['failed']}")
                if results[coord_name]['errors']:
                    self.log_message("\nErrores encontrados:")
                    for error in results[coord_name]['errors'][:5]:
                        self.log_message(f"  - {error}")
                    if len(results[coord_name]['errors']) > 5:
                        self.log_message(f"  ... y {len(results[coord_name]['errors']) - 5} errores más")

        except Exception as e:
            self.log_message(f"\nError durante la ejecución: {str(e)}", "error")
            
        return results
    
    def collect_extracted_images(self):
        """Recolecta y muestra las imágenes extraídas."""
        try:
            # Obtener imágenes procesadas
            processed_dir = Path("processed_images")
            if not processed_dir.exists():
                self.log_message("No se encontró el directorio de imágenes procesadas", "error")
                return

            # Limpiar imágenes anteriores
            self.all_images.clear()
            
            # Buscar en cada directorio de coordenadas
            coord_dirs = sorted(processed_dir.glob("cropped_images_Coord*"), 
                              key=lambda x: natural_sort_key(x.name))
            
            for coord_dir in coord_dirs:
                coord_name = coord_dir.name.replace("cropped_images_", "")
                
                # Recolectar todas las imágenes en este directorio
                coord_images = []
                for img_path in coord_dir.glob("*.png"):
                    coord_images.append({
                        'path': str(img_path.resolve()),  # Usar ruta absoluta
                        'title': img_path.name,  # Solo el nombre del archivo
                        'sort_key': natural_sort_key(img_path.stem)  # Usar el nombre sin extensión
                    })
                
                # Ordenar imágenes de esta coordenada
                coord_images.sort(key=lambda x: x['sort_key'])
                
                # Guardar imágenes de esta coordenada
                self.all_images[coord_name] = coord_images

            if not self.all_images:
                self.log_message("No se encontraron imágenes procesadas", "warning")
            else:
                total_images = sum(len(images) for images in self.all_images.values())
                self.log_message(f"Se encontraron {total_images} imágenes procesadas")
                
                # Actualizar combobox con las coordenadas disponibles
                coords = sorted(self.all_images.keys(), key=natural_sort_key)
                self.coord_combobox['values'] = coords
                
                # Seleccionar primera coordenada
                if coords:
                    self.coord_combobox.set(coords[0])
                    self.current_images = self.all_images[coords[0]]
                    self.current_image_index = 0
                    self.show_current_image()
            
        except Exception as e:
            self.log_message(f"Error al recolectar imágenes: {str(e)}", "error")
    
    def show_current_image(self):
        """Muestra la imagen actual."""
        # Limpiar frame actual
        for widget in self.current_image_frame.winfo_children():
            widget.destroy()

        if not self.current_images:
            ttk.Label(self.current_image_frame, text="No hay imágenes disponibles").pack()
            self.update_image_counter()
            return

        # Mostrar imagen actual
        current = self.current_images[self.current_image_index]
        try:
            img = Image.open(current['path'])
            # Redimensionar manteniendo proporción
            display_width = 900
            ratio = display_width / img.width
            display_height = int(img.height * ratio)
            
            img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Mantener referencia
            self.photo_references = [photo]
            
            # Mostrar título
            ttk.Label(self.current_image_frame, text=current['title']).pack()
            
            # Mostrar imagen
            label = ttk.Label(self.current_image_frame, image=photo)
            label.image = photo
            label.pack(padx=5, pady=5)

        except Exception as e:
            ttk.Label(self.current_image_frame, text=f"Error al cargar imagen: {str(e)}").pack()

        self.update_image_counter()
    
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
        if self.current_images:
            self.image_counter.config(
                text=f"Imagen {self.current_image_index + 1} de {len(self.current_images)}"
            )
        else:
            self.image_counter.config(text="0/0")
    
    def log_message(self, message, level=None):
        """
        Registra un mensaje en el panel de log.
        
        Args:
            message: Mensaje a registrar
            level: Nivel del mensaje (info, warning, error)
        """
        self.log_panel.log(message, level)
