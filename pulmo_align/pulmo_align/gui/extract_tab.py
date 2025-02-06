"""
Implementación de la pestaña de extracción de imágenes.
"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path
from PIL import Image, ImageTk
import gc
import re
import threading
import time
import pandas as pd
from typing import Dict, List
import numpy as np

from pulmo_align.processing.parallel_processor import ParallelProcessor

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
            initial_value=str(Path(self.project_root).parent / "coordenadas.csv")
        )

        # Selector de archivo de índices
        self.indices_selector = FileSelector(
            parent=self.files_frame,
            label_text="Archivo de Índices:",
            file_types=[("CSV files", "*.csv")],
            initial_value=str(Path(self.project_root).parent / "indices.csv")
        )


        # Frame para controles de procesamiento
        process_frame = ttk.Frame(self.control_frame)
        process_frame.pack(fill='x', pady=5)
        
        # Selector de CPUs
        cpu_frame = ttk.Frame(process_frame)
        cpu_frame.pack(fill='x', padx=5)
        ttk.Label(cpu_frame, text="Número de CPUs:").pack(side='left', padx=5)
        self.cpu_spinbox = ttk.Spinbox(
            cpu_frame,
            from_=1,
            to=16,  # Máximo razonable
            width=5
        )
        self.cpu_spinbox.set(4)  # Valor por defecto
        self.cpu_spinbox.pack(side='left', padx=5)
        
        # Selector de tamaño de batch
        batch_frame = ttk.Frame(process_frame)
        batch_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(batch_frame, text="Tamaño de Batch:").pack(side='left', padx=5)
        self.batch_spinbox = ttk.Spinbox(
            batch_frame,
            from_=1,
            to=100,
            width=5
        )
        self.batch_spinbox.set(10)  # Valor por defecto
        self.batch_spinbox.pack(side='left', padx=5)

        # Frame para botones
        button_frame = ttk.Frame(process_frame)
        button_frame.pack(fill='x', pady=5)
        
        # Botón para iniciar extracción
        self.start_button = ttk.Button(
            button_frame,
            text="Iniciar Extracción",
            command=self.start_extraction
        )
        self.start_button.pack(side='left', padx=5)
        
        # Botón para cancelar
        self.cancel_button = ttk.Button(
            button_frame,
            text="Cancelar",
            command=self.cancel_extraction,
            state='disabled'
        )
        self.cancel_button.pack(side='left', padx=5)
        
        # Barra de progreso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            process_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill='x', padx=5, pady=5)

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
        """Inicia el proceso de extracción de imágenes usando procesamiento paralelo."""
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
            
            # Verificar archivo de templates pre-calculados
            template_data_path = Path(__file__).parent.parent.parent.parent / "tools" / "template_analysis" / "template_analysis_results.json"
            if template_data_path.exists():
                self.log_message(f"\nUsando templates pre-calculados de: {template_data_path}")
            else:
                self.log_message("\nAdvertencia: No se encontró el archivo de templates pre-calculados", "warning")

            # Validar archivos
            if not self.coord_selector.is_valid():
                self.log_message("Error: Archivo de coordenadas no encontrado", "error")
                return
            
            if not self.indices_selector.is_valid():
                self.log_message("Error: Archivo de índices no encontrado", "error")
                return

            # Configurar procesamiento paralelo
            num_cpus = int(self.cpu_spinbox.get())
            batch_size = int(self.batch_spinbox.get())
            
            self.log_message(f"\nConfigurando procesamiento paralelo:")
            self.log_message(f"- Usando {num_cpus} CPUs")
            self.log_message(f"- Tamaño de batch: {batch_size}")
            
            # Crear procesador paralelo
            self.parallel_processor = ParallelProcessor(
                image_processor=self.image_processor,
                num_cpus=num_cpus,
                batch_size=batch_size
            )
            
            # Cargar coordenadas
            self.coord_manager.read_coordinates(self.coord_selector.get_path())
            self.log_message("Coordenadas cargadas exitosamente")
            
            # Obtener total de imágenes del archivo de coordenadas
            try:
                coord_data = pd.read_csv(self.coord_selector.get_path(), header=None)
                total_images = len(coord_data)
                self.log_message(f"Total de imágenes a procesar: {total_images}")
            except Exception as e:
                self.log_message(f"Error al leer archivo de coordenadas: {str(e)}", "error")
                return
            
            # Configurar UI para procesamiento
            self.start_button.config(state='disabled')
            self.cancel_button.config(state='normal')
            self.progress_var.set(0)
            
            # Iniciar procesamiento en thread separado
            self.processing = True
            self.processing_thread = threading.Thread(
                target=self._run_parallel_processing,
                args=(total_images,)
            )
            self.processing_thread.start()
            
            # Iniciar monitoreo de progreso
            self.monitor_progress()

        except Exception as e:
            self.log_message(f"\nError al iniciar extracción: {str(e)}", "error")
            self.start_button.config(state='normal')
            self.cancel_button.config(state='disabled')
            
    def _run_parallel_processing(self, total_images: int):
        """Ejecuta el procesamiento paralelo en un thread separado."""
        try:
            # Iniciar procesamiento
            results, errors = self.parallel_processor.process_all(
                total_images=total_images,
                coord_file=self.coord_selector.get_path(),
                indices_file=self.indices_selector.get_path()
            )
            
            if not self.processing:
                self.log_message("\nProcesamiento cancelado")
                return
                
            # Procesar resultados
            total_processed = sum(result.processed for result in results)
            total_failed = sum(result.failed for result in results)
            total_time = sum(result.processing_time for result in results)
            
            # Recolectar y mostrar imágenes procesadas
            self.collect_extracted_images()
            
            # Mostrar resumen
            self.log_message("\nProceso completado")
            self.log_message(f"Tiempo total: {total_time:.2f} segundos")
            self.log_message("\nResumen final:")
            self.log_message(f"Total de imágenes procesadas exitosamente: {total_processed}")
            self.log_message(f"Total de imágenes fallidas: {total_failed}")
            self.log_message(f"Total de operaciones: {total_processed + total_failed}")
            self.log_message(f"Velocidad promedio: {total_processed/total_time:.2f} imágenes/segundo")
            
            # Mostrar errores si los hay
            if errors:
                self.log_message("\nErrores encontrados:", "warning")
                for error in errors:
                    self.log_message(f"- {error}", "error")
            
        except Exception as e:
            self.log_message(f"\nError durante el procesamiento: {str(e)}", "error")
            
        finally:
            # Restaurar UI
            self.processing = False
            self.start_button.config(state='normal')
            self.cancel_button.config(state='disabled')
            self.progress_var.set(100)
            
    def cancel_extraction(self):
        """Cancela el proceso de extracción."""
        if hasattr(self, 'processing') and self.processing:
            self.processing = False
            self.cancel_button.config(state='disabled')
            self.log_message("\nCancelando procesamiento...")
            
    def monitor_progress(self):
        """Monitorea y actualiza el progreso del procesamiento."""
        if hasattr(self, 'processing') and self.processing:
            try:
                # Obtener progreso actual
                progress = self.parallel_processor.get_progress()
                if progress:
                    # Calcular progreso promedio
                    avg_progress = sum(progress.values()) / len(progress) * 100
                    self.progress_var.set(avg_progress)
                
                # Continuar monitoreando
                self.frame.after(100, self.monitor_progress)
                    
            except Exception as e:
                self.log_message(f"Error monitoreando progreso: {str(e)}", "error")
    
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
            
            # MODIFICACIÓN: Buscar solo directorios de Coord1 y Coord2
            coord_dirs = [
                processed_dir / "cropped_images_Coord1",
                processed_dir / "cropped_images_Coord2"
            ]
            
            for coord_dir in coord_dirs:
                if not coord_dir.exists():
                    continue
                    
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
