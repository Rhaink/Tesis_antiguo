import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os
from pathlib import Path
import sys
import gc

# Agregar el directorio raíz al path para importar el paquete
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

from pulmo_align.coordinates.coordinate_manager import CoordinateManager
from pulmo_align.image_processing.image_processor import ImageProcessor
from pulmo_align.pca_analysis.pca_analyzer import PCAAnalyzer
from pulmo_align.visualization.visualizer import Visualizer
from pulmo_align.visualization.combined_visualizer import CombinedVisualizer

class PulmoAlignViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("PulmoAlign Viewer")
        self.root.geometry("1000x800")

        # Variables para navegación de imágenes
        self.current_images = []
        self.current_image_index = 0
        self.photo_references = []

        # Variables para navegación de imágenes extraídas
        self.extracted_images = []
        self.extracted_image_index = 0
        self.extracted_photo_references = []

        # Inicializar componentes
        self.setup_gui()
        self.initialize_components()

    def setup_gui(self):
        # Notebook para pestañas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        # Pestaña para PCA Analysis
        self.pca_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.pca_frame, text='PCA Analysis')

        # Pestaña para Extract Cropped Images
        self.extract_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.extract_frame, text='Extract Images')

        # Setup PCA Analysis Tab
        self.setup_pca_tab()

        # Setup Extract Images Tab
        self.setup_extract_tab()

    def setup_pca_tab(self):
        # Frame principal con dos paneles
        self.main_frame = ttk.PanedWindow(self.pca_frame, orient='vertical')
        self.main_frame.pack(fill='both', expand=True)

        # Panel superior para controles y log (más pequeño)
        self.top_panel = ttk.Frame(self.main_frame)
        self.main_frame.add(self.top_panel, weight=1)

        # Panel inferior para imágenes (más grande)
        self.bottom_panel = ttk.Frame(self.main_frame)
        self.main_frame.add(self.bottom_panel, weight=3)

        # Frame superior para controles
        self.control_frame = ttk.Frame(self.top_panel)
        self.control_frame.pack(fill='x', padx=5, pady=5)

        # Botón para seleccionar imagen
        ttk.Button(self.control_frame, text="Seleccionar Imagen de Prueba", 
                  command=self.select_and_analyze_image).pack(side='left', padx=5)

        # Label para mostrar imagen seleccionada
        self.image_label = ttk.Label(self.control_frame, text="No se ha seleccionado imagen")
        self.image_label.pack(side='left', padx=5)

        # Frame para log con scroll (altura reducida)
        self.log_frame = ttk.LabelFrame(self.top_panel, text="Log del Proceso")
        self.log_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Scrollbar para el log
        log_scroll = ttk.Scrollbar(self.log_frame)
        log_scroll.pack(side='right', fill='y')
        
        # Text widget para el log (altura reducida)
        self.log_text = tk.Text(self.log_frame, height=5, yscrollcommand=log_scroll.set)
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
        log_scroll.config(command=self.log_text.yview)

        # Frame para visualización de imágenes
        self.image_frame = ttk.Frame(self.bottom_panel)
        self.image_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Frame para la imagen actual
        self.current_image_frame = ttk.Frame(self.image_frame)
        self.current_image_frame.pack(fill='both', expand=True)

        # Frame para botones de navegación
        nav_frame = ttk.Frame(self.image_frame)
        nav_frame.pack(fill='x', padx=5, pady=5)

        # Botones de navegación
        ttk.Button(nav_frame, text="←", command=self.prev_image).pack(side='left', padx=5)
        self.image_counter = ttk.Label(nav_frame, text="0/0")
        self.image_counter.pack(side='left', padx=5)
        ttk.Button(nav_frame, text="→", command=self.next_image).pack(side='left', padx=5)

    def setup_extract_tab(self):
        # Frame principal con dos paneles
        self.extract_main_frame = ttk.PanedWindow(self.extract_frame, orient='vertical')
        self.extract_main_frame.pack(fill='both', expand=True)

        # Panel superior para controles y log
        self.extract_top_panel = ttk.Frame(self.extract_main_frame)
        self.extract_main_frame.add(self.extract_top_panel, weight=1)

        # Panel inferior para resultados
        self.extract_bottom_panel = ttk.Frame(self.extract_main_frame)
        self.extract_main_frame.add(self.extract_bottom_panel, weight=3)

        # Frame para controles
        self.extract_control_frame = ttk.Frame(self.extract_top_panel)
        self.extract_control_frame.pack(fill='x', padx=5, pady=5)

        # Frame para archivos de entrada
        self.files_frame = ttk.LabelFrame(self.extract_control_frame, text="Archivos de Entrada")
        self.files_frame.pack(fill='x', padx=5, pady=5)

        # Coordenadas
        coord_frame = ttk.Frame(self.files_frame)
        coord_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(coord_frame, text="Archivo de Coordenadas:").pack(side='left', padx=5)
        self.coord_path = tk.StringVar(value=str(SCRIPT_DIR / "coordenadas.csv"))
        ttk.Entry(coord_frame, textvariable=self.coord_path).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(coord_frame, text="Examinar", 
                  command=lambda: self.browse_file(self.coord_path, "Seleccionar archivo de coordenadas", 
                                                 [("CSV files", "*.csv")])).pack(side='left', padx=5)

        # Índices
        indices_frame = ttk.Frame(self.files_frame)
        indices_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(indices_frame, text="Archivo de Índices:").pack(side='left', padx=5)
        self.indices_path = tk.StringVar(value=str(SCRIPT_DIR / "indices.csv"))
        ttk.Entry(indices_frame, textvariable=self.indices_path).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(indices_frame, text="Examinar", 
                  command=lambda: self.browse_file(self.indices_path, "Seleccionar archivo de índices", 
                                                 [("CSV files", "*.csv")])).pack(side='left', padx=5)

        # Botón para iniciar extracción
        ttk.Button(self.extract_control_frame, text="Iniciar Extracción", 
                  command=self.start_extraction).pack(pady=10)

        # Frame para log
        self.extract_log_frame = ttk.LabelFrame(self.extract_top_panel, text="Log del Proceso")
        self.extract_log_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Scrollbar para el log
        extract_log_scroll = ttk.Scrollbar(self.extract_log_frame)
        extract_log_scroll.pack(side='right', fill='y')

        # Text widget para el log
        self.extract_log_text = tk.Text(self.extract_log_frame, height=5, 
                                      yscrollcommand=extract_log_scroll.set)
        self.extract_log_text.pack(fill='both', expand=True, padx=5, pady=5)
        extract_log_scroll.config(command=self.extract_log_text.yview)

        # Frame para visualización de imágenes extraídas
        self.extracted_image_frame = ttk.Frame(self.extract_bottom_panel)
        self.extracted_image_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Frame para la imagen extraída actual
        self.current_extracted_frame = ttk.Frame(self.extracted_image_frame)
        self.current_extracted_frame.pack(fill='both', expand=True)

        # Frame para botones de navegación de imágenes extraídas
        extract_nav_frame = ttk.Frame(self.extracted_image_frame)
        extract_nav_frame.pack(fill='x', padx=5, pady=5)

        # Botones de navegación para imágenes extraídas
        ttk.Button(extract_nav_frame, text="←", command=self.prev_extracted_image).pack(side='left', padx=5)
        self.extracted_counter = ttk.Label(extract_nav_frame, text="0/0")
        self.extracted_counter.pack(side='left', padx=5)
        ttk.Button(extract_nav_frame, text="→", command=self.next_extracted_image).pack(side='left', padx=5)

    def browse_file(self, string_var, title, filetypes):
        filename = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes + [("All files", "*.*")],
            initialdir=str(SCRIPT_DIR)
        )
        if filename:
            string_var.set(filename)

    def start_extraction(self):
        try:
            # Limpiar imágenes anteriores
            self.extracted_images = []
            self.extracted_image_index = 0
            self.extracted_photo_references.clear()
            
            self.extract_log_message("\nIniciando proceso de extracción...")
            self.extract_log_message(f"Archivo de coordenadas: {self.coord_path.get()}")
            self.extract_log_message(f"Archivo de índices: {self.indices_path.get()}")

            # Procesar imágenes
            results = self.process_images(self.coord_path.get(), self.indices_path.get())

            # Recolectar imágenes extraídas
            self.collect_extracted_images()
            self.show_current_extracted_image()

            self.extract_log_message("\nProceso completado")
            
            # Mostrar resumen final
            total_processed = sum(result['processed'] for result in results.values())
            total_failed = sum(result['failed'] for result in results.values())
            
            self.extract_log_message("\nResumen final:")
            self.extract_log_message(f"Total de imágenes procesadas exitosamente: {total_processed}")
            self.extract_log_message(f"Total de imágenes fallidas: {total_failed}")
            self.extract_log_message(f"Total de operaciones: {total_processed + total_failed}")

        except Exception as e:
            self.extract_log_message(f"\nError durante la extracción: {str(e)}")

    def collect_extracted_images(self):
        """Recolecta todas las imágenes extraídas de los directorios de resultados."""
        self.extracted_images = []
        base_dir = Path("processed_images")
        
        if base_dir.exists():
            # Buscar en cada directorio de coordenadas
            for coord_dir in base_dir.glob("cropped_images_Coord*"):
                coord_name = coord_dir.name.replace("cropped_images_", "")
                
                # Recolectar todas las imágenes en este directorio
                for img_path in coord_dir.glob("*.png"):
                    self.extracted_images.append({
                        'path': str(img_path),
                        'title': f"{coord_name} - {img_path.name}"
                    })

        # Ordenar las imágenes por nombre de coordenada y nombre de archivo
        self.extracted_images.sort(key=lambda x: (x['title'].split(' - ')[0], x['title'].split(' - ')[1]))
        self.update_extracted_counter()

    def show_current_extracted_image(self):
        """Muestra la imagen extraída actual."""
        # Limpiar frame actual
        for widget in self.current_extracted_frame.winfo_children():
            widget.destroy()

        if not self.extracted_images:
            ttk.Label(self.current_extracted_frame, text="No hay imágenes extraídas disponibles").pack()
            return

        # Mostrar imagen actual
        current = self.extracted_images[self.extracted_image_index]
        try:
            img = Image.open(current['path'])
            # Redimensionar manteniendo proporción
            display_width = 900
            ratio = display_width / img.width
            display_height = int(img.height * ratio)
            
            img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Mantener referencia
            self.extracted_photo_references = [photo]
            
            # Mostrar título
            ttk.Label(self.current_extracted_frame, text=current['title']).pack()
            
            # Mostrar imagen
            label = ttk.Label(self.current_extracted_frame, image=photo)
            label.image = photo
            label.pack(padx=5, pady=5)

        except Exception as e:
            ttk.Label(self.current_extracted_frame, text=f"Error al cargar imagen: {str(e)}").pack()

        self.update_extracted_counter()

    def prev_extracted_image(self):
        """Navega a la imagen extraída anterior."""
        if self.extracted_images:
            self.extracted_image_index = (self.extracted_image_index - 1) % len(self.extracted_images)
            self.show_current_extracted_image()

    def next_extracted_image(self):
        """Navega a la siguiente imagen extraída."""
        if self.extracted_images:
            self.extracted_image_index = (self.extracted_image_index + 1) % len(self.extracted_images)
            self.show_current_extracted_image()

    def update_extracted_counter(self):
        """Actualiza el contador de imágenes extraídas."""
        if self.extracted_images:
            self.extracted_counter.config(
                text=f"Imagen {self.extracted_image_index + 1} de {len(self.extracted_images)}"
            )
        else:
            self.extracted_counter.config(text="0/0")

    def process_images(self, coordinates_file: str, indices_file: str) -> dict:
        """
        Procesa las imágenes y extrae las regiones de interés.
        """
        results = {coord_name: {'processed': 0, 'failed': 0, 'errors': []} 
                  for coord_name in self.coord_manager.coord_data.keys()}
        
        try:
            self.extract_log_message("\nLeyendo coordenadas...")
            self.coord_manager.read_coordinates(coordinates_file)
            
            total_images = len(self.coord_manager.coordinates)
            self.extract_log_message(f"\nTotal de imágenes a procesar: {total_images}")
            
            # Procesamos cada coordenada
            for coord_name, config in self.coord_manager.coord_data.items():
                self.extract_log_message(f"\nProcesando {coord_name}...")
                
                # Calculamos centro e intersección
                center_x, center_y = self.coord_manager.calculate_center(
                    config["sup"], config["inf"],
                    config["left"], config["right"]
                )
                intersection_x, intersection_y = self.coord_manager.calculate_intersection(
                    config["sup"], config["inf"],
                    config["left"], config["right"]
                )

                self.extract_log_message(f"Centro: ({center_x}, {center_y})")
                self.extract_log_message(f"Intersección: ({intersection_x}, {intersection_y})")
                self.extract_log_message(f"Dimensiones: {config['width']}x{config['height']}")

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
                            self.extract_log_message(f"Procesadas: {index}/{total_images}")
                            self.root.update()
                        
                    except Exception as e:
                        self.extract_log_message(f"\nError procesando imagen {index} para {coord_name}: {str(e)}")
                        results[coord_name]['failed'] += 1
                        results[coord_name]['errors'].append(f"Error en imagen {index}: {str(e)}")

                # Mostrar resultados para esta coordenada
                self.extract_log_message(f"\nResultados para {coord_name}:")
                self.extract_log_message(f"  Procesadas exitosamente: {results[coord_name]['processed']}")
                self.extract_log_message(f"  Fallidas: {results[coord_name]['failed']}")
                if results[coord_name]['errors']:
                    self.extract_log_message("\nErrores encontrados:")
                    for error in results[coord_name]['errors'][:5]:
                        self.extract_log_message(f"  - {error}")
                    if len(results[coord_name]['errors']) > 5:
                        self.extract_log_message(f"  ... y {len(results[coord_name]['errors']) - 5} errores más")

        except Exception as e:
            self.extract_log_message(f"\nError durante la ejecución: {str(e)}")
            
        return results

    def extract_log_message(self, message):
        self.extract_log_text.insert(tk.END, message + "\n")
        self.extract_log_text.see(tk.END)
        self.root.update()

    def initialize_components(self):
        try:
            self.log_message("Inicializando componentes...")
            
            self.coord_manager = CoordinateManager()
            self.image_processor = ImageProcessor(base_path=str(SCRIPT_DIR / "COVID-19_Radiography_Dataset"))
            self.visualizer = Visualizer()
            self.combined_visualizer = CombinedVisualizer()
            
            # Cargar coordenadas de búsqueda
            self.coord_manager.read_search_coordinates(str(SCRIPT_DIR / "all_search_coordinates.json"))
            self.log_message("Coordenadas de búsqueda cargadas")

            # Inicializar modelos PCA
            self.pca_models = {}
            for coord_name, config in self.coord_manager.coord_data.items():
                self.log_message(f"Inicializando PCA para {coord_name}...")
                
                # Cargar imágenes de entrenamiento
                training_images = self.image_processor.load_training_images(
                    coord_name=coord_name,
                    target_size=(config['width'], config['height'])
                )
                
                if training_images:
                    # Inicializar y entrenar PCA
                    pca = PCAAnalyzer()
                    pca.train(training_images)
                    self.pca_models[coord_name] = pca
                    
                    model_info = pca.get_model_info()
                    self.log_message(f"PCA {coord_name}: {model_info['n_components']} componentes")

            self.log_message("Inicialización completada")
            
        except Exception as e:
            self.log_message(f"Error en inicialización: {str(e)}")

    def select_and_analyze_image(self):
        filetypes = [
            ('Imágenes', '*.png;*.jpg;*.jpeg'),
            ('Todos los archivos', '*.*')
        ]
        
        initial_dir = str(SCRIPT_DIR / "COVID-19_Radiography_Dataset")
        if not Path(initial_dir).exists():
            initial_dir = "."

        filepath = filedialog.askopenfilename(
            title='Seleccionar imagen de prueba',
            filetypes=filetypes,
            initialdir=initial_dir
        )
        
        if filepath:
            self.image_label.config(text=f"Imagen seleccionada: {Path(filepath).name}")
            self.analyze_image(filepath)

    def analyze_image(self, image_path):
        try:
            # Limpiar visualizaciones anteriores
            self.current_images = []
            self.current_image_index = 0
            self.photo_references.clear()
            gc.collect()

            # Limpiar directorio de visualizaciones
            viz_dir = Path("visualization_results")
            if viz_dir.exists():
                for file in viz_dir.glob("*.png"):
                    try:
                        file.unlink()
                    except:
                        pass

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
                
                # Generar visualización combinada de todas las coordenadas
                self.log_message("\nGenerando visualización combinada...")
                self.combined_visualizer.visualize_combined_results(
                    image=image,
                    results=results,
                    save=True
                )
                
                # Generar visualización específica de Coord1 y Coord2
                self.log_message("\nGenerando visualización de Coord1 y Coord2...")
                self.combined_visualizer.visualize_coord1_coord2(
                    image=image,
                    results=results,
                    save=True
                )

                # Recolectar todas las imágenes generadas
                self.collect_result_images()
                self.show_current_image()
            else:
                self.log_message("No se obtuvieron resultados para analizar")

            self.log_message("Análisis completado")

        except Exception as e:
            self.log_message(f"Error en análisis: {str(e)}")

    def collect_result_images(self):
        # Recolectar todas las imágenes de resultados en orden específico
        self.current_images = []
        
        # 1. Primero Coord1 y Coord2
        if Path("visualization_results/coord1_coord2_results.png").exists():
            self.current_images.append({
                'path': "visualization_results/coord1_coord2_results.png",
                'title': "Resultados Coord1 y Coord2"
            })

        # 2. Visualización combinada
        if Path("visualization_results/combined_results.png").exists():
            self.current_images.append({
                'path': "visualization_results/combined_results.png",
                'title': "Resultados Combinados"
            })

        # 3. Resultados individuales
        for i in range(1, 16):
            coord_name = f"Coord{i}"
            result_path = f"visualization_results/{coord_name}_results.png"
            if Path(result_path).exists():
                self.current_images.append({
                    'path': result_path,
                    'title': f"Resultados {coord_name}"
                })

        self.current_image_index = 0
        self.update_image_counter()

    def show_current_image(self):
        # Limpiar frame actual
        for widget in self.current_image_frame.winfo_children():
            widget.destroy()

        if not self.current_images:
            ttk.Label(self.current_image_frame, text="No hay imágenes disponibles").pack()
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
            self.photo_references = [photo]  # Solo mantener la referencia actual
            
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
        if self.current_images:
            self.current_image_index = (self.current_image_index - 1) % len(self.current_images)
            self.show_current_image()

    def next_image(self):
        if self.current_images:
            self.current_image_index = (self.current_image_index + 1) % len(self.current_images)
            self.show_current_image()

    def update_image_counter(self):
        if self.current_images:
            self.image_counter.config(
                text=f"Imagen {self.current_image_index + 1} de {len(self.current_images)}"
            )
        else:
            self.image_counter.config(text="0/0")

    def log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()

def main():
    root = tk.Tk()
    app = PulmoAlignViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
