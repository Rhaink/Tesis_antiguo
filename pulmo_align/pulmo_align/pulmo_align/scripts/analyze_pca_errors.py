"""
Script para el análisis PCA y cálculo de errores en imágenes pulmonares.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import tkinter as tk
from tkinter import filedialog

# Configuración de rutas relativas al directorio del script
SCRIPT_DIR = Path(__file__).parent.parent.parent  # pulmo_align/
SCRIPT_DIR_2 = Path(__file__).parent  # Tesis/
DATASET_PATH = SCRIPT_DIR_2 / "COVID-19_Radiography_Dataset"
INDICES_PATH = SCRIPT_DIR / "indices.csv"
COORDINATES_PATH = SCRIPT_DIR / "all_search_coordinates.json"

# Agregar el directorio raíz al path para importar el paquete
sys.path.append(str(SCRIPT_DIR))

from pulmo_align.coordinates.coordinate_manager import CoordinateManager
from pulmo_align.image_processing.image_processor import ImageProcessor
from pulmo_align.pca_analysis.pca_analyzer import PCAAnalyzer
from pulmo_align.visualization.visualizer import Visualizer
from pulmo_align.visualization.combined_visualizer import CombinedVisualizer

def validate_paths():
    """
    Valida que existan las rutas necesarias para el programa.
    
    Raises:
        FileNotFoundError: Si no se encuentran las rutas requeridas
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No se encontró el directorio del dataset en: {DATASET_PATH}")
    if not INDICES_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de índices en: {INDICES_PATH}")
    if not COORDINATES_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de coordenadas en: {COORDINATES_PATH}")

def initialize_pca_models(coord_manager: CoordinateManager,
                         image_processor: ImageProcessor) -> Dict:
    """
    Inicializa los modelos PCA para cada coordenada.
    
    Args:
        coord_manager (CoordinateManager): Gestor de coordenadas
        image_processor (ImageProcessor): Procesador de imágenes
        
    Returns:
        Dict: Modelos PCA inicializados
    """
    print("\nInicializando modelos PCA...")
    pca_models = {}

    for coord_name, config in coord_manager.coord_data.items():
        print(f"\nProcesando {coord_name}...")
        try:
            # Cargar imágenes de entrenamiento
            training_images = image_processor.load_training_images(
                coord_name=coord_name,
                target_size=(config['width'], config['height'])
            )
            
            if training_images:
                print(f"Imágenes de entrenamiento cargadas: {len(training_images)}")
                
                # Inicializar y entrenar PCA
                pca = PCAAnalyzer()
                pca.train(training_images)
                pca_models[coord_name] = pca
                
                # Mostrar información del modelo
                model_info = pca.get_model_info()
                print(f"PCA: {model_info['n_components']} componentes, "
                      f"varianza: {model_info['explained_variance_ratio']:.2%}")
            else:
                print(f"No hay imágenes de entrenamiento para {coord_name}")
                
        except Exception as e:
            print(f"Error en {coord_name}: {str(e)}")
            continue

    if not pca_models:
        raise ValueError("No se pudo inicializar ningún modelo PCA")
        
    return pca_models

def process_image(image_path: str,
                 coord_manager: CoordinateManager,
                 image_processor: ImageProcessor,
                 pca_models: Dict,
                 visualizer: Visualizer,
                 combined_visualizer: CombinedVisualizer) -> None:
    """
    Procesa una imagen de prueba y visualiza los resultados.
    
    Args:
        image_path (str): Ruta a la imagen de prueba
        coord_manager (CoordinateManager): Gestor de coordenadas
        image_processor (ImageProcessor): Procesador de imágenes
        pca_models (Dict): Modelos PCA
        visualizer (Visualizer): Visualizador de resultados
        combined_visualizer (CombinedVisualizer): Visualizador de resultados combinados
    """
    try:
        print(f"\nProcesando imagen: {image_path}")
        
        # Cargar y redimensionar imagen
        image = image_processor.load_and_resize_image(image_path)
        results = {}
        
        # Procesar cada coordenada
        for coord_name, config in coord_manager.coord_data.items():
            if coord_name not in pca_models:
                print(f"Saltando {coord_name}: no hay modelo PCA")
                continue
                
            try:
                print(f"\nAnalizando {coord_name}...")
                
                # Obtener coordenadas de búsqueda
                search_coordinates = coord_manager.get_search_coordinates(coord_name)
                print(f"Coordenadas de búsqueda: {len(search_coordinates)}")
                
                # Analizar región de búsqueda
                min_error, min_error_coords, errors = pca_models[coord_name].analyze_search_region(
                    image=image,
                    search_coordinates=search_coordinates,
                    template_width=config['width'],
                    template_height=config['height'],
                    intersection_x=config['left'],
                    intersection_y=config['sup']
                )

                min_error_step = errors.index(min(errors)) + 1

                print(f"Error mínimo: {min_error:.4f} en coordenadas: {min_error_coords}")
                print(f"Encontrado en el paso: {min_error_step}")

                results[coord_name] = {
                    'min_error': min_error,
                    'min_error_coords': min_error_coords,
                    'min_error_step': min_error_step,
                    'errors': errors,
                    'search_coordinates': search_coordinates
                }
                
                # Visualizar distribución de errores
                visualizer.plot_error_distribution(
                    errors=errors,
                    coord_name=coord_name,
                    save=True
                )
                
                # Visualizar camino de búsqueda
                visualizer.plot_search_path(
                    search_coordinates=search_coordinates,
                    min_error_coords=min_error_coords,
                    coord_name=coord_name,
                    save=True
                )
                
            except Exception as e:
                print(f"Error procesando {coord_name}: {str(e)}")
                continue

        # Visualizar resultados finales
        if results:
            print("\nGenerando visualizaciones...")
            visualizer.visualize_results(
                image=image,
                coord_config=coord_manager.coord_data,
                results=results,
                pca_models=pca_models,
                save=True
            )
            
            # Generar visualización combinada de todas las coordenadas
            print("\nGenerando visualización combinada de todas las coordenadas...")
            combined_visualizer.visualize_combined_results(
                image=image,
                results=results,
                save=True
            )
            
            # Generar visualización específica de Coord1 y Coord2
            print("\nGenerando visualización de Coord1 y Coord2...")
            combined_visualizer.visualize_coord1_coord2(
                image=image,
                results=results,
                save=True
            )
        else:
            print("No se obtuvieron resultados para analizar")

    except Exception as e:
        print(f"Error procesando imagen: {str(e)}")

def select_test_image() -> str:
    """
    Abre un diálogo para seleccionar la imagen de prueba.
    
    Returns:
        str: Ruta de la imagen seleccionada
    """
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de tkinter
    
    file_path = filedialog.askopenfilename(
        title='Seleccionar imagen de prueba',
        filetypes=[
            ('Imágenes', '*.png;*.jpg;*.jpeg'),
            ('Todos los archivos', '*.*')
        ],
        initialdir=str(DATASET_PATH)
    )
    
    if not file_path:
        raise ValueError("No se seleccionó ninguna imagen")
        
    return file_path

def main():
    """Función principal del script."""
    try:
        print("Iniciando programa...")
        
        # Validar rutas
        validate_paths()
        
        # Inicializar componentes
        coord_manager = CoordinateManager()
        image_processor = ImageProcessor(base_path=str(DATASET_PATH))
        visualizer = Visualizer()
        combined_visualizer = CombinedVisualizer()
        
        # Cargar coordenadas de búsqueda
        print("Cargando coordenadas de búsqueda...")
        coord_manager.read_search_coordinates(str(COORDINATES_PATH))
        
        # Inicializar modelos PCA
        pca_models = initialize_pca_models(coord_manager, image_processor)

        # Solicitar al usuario que seleccione la imagen de prueba
        try:
            test_image_path = select_test_image()
            print(f"Imagen seleccionada: {test_image_path}")
            
            process_image(
                image_path=test_image_path,
                coord_manager=coord_manager,
                image_processor=image_processor,
                pca_models=pca_models,
                visualizer=visualizer,
                combined_visualizer=combined_visualizer
            )
        except ValueError as e:
            print(f"\nError: {str(e)}")
        except Exception as e:
            print(f"\nError procesando la imagen seleccionada: {str(e)}")

    except Exception as e:
        print(f"\nError en la ejecución: {str(e)}")
    
    print("\nPrograma finalizado")

if __name__ == "__main__":
    main()
