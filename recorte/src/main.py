"""
Script principal para el proceso de recorte de imágenes pulmonares.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from coordinates.coordinate_manager import CoordinateManager
from image_processing.image_processor import ImageProcessor

def process_images(coord_manager: CoordinateManager,
                  image_processor: ImageProcessor,
                  coordinates_file: str,
                  indices_file: str) -> Dict:
    """
    Procesa las imágenes y extrae las regiones de interés.
    Todas las coordenadas se manejan en sistema 0-based (0-63).
    
    Args:
        coord_manager: Gestor de coordenadas
        image_processor: Procesador de imágenes
        coordinates_file: Ruta al archivo de coordenadas
        indices_file: Ruta al archivo de índices
        
    Returns:
        Dict: Resultados del procesamiento
    """
    results = {coord_name: {'processed': 0, 'failed': 0, 'errors': []} 
              for coord_name in coord_manager.coord_data.keys()}
    
    try:
        print("\nIniciando procesamiento de imágenes...")
        print(f"Leyendo coordenadas desde: {coordinates_file}")
        coord_manager.read_coordinates(coordinates_file)
        
        total_images = len(coord_manager.coordinates)
        print(f"\nTotal de imágenes a procesar: {total_images}")
        
        # Procesar cada coordenada
        for coord_name, config in coord_manager.coord_data.items():
            print(f"\nProcesando {coord_name}...")
            
            # Calcular centro e intersección
            center_x, center_y = coord_manager.calculate_center(
                config["sup"], config["inf"],
                config["left"], config["right"]
            )
            intersection_x, intersection_y = coord_manager.calculate_intersection(
                config["sup"], config["inf"],
                config["left"], config["right"]
            )

            print(f"Centro: ({center_x}, {center_y})")
            print(f"Intersección: ({intersection_x}, {intersection_y})")
            print(f"Dimensiones: {config['width']}x{config['height']}")

            # Procesar cada imagen
            for index, coords in coord_manager.coordinates.items():
                try:
                    # Obtener ruta de la imagen
                    image_path = image_processor.get_image_path(index, indices_file)
                    
                    # Cargar y redimensionar imagen
                    image = image_processor.load_and_resize_image(image_path)
                    
                    # Obtener coordenadas 
                    new_x, new_y = coords[coord_name]
                    
                    # Crear región de búsqueda
                    search_region = np.zeros((64, 64))
                    search_coords = coord_manager.get_search_coordinates(coord_name.lower())
                    for y, x in search_coords:
                        search_region[y, x] = 1  
                    
                    # Extraer región usando template
                    print(f"\nExtrayendo región para imagen {index}:")
                    print(f"Punto etiquetado: ({new_x}, {new_y})")
                    
                    try:
                        cropped_image = image_processor.extract_region(
                            image=image,
                            search_region=search_region,
                            labeled_point=(new_x, new_y),  
                            coord_num=int(coord_name.replace("Coord", "")),
                            template_size=config["width"]
                        )
                        print("Región extraída exitosamente")
                    except Exception as e:
                        print(f"Error en extracción: {str(e)}")
                        raise
                    
                    # Guardar imagen recortada
                    success = image_processor.save_cropped_image(
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
                        print(f"Procesadas: {index}/{total_images}")
                    
                except Exception as e:
                    print(f"\nError procesando imagen {index} para {coord_name}: {str(e)}")
                    results[coord_name]['failed'] += 1
                    results[coord_name]['errors'].append(f"Error en imagen {index}: {str(e)}")

            # Mostrar resultados para esta coordenada
            print(f"\nResultados para {coord_name}:")
            print(f"  Procesadas exitosamente: {results[coord_name]['processed']}")
            print(f"  Fallidas: {results[coord_name]['failed']}")
            if results[coord_name]['errors']:
                print("\nErrores encontrados:")
                for error in results[coord_name]['errors'][:5]:
                    print(f"  - {error}")
                if len(results[coord_name]['errors']) > 5:
                    print(f"  ... y {len(results[coord_name]['errors']) - 5} errores más")

    except Exception as e:
        print(f"\nError durante la ejecución: {str(e)}")
        
    return results

def main():
    """Función principal del script."""
    try:
        # Configurar rutas
        PROJECT_ROOT = Path(__file__).parent.parent
        DATASET_PATH = PROJECT_ROOT.parent / "COVID-19_Radiography_Dataset"
        COORDINATES_FILE = PROJECT_ROOT.parent / "coordenadas/coordenadas.csv"
        INDICES_FILE = PROJECT_ROOT.parent / "indices/indices.csv"
        VISUALIZATION_DIR = PROJECT_ROOT.parent / "resultados/recorte/imagenes_recortadas/sahs"
        
        # Validar rutas
        if not DATASET_PATH.exists():
            raise FileNotFoundError(f"No se encontró el directorio del dataset: {DATASET_PATH}")
        if not COORDINATES_FILE.exists():
            raise FileNotFoundError(f"No se encontró el archivo de coordenadas: {COORDINATES_FILE}")
        if not INDICES_FILE.exists():
            raise FileNotFoundError(f"No se encontró el archivo de índices: {INDICES_FILE}")
            
        print(f"Usando dataset en: {DATASET_PATH}")
        print(f"Archivo de coordenadas: {COORDINATES_FILE}")
        print(f"Archivo de índices: {INDICES_FILE}")

        # Inicializar componentes
        coord_manager = CoordinateManager()
        image_processor = ImageProcessor(
            base_path=str(DATASET_PATH),
            visualization_dir=str(VISUALIZATION_DIR)
        )
        
        # Cargar coordenadas de búsqueda 
        coord_manager.read_search_coordinates(str(PROJECT_ROOT.parent / "resultados/region_busqueda/json/all_search_coordinates.json"))

        # Procesar imágenes
        results = process_images(
            coord_manager=coord_manager,
            image_processor=image_processor,
            coordinates_file=str(COORDINATES_FILE),
            indices_file=str(INDICES_FILE)
        )
        
        # Mostrar resumen final
        print("\nResumen final:")
        total_processed = sum(result['processed'] for result in results.values())
        total_failed = sum(result['failed'] for result in results.values())
        print(f"Total de imágenes procesadas exitosamente: {total_processed}")
        print(f"Total de imágenes fallidas: {total_failed}")
        print(f"Total de operaciones: {total_processed + total_failed}")
        
        # Mostrar coordenadas con errores
        coords_with_errors = [coord for coord, result in results.items() if result['failed'] > 0]
        if coords_with_errors:
            print("\nCoordenadas con errores:")
            for coord in coords_with_errors:
                print(f"  - {coord}: {results[coord]['failed']} errores")
        
    except Exception as e:
        print(f"\nError durante la ejecución: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
