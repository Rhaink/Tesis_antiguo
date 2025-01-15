"""
Script principal para la extracción de imágenes recortadas.

Este script utiliza las clases CoordinateManager e ImageProcessor para:
1. Cargar y procesar coordenadas
2. Extraer regiones de interés de las imágenes
3. Guardar las imágenes recortadas
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Configuración de rutas relativas al directorio del script
SCRIPT_DIR = Path(__file__).parent.parent.parent  # pulmo_align/
DATASET_PATH = SCRIPT_DIR / "COVID-19_Radiography_Dataset"
INDICES_PATH = SCRIPT_DIR / "indices.csv"
COORDINATES_PATH = SCRIPT_DIR / "coordenadas.csv"

# Agregar el directorio raíz al path para importar el paquete
sys.path.append(str(SCRIPT_DIR))

from pulmo_align.coordinates.coordinate_manager import CoordinateManager
from pulmo_align.image_processing.image_processor import ImageProcessor

def process_images(coordinates_file: str, indices_file: str) -> Dict:
    """
    Procesa las imágenes y extrae las regiones de interés.
    
    Args:
        coordinates_file (str): Ruta al archivo de coordenadas
        indices_file (str): Ruta al archivo de índices
        
    Returns:
        Dict: Resultados del procesamiento
    """
    # Inicializar gestores
    coord_manager = CoordinateManager()
    
    print(f"\nUsando dataset en: {DATASET_PATH}")
    
    image_processor = ImageProcessor(base_path=str(DATASET_PATH))
    
    # Resultados para cada coordenada
    results = {coord_name: {'processed': 0, 'failed': 0, 'errors': []} 
              for coord_name in coord_manager.coord_data.keys()}
    
    try:
        print("\nIniciando procesamiento de imágenes...")
        print(f"Leyendo coordenadas desde: {coordinates_file}")
        coord_manager.read_coordinates(coordinates_file)
        
        total_images = len(coord_manager.coordinates)
        print(f"\nTotal de imágenes a procesar: {total_images}")
        
        # Procesamos cada coordenada
        for coord_name, config in coord_manager.coord_data.items():
            print(f"\nProcesando {coord_name}...")
            
            # Calculamos centro e intersección
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

            # Procesamos cada imagen
            for index, coords in coord_manager.coordinates.items():
                try:
                    # Obtener ruta de la imagen
                    image_path = image_processor.get_image_path(index, indices_file)
                    
                    # Cargar y redimensionar imagen
                    image = image_processor.load_and_resize_image(image_path)
                    
                    # Obtener nuevas coordenadas
                    new_x, new_y = coords[coord_name]
                    
                    # Extraer región
                    cropped_image = image_processor.extract_region(
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
                    
                    # Mostrar progreso
                    if index % 10 == 0:
                        print(f"Procesadas: {index}/{total_images}", end='\r')
                    
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
                for error in results[coord_name]['errors'][:5]:  # Mostrar solo los primeros 5 errores
                    print(f"  - {error}")
                if len(results[coord_name]['errors']) > 5:
                    print(f"  ... y {len(results[coord_name]['errors']) - 5} errores más")

    except Exception as e:
        print(f"\nError durante la ejecución: {str(e)}")
        
    return results

def main():
    """Función principal del script."""
    print(f"Usando archivo de coordenadas: {COORDINATES_PATH}")
    print(f"Usando archivo de índices: {INDICES_PATH}")

    try:
        # Procesar imágenes
        results = process_images(str(COORDINATES_PATH), str(INDICES_PATH))
        
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

if __name__ == "__main__":
    main()
