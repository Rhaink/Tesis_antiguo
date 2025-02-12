#!/usr/bin/env python3
"""
Script para predecir puntos anatómicos en nuevas imágenes.
"""

import os
import sys
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple
import json

# Configurar rutas base
SCRIPT_DIR = Path(__file__).parent  # scripts/
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Tesis/
sys.path.append(str(SCRIPT_DIR.parent))  # prediccion/

from src.pca_analyzer import PCAAnalyzer
from src.coordinate_manager import CoordinateManager
from src.image_processor import ImageProcessor
from src.visualizer import Visualizer
from src.combined_visualizer import CombinedVisualizer
from src.template_processor import TemplateProcessor

def setup_logging(log_file: str = "prediction.log"):
    """Configura el sistema de logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_models(models_dir: str) -> Dict[str, PCAAnalyzer]:
    """
    Carga los modelos PCA entrenados.
    
    Args:
        models_dir: Directorio con los modelos
        
    Returns:
        Diccionario de modelos PCA por punto de interés
    """
    models = {}
    models_path = Path(models_dir)
    
    if not models_path.exists():
        raise FileNotFoundError(f"No se encontró el directorio {models_dir}")
    
    for model_file in models_path.glob("*_model.pkl"):
        coord_name = model_file.stem.replace("_model", "")
        try:
            model = PCAAnalyzer(model_path=str(model_file))
            models[coord_name] = model
            logging.info(f"Modelo cargado: {coord_name}")
        except Exception as e:
            logging.error(f"Error cargando modelo {coord_name}: {str(e)}")
    
    if not models:
        raise ValueError("No se encontraron modelos válidos")
    
    return models

def predict_points(image_path: str,
                  models: Dict[str, PCAAnalyzer],
                  coord_manager: CoordinateManager,
                  image_processor: ImageProcessor,
                  template_processor: TemplateProcessor,
                  visualizer: Visualizer,
                  combined_visualizer: CombinedVisualizer) -> Dict:
    """
    Predice puntos anatómicos en una imagen.
    
    El proceso para cada punto anatómico:
    1. Carga los datos del template y región de búsqueda
    2. Para cada coordenada en la región de búsqueda:
       - Mueve el template completo usando el punto de intersección como ancla
       - Extrae la región usando las dimensiones originales del template
       - Calcula el error de reconstrucción PCA
    3. Encuentra el punto donde el error de reconstrucción es mínimo
    4. Visualiza los resultados
    
    Args:
        image_path: Ruta de la imagen
        models: Diccionario de modelos PCA
        coord_manager: Gestor de coordenadas
        image_processor: Procesador de imágenes
        template_processor: Procesador de templates
        visualizer: Visualizador de resultados
        combined_visualizer: Visualizador de resultados combinados
        
    Returns:
        Diccionario con resultados de predicción
    """
    logging.info(f"\nAnalizando imagen: {image_path}")
    
    # Cargar y procesar imagen
    image = image_processor.load_and_resize_image(image_path)
    results = {}
    
    # Analizar cada punto
    for coord_name, model in models.items():
        try:
            logging.info(f"\nAnalizando {coord_name}...")
            
            # Obtener datos del template y región
            template_data = template_processor.load_template_data(coord_name)
            if template_data is None:
                raise ValueError(f"No se encontraron datos del template para {coord_name}")
            
            # Obtener coordenadas de búsqueda
            search_coordinates = coord_manager.get_search_coordinates(coord_name)
            logging.info(f"Coordenadas de búsqueda: {len(search_coordinates)}")
            
            # Obtener dimensiones del template para logging
            template_bounds = template_data["template_bounds"]
            logging.info(f"Dimensiones del template: {template_bounds['width']}x{template_bounds['height']}")
            
            # Obtener punto de intersección para logging
            intersection = template_data["intersection_point"]
            logging.info(f"Punto de intersección: ({intersection['x']}, {intersection['y']})")
            
            # Analizar región de búsqueda
            min_error, min_error_coords, errors = model.analyze_search_region(
                image=image,
                search_coordinates=search_coordinates,
                template_processor=template_processor,
                template_data=template_data
            )
            
            min_error_step = errors.index(min(errors)) + 1
            
            logging.info(f"Error mínimo: {min_error:.4f}")
            logging.info(f"Coordenadas óptimas: {min_error_coords}")
            logging.info(f"Paso: {min_error_step} de {len(search_coordinates)}")
            
            # Guardar resultados
            results[coord_name] = {
                'min_error': float(min_error),
                'min_error_coords': min_error_coords,
                'min_error_step': min_error_step,
                'search_coordinates': search_coordinates,
                'template_bounds': template_bounds,
                'intersection_point': intersection,
                'region_bounds': template_data["region_bounds"]
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
            logging.error(f"Error procesando {coord_name}: {str(e)}")
            continue
    
    if results:
        # Visualizaciones individuales
        visualizer.visualize_results(
            image=image,
            coord_config=coord_manager.template_data,  # Usar template_data en lugar de coord_data
            results=results,
            pca_models=models,
            save=True
        )
        
        # Visualización combinada
        combined_visualizer.visualize_combined_results(
            image=image,
            results=results,
            save=True
        )
        
        # Visualización específica de Coord1 y Coord2
        combined_visualizer.visualize_coord1_coord2(
            image=image,
            results=results,
            save=True
        )
    
    return results

def save_results(results: Dict, output_file: str):
    """
    Guarda los resultados en formato JSON.
    
    Args:
        results: Diccionario con resultados
        output_file: Ruta del archivo de salida
    """
    # Convertir tuplas a listas para serialización JSON
    serializable_results = {}
    for coord_name, result in results.items():
        serializable_results[coord_name] = {
            'min_error': result['min_error'],
            'min_error_coords': list(result['min_error_coords']),
            'min_error_step': result['min_error_step']
        }
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description="Predicción de puntos anatómicos")
    
    parser.add_argument(
        "--image_path",
        type=str,
        default=str(PROJECT_ROOT / "COVID-19_Radiography_Dataset/Normal/images/Normal-3.png"),
        help="Ruta de la imagen a analizar"
    )
    
    parser.add_argument(
        "--coord_file",
        type=str,
        default=str(PROJECT_ROOT / "tools/template_analysis/template_analysis_results.json"),
        help="Ruta al archivo de análisis de templates"
    )
    
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directorio con modelos entrenados"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="prediction_results",
        help="Directorio para resultados"
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging()
    
    try:
        # Crear directorio de salida
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar componentes
        coord_manager = CoordinateManager()
        coord_manager.read_search_coordinates(args.coord_file)
        
        template_processor = TemplateProcessor(args.coord_file)
        
        image_processor = ImageProcessor(
            base_path=".",
            template_data_path=args.coord_file,
            output_dir=str(output_dir)
        )
        
        visualizer = Visualizer(output_dir=str(output_dir))
        combined_visualizer = CombinedVisualizer(output_dir=str(output_dir))
        
        # Cargar modelos
        logging.info("Cargando modelos...")
        models = load_models(args.models_dir)
        
        # Predecir puntos
        results = predict_points(
            image_path=args.image_path,
            models=models,
            coord_manager=coord_manager,
            image_processor=image_processor,
            template_processor=template_processor,
            visualizer=visualizer,
            combined_visualizer=combined_visualizer
        )
        
        # Guardar resultados
        results_file = output_dir / "results.json"
        save_results(results, str(results_file))
        
        logging.info(f"\nResultados guardados en: {results_file}")
        logging.info("\nProceso completado exitosamente")
        
    except Exception as e:
        logging.error(f"\nError en predicción: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
