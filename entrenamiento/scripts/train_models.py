#!/usr/bin/env python3
"""
Script para entrenar modelos PCA para cada punto anatómico.
"""

import os
import sys
from pathlib import Path
import argparse
import logging
from typing import Dict

# Configurar rutas base
BASE_DIR = Path("/home/donrobot/projects")
PROJECT_ROOT = BASE_DIR / "Tesis"
sys.path.append(str(PROJECT_ROOT / "entrenamiento"))

from src.pca_analyzer import PCAAnalyzer
from src.coordinate_manager import CoordinateManager
from src.image_processor import ImageProcessor
from src.visualizer import Visualizer

def setup_logging(log_file: str = "training.log"):
    """Configura el sistema de logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train_models(base_path: str,
                coord_file: str,
                models_dir: str,
                output_dir: str) -> Dict:
    """
    Entrena modelos PCA para cada punto anatómico.
    
    Args:
        base_path: Ruta base del proyecto
        coord_file: Ruta al archivo de coordenadas
        models_dir: Directorio para guardar modelos
        output_dir: Directorio para visualizaciones
        
    Returns:
        Diccionario con información de entrenamiento
    """
    # Crear directorios necesarios
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Inicializar componentes
    coord_manager = CoordinateManager()
    image_processor = ImageProcessor(
        base_path=base_path,
        template_data_path=coord_file,
        output_dir=output_dir
    )
    visualizer = Visualizer(output_dir=output_dir)
    
    # Cargar coordenadas
    logging.info("Cargando coordenadas de búsqueda...")
    coord_manager.read_search_coordinates(coord_file)
    
    training_info = {}
    
    # Entrenar modelo para cada punto
    for coord_name in coord_manager.get_all_coordinate_names():
        logging.info(f"\nEntrenando modelo para {coord_name}...")
        
        try:
            # Cargar imágenes de entrenamiento
            logging.info("Cargando imágenes de entrenamiento...")
            training_images = image_processor.load_training_images(
                coord_name=coord_name
            )
            
            logging.info(f"Imágenes cargadas: {len(training_images)}")
            
            # Crear y entrenar modelo PCA
            pca = PCAAnalyzer()
            pca.train(training_images)
            
            # Guardar modelo
            model_path = Path(models_dir) / f"{coord_name}_model.pkl"
            pca.save_model(str(model_path))
            
            # Obtener información del modelo
            model_info = pca.get_model_info()
            training_info[coord_name] = model_info
            
            logging.info(f"Componentes PCA: {model_info['n_components']}")
            logging.info(f"Varianza explicada: {model_info['explained_variance_ratio']:.4f}")
            
            # Visualizar eigenfaces con nombre específico por coordenada
            visualizer.plot_eigenfaces(
                eigenfaces=pca.eigenfaces,
                mean_face=pca.mean_face,
                save=True,
                filename=f"{coord_name}_eigenfaces.png"
            )
            
        except Exception as e:
            logging.error(f"Error entrenando {coord_name}: {str(e)}")
            continue
    
    return training_info

def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos PCA")
    
    parser.add_argument(
        "--base_path",
        type=str,
        default=str(PROJECT_ROOT / "COVID-19_Radiography_Dataset"),
        help="Ruta base del proyecto"
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
        default=str(PROJECT_ROOT / "models"),
        help="Directorio para guardar modelos"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "visualization_results"),
        help="Directorio para visualizaciones"
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging()
    
    # Entrenar modelos
    try:
        logging.info("Iniciando entrenamiento de modelos...")
        training_info = train_models(
            base_path=args.base_path,
            coord_file=args.coord_file,
            models_dir=args.models_dir,
            output_dir=args.output_dir
        )
        
        logging.info("\nEntrenamiento completado exitosamente")
        logging.info("\nResumen de entrenamiento:")
        for coord_name, info in training_info.items():
            logging.info(f"\n{coord_name}:")
            logging.info(f"  Componentes: {info['n_components']}")
            logging.info(f"  Varianza explicada: {info['explained_variance_ratio']:.4f}")
        
    except Exception as e:
        logging.error(f"\nError en entrenamiento: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
