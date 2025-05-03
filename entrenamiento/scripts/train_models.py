#!/usr/bin/env python3
"""
Script para entrenar modelos PCA o Kernel PCA para cada punto anatómico.
"""

import os
import sys
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Optional



# Obtener la ruta absoluta del directorio del script
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
# Ruta base del proyecto
BASE_DIR = Path("/home/donrobot/projects")
PROJECT_ROOT = BASE_DIR / "Tesis"
# Agregar el directorio de entrenamiento al sys.path
sys.path.append(str(PROJECT_ROOT / "entrenamiento"))

from src.pca_analyzer import PCAAnalyzer

# Definir rutas absolutas para archivos y directorios
DATASET_PATH = PROJECT_ROOT / "COVID-19_Radiography_Dataset"
COORD_FILE = PROJECT_ROOT / "resultados/analisis_regiones/dataset_aligned_maestro_1/analisis/template_analysis_results.json"
MODELS_DIR = PROJECT_ROOT / "resultados/entrenamiento/dataset_aligned_maestro_1/models"
OUTPUT_DIR = PROJECT_ROOT / "resultados/entrenamiento/dataset_aligned_maestro_1/visualization_results"
LOG_FILE = PROJECT_ROOT / "resultados/entrenamiento/dataset_aligned_maestro_1/training.log"

from src.pca_analyzer import PCAAnalyzer
from src.coordinate_manager import CoordinateManager
from src.image_processor import ImageProcessor
from src.visualizer import Visualizer

def setup_logging(log_file: str = str(LOG_FILE)):
    """Configura el sistema de logging."""
    # Crear el directorio del archivo de log si no existe
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

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
                 output_dir: str,
                 use_kernel: bool = False,
                 kernel_type: str = 'rbf',
                 kernel_params: Optional[Dict] = None) -> Dict:
    """
    Entrena modelos PCA o Kernel PCA para cada punto anatómico.
    
    Args:
        base_path: Ruta base del dataset
        coord_file: Archivo con coordenadas de búsqueda
        models_dir: Directorio para guardar modelos
        output_dir: Directorio para visualizaciones
        use_kernel: Si se debe usar Kernel PCA en lugar de PCA lineal
        kernel_type: Tipo de kernel ('rbf', 'poly', 'sigmoid')
        kernel_params: Parámetros adicionales para el kernel
    
    Returns:
        Dict: Información de entrenamiento para cada punto anatómico
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

    # Preparar parámetros del kernel
    if kernel_params is None:
        kernel_params = {}
        
    # Para kernel RBF, configurar gamma por defecto
    if kernel_type == 'rbf' and 'gamma' not in kernel_params:
        kernel_params['gamma'] = 0.1  # Valor inicial, ajustar según resultados

    # Cargar coordenadas
    logging.info("Cargando coordenadas de búsqueda...")
    coord_manager.read_search_coordinates(coord_file)

    training_info = {}

    # Mostrar información sobre el tipo de modelo
    if use_kernel:
        logging.info(f"Entrenando modelos Kernel PCA con kernel '{kernel_type}'")
        for param, value in kernel_params.items():
            logging.info(f"  Parámetro {param}: {value}")
    else:
        logging.info("Entrenando modelos PCA lineal estándar")

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

            # Crear y entrenar modelo PCA o Kernel PCA
            pca = PCAAnalyzer(use_kernel=use_kernel)
            
            if use_kernel:
                pca.train(
                    training_images, 
                    kernel=kernel_type, 
                    kernel_params=kernel_params
                )
            else:
                pca.train(training_images)

            # Guardar modelo
            model_suffix = "_kernel" if use_kernel else ""
            model_path = Path(models_dir) / f"{coord_name}{model_suffix}_model.pkl"
            pca.save_model(str(model_path))

            # Obtener información del modelo
            model_info = pca.get_model_info()
            training_info[coord_name] = model_info

            # Mostrar información del modelo
            logging.info(f"Componentes: {model_info['n_components']}")
            if 'explained_variance_ratio' in model_info:
                logging.info(f"Varianza explicada: {model_info['explained_variance_ratio']:.4f}")
                
            if use_kernel:
                logging.info(f"Kernel: {model_info.get('kernel_type', 'desconocido')}")

            # Visualizar eigenfaces con nombre específico por coordenada
            visualizer.plot_eigenfaces(
                eigenfaces=pca.eigenfaces,
                mean_face=pca.mean_face,
                save=True,
                filename=f"{coord_name}{model_suffix}_eigenfaces.png"
            )

        except Exception as e:
            logging.error(f"Error entrenando {coord_name}: {str(e)}")
            continue

    return training_info

def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos PCA/Kernel PCA")

    parser.add_argument(
        "--base_path",
        type=str,
        default=str(DATASET_PATH),
        help="Ruta base del proyecto"
    )

    parser.add_argument(
        "--coord_file",
        type=str,
        default=str(COORD_FILE),
        help="Ruta al archivo de análisis de templates"
    )

    parser.add_argument(
        "--models_dir",
        type=str,
        default=str(MODELS_DIR),
        help="Directorio para guardar modelos"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directorio para visualizaciones"
    )
    
    parser.add_argument(
        "--use_kernel",
        action="store_true",
        help="Usar Kernel PCA en lugar de PCA lineal"
    )
    
    parser.add_argument(
        "--kernel",
        type=str,
        choices=["rbf", "poly", "sigmoid", "linear"],
        default="rbf",
        help="Tipo de kernel para Kernel PCA"
    )
    
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Parámetro gamma para kernel RBF o Sigmoid"
    )
    
    parser.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Grado del polinomio para kernel Poly"
    )

    args = parser.parse_args()

    # Configurar logging
    setup_logging()

    # Preparar parámetros del kernel
    kernel_params = {}
    if args.kernel == "rbf" or args.kernel == "sigmoid":
        kernel_params["gamma"] = args.gamma
    elif args.kernel == "poly":
        kernel_params["degree"] = args.degree
        kernel_params["gamma"] = args.gamma

    # Entrenar modelos
    try:
        logging.info("Iniciando entrenamiento de modelos...")
        training_info = train_models(
            base_path=args.base_path,
            coord_file=args.coord_file,
            models_dir=args.models_dir,
            output_dir=args.output_dir,
            use_kernel=args.use_kernel,
            kernel_type=args.kernel,
            kernel_params=kernel_params
        )

        logging.info("\nEntrenamiento completado exitosamente")
        logging.info("\nResumen de entrenamiento:")
        
        # Mostrar información específica según el tipo de modelo
        for coord_name, info in training_info.items():
            logging.info(f"\n{coord_name}:")
            logging.info(f"  Componentes: {info['n_components']}")
            
            if 'explained_variance_ratio' in info:
                logging.info(f"  Varianza explicada: {info['explained_variance_ratio']:.4f}")
                
            if args.use_kernel:
                logging.info(f"  Tipo de kernel: {info.get('kernel_type', 'desconocido')}")

    except Exception as e:
        logging.error(f"\nError en entrenamiento: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()