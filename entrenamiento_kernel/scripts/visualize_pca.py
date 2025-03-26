#!/usr/bin/env python3
"""
Script para visualizar los componentes principales de las imágenes de entrenamiento.
"""

import os
import sys
from pathlib import Path
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

# Configurar rutas base
BASE_DIR = Path("/home/donrobot/projects")
PROJECT_ROOT = BASE_DIR / "Tesis"
sys.path.append(str(PROJECT_ROOT / "entrenamiento"))

from src.pca_analyzer import PCAAnalyzer
from src.image_processor import ImageProcessor
from src.visualizer import Visualizer

def setup_logging(log_file: str = "pca_visualization.log"):
    """Configura el sistema de logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def plot_cumulative_variance(pca_analyzer, output_path):
    """
    Genera gráfico de varianza explicada acumulada.
    """
    cumulative_variance = np.cumsum(pca_analyzer.pca.explained_variance_ratio_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.title('Varianza Explicada Acumulada vs Número de Componentes')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_reconstruction_examples(pca_analyzer, original_images, output_dir, n_examples=5):
    """
    Genera visualizaciones de reconstrucciones usando diferente número de componentes.
    Muestra las imágenes en un formato de 2 filas x 3 columnas.
    """
    n_components_list = [1, 5, 10, 20, pca_analyzer.n_components]
    n_examples = min(n_examples, len(original_images))
    
    for idx in range(n_examples):
        plt.figure(figsize=(12, 8))
        
        # Primera fila
        # Imagen original
        plt.subplot(2, 3, 1)
        plt.imshow(original_images[idx], cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        # Primeros dos componentes de reconstrucción
        for i in range(2):
            n_comp = n_components_list[i]
            omega = pca_analyzer.pca.transform(original_images[idx].reshape(1, -1))
            omega[0, n_comp:] = 0
            reconstructed = pca_analyzer.pca.inverse_transform(omega)
            reconstructed = reconstructed.reshape(original_images[idx].shape)
            
            plt.subplot(2, 3, i + 2)
            plt.imshow(reconstructed, cmap='gray')
            plt.title(f'{n_comp} comp.')
            plt.axis('off')
        
        # Segunda fila
        # Últimos tres componentes de reconstrucción
        for i in range(3):
            n_comp = n_components_list[i + 2]
            omega = pca_analyzer.pca.transform(original_images[idx].reshape(1, -1))
            omega[0, n_comp:] = 0
            reconstructed = pca_analyzer.pca.inverse_transform(omega)
            reconstructed = reconstructed.reshape(original_images[idx].shape)
            
            plt.subplot(2, 3, i + 4)
            plt.imshow(reconstructed, cmap='gray')
            plt.title(f'{n_comp} comp.')
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'reconstruction_example_{idx+1}.png'))
        plt.close()

def analyze_pca_components(coord_name: str,
                         base_path: str,
                         coord_file: str,
                         output_dir: str):
    """
    Analiza y visualiza los componentes principales para un punto anatómico.
    
    Args:
        coord_name: Nombre del punto anatómico (coord1 o coord2)
        base_path: Ruta base del proyecto
        coord_file: Ruta al archivo de coordenadas
        output_dir: Directorio para guardar visualizaciones
    """
    # Crear directorio de salida
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Inicializar componentes
    image_processor = ImageProcessor(
        base_path=base_path,
        template_data_path=coord_file,
        output_dir=str(output_dir)
    )
    visualizer = Visualizer(output_dir=str(output_dir))
    
    try:
        # Cargar imágenes de entrenamiento
        logging.info(f"Cargando imágenes de entrenamiento para {coord_name}...")
        training_images = image_processor.load_training_images(coord_name)
        logging.info(f"Imágenes cargadas: {len(training_images)}")
        
        # Crear y entrenar modelo PCA
        pca = PCAAnalyzer()
        pca.train(training_images)
        
        # Obtener información del modelo
        model_info = pca.get_model_info()
        logging.info("\nInformación del modelo PCA:")
        logging.info(f"Componentes: {model_info['n_components']}")
        logging.info(f"Varianza explicada: {model_info['explained_variance_ratio']:.4f}")
        
        # Visualizar eigenfaces
        logging.info("\nGenerando visualización de eigenfaces...")
        visualizer.plot_eigenfaces(
            eigenfaces=pca.eigenfaces,
            mean_face=pca.mean_face,
            save=True,
            filename=f"{coord_name}_eigenfaces.png"
        )
        
        # Generar gráfico de varianza acumulada
        logging.info("Generando gráfico de varianza acumulada...")
        plot_cumulative_variance(
            pca,
            output_path=str(output_dir / f"{coord_name}_cumulative_variance.png")
        )
        
        # Generar ejemplos de reconstrucción
        logging.info("Generando ejemplos de reconstrucción...")
        plot_reconstruction_examples(
            pca,
            training_images,
            str(output_dir),
            n_examples=5
        )
        
    except Exception as e:
        logging.error(f"Error durante el análisis: {str(e)}")
        raise

def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Visualización de componentes principales"
    )
    
    parser.add_argument(
        "--coord_name",
        type=str,
        choices=['coord1', 'coord2'],
        required=True,
        help="Nombre del punto anatómico a analizar"
    )
    
    parser.add_argument(
        "--base_path",
        type=str,
        default=str(PROJECT_ROOT / "COVID-19_Radiography_Dataset"),
        help="Ruta base del proyecto"
    )
    
    parser.add_argument(
        "--coord_file",
        type=str,
        default=str(PROJECT_ROOT / "resultados/analisis_regiones/prueba_1/template_analysis_results.json"),
        help="Ruta al archivo de análisis de templates"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "resultados/entrenamiento/prueba_kernel/visualization_results/pca_analysis"),
        help="Directorio para visualizaciones"
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging()
    
    try:
        logging.info(f"\nIniciando análisis PCA para {args.coord_name}...")
        analyze_pca_components(
            coord_name=args.coord_name,
            base_path=args.base_path,
            coord_file=args.coord_file,
            output_dir=args.output_dir
        )
        logging.info("\nAnálisis completado exitosamente")
        
    except Exception as e:
        logging.error(f"\nError en análisis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
