#!/usr/bin/env python3
"""
Programa para predecir y guardar coordenadas de puntos anatómicos en radiografías pulmonares.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging
import sys

# Agregar el directorio de PulmoAlign al path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PULMO_ALIGN_DIR = PROJECT_ROOT / "Tesis/pulmo_align/pulmo_align"
PULMO_ALIGN_DATA = PULMO_ALIGN_DIR
sys.path.append(str(PULMO_ALIGN_DIR))

from pulmo_align.coordinates.coordinate_manager import CoordinateManager
from pulmo_align.image_processing.image_processor import ImageProcessor
from pulmo_align.pca_analysis.pca_analyzer import PCAAnalyzer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('coordinate_prediction.log'),
        logging.StreamHandler()
    ]
)

class CoordinatePredictor:
    """
    Clase para predecir y guardar coordenadas de puntos anatómicos en radiografías.
    """
    
    def __init__(self, project_root: Path):
        """
        Inicializa el predictor de coordenadas.
        
        Args:
            project_root: Ruta raíz del proyecto PulmoAlign
        """
        self.project_root = project_root
        self.coord_manager = CoordinateManager()
        self.image_processor = ImageProcessor(
            base_path=str(project_root / "COVID-19_Radiography_Dataset")
        )
        self.pca_models = {}
        
    def initialize_models(self):
        """
        Inicializa y entrena los modelos PCA para cada punto anatómico.
        """
        logging.info("Iniciando inicialización de modelos PCA...")
        
        # Cargar coordenadas de búsqueda
        search_coords_path = PULMO_ALIGN_DATA / "all_search_coordinates.json"
        self.coord_manager.read_search_coordinates(str(search_coords_path))
        logging.info("Coordenadas de búsqueda cargadas")
        
        # Inicializar modelos PCA
        for coord_name, config in tqdm(self.coord_manager.coord_data.items(), 
                                     desc="Entrenando modelos PCA"):
            try:
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
                    logging.info(f"PCA {coord_name}: {model_info['n_components']} componentes")
                else:
                    logging.warning(f"No se encontraron imágenes de entrenamiento para {coord_name}")
                    
            except Exception as e:
                logging.error(f"Error entrenando modelo para {coord_name}: {str(e)}")
                continue
        
        logging.info(f"Inicialización completada. {len(self.pca_models)} modelos entrenados")

    def predict_coordinates(self, image_path: str) -> Dict[str, Tuple[int, int]]:
        """
        Predice las coordenadas de los puntos anatómicos en una imagen.
        
        Args:
            image_path: Ruta de la imagen a analizar
            
        Returns:
            Dict[str, Tuple[int, int]]: Diccionario con las coordenadas predichas
        """
        try:
            # Cargar y procesar imagen
            image = self.image_processor.load_and_resize_image(image_path)
            predictions = {}
            
            # Predecir cada punto
            for coord_name, config in self.coord_manager.coord_data.items():
                if coord_name not in self.pca_models:
                    logging.warning(f"Saltando {coord_name}: no hay modelo PCA")
                    continue
                    
                try:
                    # Obtener coordenadas de búsqueda
                    search_coordinates = self.coord_manager.get_search_coordinates(coord_name)
                    
                    # Analizar región de búsqueda
                    min_error, min_error_coords, _ = self.pca_models[coord_name].analyze_search_region(
                        image=image,
                        search_coordinates=search_coordinates,
                        template_width=config['width'],
                        template_height=config['height'],
                        intersection_x=config['left'],
                        intersection_y=config['sup']
                    )
                    
                    predictions[coord_name] = min_error_coords
                    
                except Exception as e:
                    logging.error(f"Error prediciendo {coord_name}: {str(e)}")
                    predictions[coord_name] = (0, 0)
                    continue
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error procesando imagen {image_path}: {str(e)}")
            return {}

    def save_predictions_to_csv(self, 
                              predictions_dict: Dict[int, Dict[str, Tuple[int, int]]], 
                              output_file: str):
        """
        Guarda las predicciones en formato CSV.
        
        Args:
            predictions_dict: Diccionario con predicciones por imagen
            output_file: Ruta del archivo de salida
        """
        try:
            # Convertir predicciones a formato CSV
            rows = []
            for index, predictions in predictions_dict.items():
                row = [index]  # Índice
                
                # Agregar coordenadas x,y para cada punto
                for i in range(1, 16):
                    coord_name = f"Coord{i}"
                    x, y = predictions.get(coord_name, (0, 0))
                    row.extend([x, y])
                    
                # Agregar identificador de imagen
                row.append(f"image_{index}")
                rows.append(row)
                
            # Crear DataFrame y guardar
            df = pd.DataFrame(rows)
            df.to_csv(output_file, header=False, index=False)
            logging.info(f"Predicciones guardadas en {output_file}")
            
        except Exception as e:
            logging.error(f"Error guardando predicciones: {str(e)}")

def main():
    """Función principal del programa."""
    try:
        # Configurar rutas
        project_root = PULMO_ALIGN_DIR
        indices_file = PULMO_ALIGN_DATA / "indices.csv"
        output_file = SCRIPT_DIR / "predicted_coordinates.csv"
        
        # Inicializar predictor
        logging.info("Iniciando predictor de coordenadas...")
        predictor = CoordinatePredictor(project_root)
        predictor.initialize_models()
        
        # Diccionario para almacenar predicciones
        all_predictions = {}
        
        # Procesar imágenes
        indices_df = pd.read_csv(indices_file, header=None)
        for index, row in tqdm(indices_df.iterrows(), 
                             desc="Procesando imágenes",
                             total=len(indices_df)):
            try:
                image_path = predictor.image_processor.get_image_path(row[0], str(indices_file))
                predictions = predictor.predict_coordinates(image_path)
                all_predictions[row[0]] = predictions
                
            except Exception as e:
                logging.error(f"Error procesando índice {row[0]}: {str(e)}")
                continue
        
        # Guardar resultados
        predictor.save_predictions_to_csv(all_predictions, str(output_file))
        logging.info("Proceso completado exitosamente")
        
    except Exception as e:
        logging.error(f"Error en ejecución principal: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
