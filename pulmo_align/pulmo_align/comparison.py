"""
Programa para comparar las coordenadas predichas con las originales.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from pulmo_align.coordinates.coordinate_manager import CoordinateManager
from pulmo_align.image_processing.image_processor import ImageProcessor
from pulmo_align.pca_analysis.pca_analyzer import PCAAnalyzer

class CoordinateComparator:
    def __init__(self, dataset_path: str):
        """
        Inicializa el comparador de coordenadas.
        
        Args:
            dataset_path (str): Ruta al directorio del dataset COVID-19_Radiography_Dataset
        """
        self.dataset_path = Path(dataset_path)
        self.coord_manager = CoordinateManager()
        self.image_processor = ImageProcessor(base_path=str(self.dataset_path))
        self.pca_models = {}  # Un modelo PCA por cada coordenada
        
    def load_coordinates(self, coord_file: str, search_coords_file: str):
        """
        Carga las coordenadas desde los archivos.
        
        Args:
            coord_file (str): Ruta al archivo de coordenadas originales
            search_coords_file (str): Ruta al archivo de coordenadas de búsqueda
        """
        self.coord_manager.read_coordinates(coord_file)
        self.coord_manager.read_search_coordinates(search_coords_file)
    
    def train_pca_models(self, output_base_path: str):
        """
        Entrena los modelos PCA para cada coordenada.
        
        Args:
            output_base_path (str): Ruta base donde se encuentran las imágenes recortadas
        """
        for i in range(1, 16):
            coord_name = f"Coord{i}"
            print(f"Entrenando modelo PCA para {coord_name}...")
            
            # Cargar imágenes de entrenamiento
            training_images = self.image_processor.load_training_images(
                coord_name,
                target_size=(
                    self.coord_manager.coord_data[coord_name]['width'],
                    self.coord_manager.coord_data[coord_name]['height']
                )
            )
            
            if training_images:
                # Crear y entrenar modelo PCA
                pca_model = PCAAnalyzer()
                pca_model.train(training_images)
                self.pca_models[coord_name] = pca_model
                print(f"Modelo PCA para {coord_name} entrenado con {len(training_images)} imágenes")
            else:
                print(f"No se encontraron imágenes de entrenamiento para {coord_name}")

    def visualize_comparison(self, image_index: int, indices_file: str, save_path: str = None):
        """
        Visualiza la comparación entre coordenadas originales y predichas.
        
        Args:
            image_index (int): Índice de la imagen a comparar
            indices_file (str): Ruta al archivo de índices
            save_path (str, optional): Ruta donde guardar la visualización
        """
        # Obtener coordenadas originales
        coords = self.coord_manager.get_image_coordinates(image_index)
        if not coords:
            print(f"No se encontraron coordenadas para el índice {image_index}")
            return
            
        # Cargar imagen
        try:
            image_path = self.image_processor.get_image_path(image_index, indices_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error al cargar la imagen: {image_path}")
                return
                
            # Convertir BGR a RGB para matplotlib
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Crear figura
            plt.figure(figsize=(15, 10))
            plt.imshow(image)
            
            # Obtener dimensiones de la imagen
            height, width = image.shape[:2]
            scale_x = width / 64.0
            scale_y = height / 64.0
            
            # Plotear puntos originales (verde)
            for coord_name, (x, y) in coords.items():
                # Escalar coordenadas al tamaño real de la imagen
                scaled_x = x * scale_x
                scaled_y = y * scale_y
                plt.plot(scaled_x, scaled_y, 'go', markersize=10, label=f'{coord_name} (Original)')
                plt.text(scaled_x+5, scaled_y+5, coord_name, color='green', fontsize=8)
            
            # Obtener predicciones usando PCA
            try:
                for coord_name in coords.keys():
                    if coord_name not in self.pca_models:
                        print(f"No hay modelo PCA para {coord_name}")
                        continue
                        
                    # Obtener configuración y coordenadas de búsqueda
                    config = self.coord_manager.get_coordinate_config(coord_name)
                    search_coords = self.coord_manager.get_search_coordinates(coord_name)
                    
                    if not search_coords:
                        print(f"No hay coordenadas de búsqueda para {coord_name}")
                        continue
                    
                    # Realizar predicción
                    try:
                        # Convertir a escala de grises y aplicar SAHS
                        if len(image.shape) > 2:
                            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        else:
                            gray_image = image.copy()
                            
                        # Aplicar mejora de contraste SAHS
                        enhanced_image = self.image_processor.contrast_enhancer.enhance_contrast_sahs(gray_image)
                        if enhanced_image is None:
                            print(f"Error al mejorar contraste para {coord_name}")
                            continue
                            
                        # Redimensionar imagen mejorada a 64x64 para PCA
                        image_resized = cv2.resize(enhanced_image, (64, 64))
                        
                        # Escalar coordenadas de búsqueda al tamaño 64x64
                        scaled_search_coords = []
                        for sx, sy in search_coords:
                            scaled_sx = int((sx / width) * 64)
                            scaled_sy = int((sy / height) * 64)
                            scaled_search_coords.append((scaled_sx, scaled_sy))
                        
                        min_error, (pred_x, pred_y), _ = self.pca_models[coord_name].analyze_search_region(
                            image_resized,
                            scaled_search_coords,
                            config['width'],
                            config['height'],
                            config['left'],
                            config['sup']
                        )
                        
                        # Escalar predicción al tamaño real de la imagen
                        scaled_pred_x = pred_x * scale_x
                        scaled_pred_y = pred_y * scale_y
                        
                        # Plotear predicción
                        plt.plot(scaled_pred_x, scaled_pred_y, 'ro', markersize=10)
                        plt.text(scaled_pred_x+5, scaled_pred_y+5, f'{coord_name} (Pred)', color='red', fontsize=8)
                        
                        # Dibujar línea entre original y predicción
                        orig_x, orig_y = coords[coord_name]
                        scaled_orig_x = orig_x * scale_x
                        scaled_orig_y = orig_y * scale_y
                        plt.plot([scaled_orig_x, scaled_pred_x], [scaled_orig_y, scaled_pred_y], 'y--', alpha=0.5)
                        
                    except Exception as e:
                        print(f"Error al predecir {coord_name}: {str(e)}")
            except Exception as e:
                print(f"Error al cargar predicciones: {str(e)}")
            
            plt.title(f'Comparación de Coordenadas - Imagen {image_index}')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path)
                print(f"Visualización guardada en: {save_path}")
            else:
                plt.show()
            plt.close()
                
        except Exception as e:
            print(f"Error al procesar la imagen {image_index}: {str(e)}")
    
    def calculate_statistics(self, image_index: int, indices_file: str) -> dict:
        """
        Calcula estadísticas de comparación para una imagen.
        
        Args:
            image_index (int): Índice de la imagen
            
        Returns:
            dict: Diccionario con estadísticas de comparación
        """
        coords = self.coord_manager.get_image_coordinates(image_index)
        if not coords:
            return None
            
        stats = {
            'image_index': image_index,
            'coordinates': coords,
            'differences': {}
        }
        
        # Cargar imagen
        try:
            image_path = self.image_processor.get_image_path(image_index, indices_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error al cargar la imagen: {image_path}")
                return None

            # Convertir a escala de grises y aplicar SAHS
            if len(image.shape) > 2:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image.copy()
                
            # Aplicar mejora de contraste SAHS
            enhanced_image = self.image_processor.contrast_enhancer.enhance_contrast_sahs(gray_image)
            if enhanced_image is None:
                print("Error al mejorar contraste de la imagen")
                return None
                
            # Redimensionar imagen mejorada a 64x64 para PCA
            image_resized = cv2.resize(enhanced_image, (64, 64))
            
            # Obtener dimensiones originales para escalar coordenadas
            height, width = image.shape[:2]
            
            # Calcular diferencias entre originales y predicciones
            for coord_name, (orig_x, orig_y) in coords.items():
                # Escalar coordenadas originales a 64x64
                scaled_orig_x = int((orig_x / width) * 64)
                scaled_orig_y = int((orig_y / height) * 64)
                if coord_name not in self.pca_models:
                    continue
                    
                config = self.coord_manager.get_coordinate_config(coord_name)
                search_coords = self.coord_manager.get_search_coordinates(coord_name)
                
                if not search_coords:
                    continue
                
                try:
                    # Escalar coordenadas de búsqueda a 64x64
                    scaled_search_coords = []
                    for sx, sy in search_coords:
                        scaled_sx = int((sx / width) * 64)
                        scaled_sy = int((sy / height) * 64)
                        scaled_search_coords.append((scaled_sx, scaled_sy))
                    
                    min_error, (pred_x, pred_y), _ = self.pca_models[coord_name].analyze_search_region(
                        image_resized,
                        scaled_search_coords,
                        config['width'],
                        config['height'],
                        config['left'],
                        config['sup']
                    )
                    
                    # Escalar predicción de vuelta al tamaño original
                    pred_x = int((pred_x / 64) * width)
                    pred_y = int((pred_y / 64) * height)
                    
                    # Calcular error euclidiano
                    error = np.sqrt((orig_x - pred_x)**2 + (orig_y - pred_y)**2)
                    
                    stats['differences'][coord_name] = {
                        'original': (orig_x, orig_y),
                        'predicted': (pred_x, pred_y),
                        'error': error,
                        'pca_error': min_error
                    }
                except Exception as e:
                    print(f"Error al calcular estadísticas para {coord_name}: {str(e)}")
        except Exception as e:
            print(f"Error al calcular estadísticas: {str(e)}")
        
        return stats

    def calculate_summary_statistics(self, indices_file: str, start_index: int = 0, end_index: int = 99) -> dict:
        """
        Calcula estadísticas resumen para un rango de imágenes.
        
        Args:
            start_index (int): Índice inicial
            end_index (int): Índice final
            
        Returns:
            dict: Diccionario con estadísticas resumen
        """
        summary = {
            'total_images': 0,
            'total_predictions': 0,
            'errors_by_coordinate': {},
            'overall_mean_error': 0,
            'overall_std_error': 0
        }
        
        all_errors = []
        
        for i in range(start_index, end_index + 1):
            stats = self.calculate_statistics(i, indices_file)
            if not stats:
                continue
                
            summary['total_images'] += 1
            
            for coord_name, data in stats['differences'].items():
                if coord_name not in summary['errors_by_coordinate']:
                    summary['errors_by_coordinate'][coord_name] = {
                        'errors': [],
                        'mean': 0,
                        'std': 0,
                        'min': float('inf'),
                        'max': 0
                    }
                
                error = data['error']
                summary['errors_by_coordinate'][coord_name]['errors'].append(error)
                summary['total_predictions'] += 1
                all_errors.append(error)
                
                # Actualizar min/max
                summary['errors_by_coordinate'][coord_name]['min'] = min(
                    summary['errors_by_coordinate'][coord_name]['min'], 
                    error
                )
                summary['errors_by_coordinate'][coord_name]['max'] = max(
                    summary['errors_by_coordinate'][coord_name]['max'], 
                    error
                )
        
        # Calcular estadísticas por coordenada
        for coord_stats in summary['errors_by_coordinate'].values():
            if coord_stats['errors']:
                coord_stats['mean'] = np.mean(coord_stats['errors'])
                coord_stats['std'] = np.std(coord_stats['errors'])
        
        # Calcular estadísticas globales
        if all_errors:
            summary['overall_mean_error'] = np.mean(all_errors)
            summary['overall_std_error'] = np.std(all_errors)
        
        return summary

def main():
    """Función principal del programa."""
    # Configurar rutas
    project_root = Path(__file__).parent
    dataset_path = project_root / "COVID-19_Radiography_Dataset"
    coord_file = project_root / "coordenadas.csv"
    search_coords_file = project_root / "all_search_coordinates.json"
    indices_file = project_root / "indices.csv"
    output_dir = project_root / "comparison_results"
    output_dir.mkdir(exist_ok=True)
    
    # Crear comparador
    comparator = CoordinateComparator(str(dataset_path))
    
    # Cargar coordenadas
    comparator.load_coordinates(str(coord_file), str(search_coords_file))
    
    # Entrenar modelos PCA
    comparator.train_pca_models(str(project_root / "processed_images"))
    
    # Visualizar algunas imágenes de ejemplo
    for image_index in range(5):  # Primeras 5 imágenes
        output_path = output_dir / f"comparison_{image_index}.png"
        comparator.visualize_comparison(image_index, str(indices_file), str(output_path))
        
        # Calcular estadísticas
        stats = comparator.calculate_statistics(image_index, str(indices_file))
        if stats:
            print(f"\nEstadísticas para imagen {image_index}:")
            for coord_name, data in stats['differences'].items():
                orig = data['original']
                pred = data['predicted']
                error = data['error']
                pca_error = data['pca_error']
                print(f"{coord_name}:")
                print(f"  Original: ({orig[0]}, {orig[1]})")
                print(f"  Predicho: ({pred[0]}, {pred[1]})")
                print(f"  Error espacial: {error:.2f} píxeles")
                print(f"  Error PCA: {pca_error:.4f}")
    
    # Calcular y mostrar estadísticas resumen
    print("\nCalculando estadísticas resumen...")
    summary = comparator.calculate_summary_statistics(str(indices_file), 0, 99)
    
    print(f"\nResumen de predicciones:")
    print(f"Total de imágenes procesadas: {summary['total_images']}")
    print(f"Total de predicciones realizadas: {summary['total_predictions']}")
    print(f"Error promedio global: {summary['overall_mean_error']:.2f} ± {summary['overall_std_error']:.2f} píxeles")
    
    print("\nEstadísticas por coordenada:")
    for coord_name, stats in summary['errors_by_coordinate'].items():
        print(f"\n{coord_name}:")
        print(f"  Error promedio: {stats['mean']:.2f} ± {stats['std']:.2f} píxeles")
        print(f"  Error mínimo: {stats['min']:.2f} píxeles")
        print(f"  Error máximo: {stats['max']:.2f} píxeles")

if __name__ == "__main__":
    main()
