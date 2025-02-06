"""
Módulo para la visualización combinada de resultados del análisis de imágenes pulmonares.

Este módulo proporciona funcionalidad para generar visualizaciones que combinen
los resultados de todas las coordenadas o un subconjunto específico en una sola imagen,
mostrando únicamente los puntos óptimos encontrados.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

class CombinedVisualizer:
    """
    Clase para visualizar resultados combinados del análisis de imágenes pulmonares.
    
    Esta clase se especializa en generar visualizaciones que muestran todas las
    coordenadas encontradas o un subconjunto específico en una sola imagen,
    destacando los puntos óptimos.
    
    Attributes:
        output_dir (Path): Directorio para guardar las visualizaciones
    """
    
    def __init__(self, output_dir: str = "visualization_results"):
        """
        Inicializa el visualizador combinado.
        
        Args:
            output_dir (str): Directorio para guardar las visualizaciones
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_combined_results(self,
                                 image: np.ndarray,
                                 results: Dict[str, Dict],
                                 save: bool = True) -> None:
        """
        Visualiza todas las coordenadas encontradas en una sola imagen.
        
        Args:
            image (np.ndarray): Imagen analizada
            results (Dict[str, Dict]): Resultados del análisis para todas las coordenadas
            save (bool): Si True, guarda la visualización en un archivo
        """
        plt.figure(figsize=(15, 12))
        
        # Mostrar la imagen original
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Puntos óptimos encontrados para todas las coordenadas")
        
        # Agregar cada punto óptimo
        for coord_name, result in results.items():
            min_x, min_y = result['min_error_coords']
            plt.plot(min_x, min_y, 'g*', markersize=15,
                    label=f'{coord_name} ({min_x}, {min_y})')
        
        plt.xlim(0, 64)
        plt.ylim(64, 0)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "combined_results.png"
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Visualización combinada guardada en: {output_path}")
        
        plt.close()

    def visualize_coord1_coord2(self,
                              image: np.ndarray,
                              results: Dict[str, Dict],
                              save: bool = True) -> None:
        """
        Visualiza específicamente las coordenadas 1 y 2 en una sola imagen.
        
        Args:
            image (np.ndarray): Imagen analizada
            results (Dict[str, Dict]): Resultados del análisis para todas las coordenadas
            save (bool): Si True, guarda la visualización en un archivo
        """
        plt.figure(figsize=(15, 12))
        
        # Mostrar la imagen original
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Puntos óptimos encontrados para Coord1 y Coord2")
        
        # Agregar solo los puntos de Coord1 y Coord2
        for coord_name in ['Coord1', 'Coord2']:
            if coord_name in results:
                min_x, min_y = results[coord_name]['min_error_coords']
                plt.plot(min_x, min_y, 'g*', markersize=15,
                        label=f'{coord_name} ({min_x}, {min_y})')
        
        plt.xlim(0, 64)
        plt.ylim(64, 0)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "coord1_coord2_results.png"
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Visualización de Coord1 y Coord2 guardada en: {output_path}")
        
        plt.close()
