"""
Módulo para la visualización combinada de resultados del análisis de imágenes pulmonares.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

class CombinedVisualizer:
    """
    Clase para visualizar resultados combinados del análisis de imágenes pulmonares.
    
    Esta clase se especializa en generar visualizaciones que muestran todas las
    coordenadas encontradas o un subconjunto específico en una sola imagen,
    destacando los puntos óptimos.
    """
    
    def __init__(self, output_dir: str = "visualization_results"):
        """
        Inicializa el visualizador combinado.
        
        Args:
            output_dir: Directorio para guardar las visualizaciones
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_combined_results(self,
                                 image: np.ndarray,
                                 results: Dict[str, Dict],
                                 save: bool = True) -> None:
        """
        Visualiza todas las coordenadas encontradas en una sola imagen.
        Todo se maneja en formato (y,x) internamente, pero para matplotlib
        convertimos a (x,y) al momento de graficar.
        
        Args:
            image: Imagen analizada
            results: Resultados del análisis para todas las coordenadas
            save: Si True, guarda la visualización
        """
        plt.figure(figsize=(15, 12))
        
        # Mostrar la imagen original
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Resultados combinados de todos los puntos anatómicos")
        
        # Colores para diferentes templates
        colors = ['g', 'r', 'b', 'c', 'm', 'y']
        
        # Agregar cada punto y su template
        for idx, (coord_name, result) in enumerate(results.items()):
            color = colors[idx % len(colors)]
            template_bounds = result['template_bounds']
            region_bounds = result.get('region_bounds', {})
            
            # Dibujar región de búsqueda 
            # Matplotlib espera (x,y), por lo que invertimos al graficar
            region_rect = plt.Rectangle(
                (region_bounds['left'], region_bounds['sup']),  # (x,y) para matplotlib
                region_bounds['width'],
                region_bounds['height'],
                fill=False, edgecolor=color, linewidth=1, linestyle='--',
                label=f'{coord_name} - Región'
            )
            plt.gca().add_patch(region_rect)
            
            # Obtener coordenadas del mejor punto encontrado (y,x)
            best_y, best_x = result['min_error_coords']
            
            # Obtener punto de intersección (y,x)
            intersection = result['intersection_point']
            intersection_y = int(intersection['y'])
            intersection_x = int(intersection['x'])
            
            # Calcular esquina superior izquierda del template
            template_start_x = best_x - intersection_x
            template_start_y = best_y - intersection_y
            
            # Dibujar template 
            template_rect = plt.Rectangle(
                (template_start_x, template_start_y),  # (x,y) para matplotlib
                template_bounds['width'],
                template_bounds['height'],
                fill=False, edgecolor=color, linewidth=2,
                label=f'{coord_name} - Template'
            )
            plt.gca().add_patch(template_rect)
            
            # Dibujar mejor punto encontrado
            plt.plot(best_x, best_y, f'{color}*', markersize=15,
                    label=f'{coord_name} - Mejor punto')
            
            # Dibujar punto de intersección en su posición relativa al template
            plt.plot(best_x, best_y, f'{color}+', markersize=10,
                    label=f'{coord_name} - Punto de intersección')
        
        plt.xlim(0, 63)
        plt.ylim(63, 0)  # Y aumenta hacia abajo en imágenes
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
        Todo se maneja en formato (y,x) internamente, pero para matplotlib
        convertimos a (x,y) al momento de graficar.
        
        Args:
            image: Imagen analizada
            results: Resultados del análisis para todas las coordenadas
            save: Si True, guarda la visualización
        """
        plt.figure(figsize=(15, 12))
        
        # Mostrar la imagen original
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Resultados de Coord1 y Coord2")
        
        # Colores específicos para cada coordenada
        colors = {'coord1': 'g', 'coord2': 'r'}
        
        # Agregar solo los puntos de Coord1 y Coord2
        for coord_name in ['coord1', 'coord2']:
            if coord_name in results:
                color = colors[coord_name]
                result = results[coord_name]
                template_bounds = result['template_bounds']
                region_bounds = result.get('region_bounds', {})
                
                # Dibujar región de búsqueda
                # Matplotlib espera (x,y), por lo que invertimos al graficar
                region_rect = plt.Rectangle(
                    (region_bounds['left'], region_bounds['sup']),  # (x,y) para matplotlib
                    region_bounds['width'],
                    region_bounds['height'],
                    fill=False, edgecolor=color, linewidth=1, linestyle='--',
                    label=f'{coord_name} - Región'
                )
                plt.gca().add_patch(region_rect)
                
                # Obtener coordenadas del mejor punto encontrado (y,x)
                best_y, best_x = result['min_error_coords']
                
                # Obtener punto de intersección (y,x)
                intersection = result['intersection_point']
                intersection_y = int(intersection['y'])
                intersection_x = int(intersection['x'])
                
                # Calcular esquina superior izquierda del template
                template_start_x = best_x - intersection_x
                template_start_y = best_y - intersection_y
                
                # Dibujar template
                template_rect = plt.Rectangle(
                    (template_start_x, template_start_y),  # (x,y) para matplotlib
                    template_bounds['width'],
                    template_bounds['height'],
                    fill=False, edgecolor=color, linewidth=2,
                    label=f'{coord_name} - Template'
                )
                plt.gca().add_patch(template_rect)
                
                # Dibujar mejor punto encontrado
                plt.plot(best_x, best_y, f'{color}*', markersize=15,
                        label=f'{coord_name} - Mejor punto')
                
                # Dibujar punto de intersección en su posición relativa al template
                plt.plot(best_x, best_y, f'{color}+', markersize=10,
                        label=f'{coord_name} - Punto de intersección')
        
        plt.xlim(0, 63)
        plt.ylim(63, 0)  # Y aumenta hacia abajo en imágenes
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "coord1_coord2_results.png"
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Visualización de Coord1 y Coord2 guardada en: {output_path}")
        
        plt.close()
