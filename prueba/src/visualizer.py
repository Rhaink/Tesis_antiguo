"""
Módulo para la visualización de resultados del análisis PCA.
Versión simplificada del proyecto original pulmo_align.
"""

import matplotlib
matplotlib.use('Agg')  # Usar backend sin interfaz gráfica
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class Visualizer:
    """
    Clase para visualizar resultados del análisis PCA.
    
    Esta clase maneja:
    - Visualización de distribución de errores
    - Visualización de caminos de búsqueda
    - Visualización de resultados finales
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Inicializa el visualizador.
        
        Args:
            output_dir: Directorio opcional para guardar visualizaciones
        """
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Configuración de estilo para matplotlib
        plt.style.use('default')

    def plot_error_distribution(self,
                              errors: List[float],
                              coord_name: str,
                              save: bool = False,
                              show: bool = False) -> Optional[str]:
        """
        Visualiza la distribución de errores para un punto.
        
        Args:
            errors: Lista de errores
            coord_name: Nombre del punto anatómico
            save: Si se debe guardar la visualización
            show: Si se debe mostrar la visualización
            
        Returns:
            Ruta del archivo guardado si save=True
        """
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.75)
        plt.title(f'Distribución de Errores - {coord_name}')
        plt.xlabel('Error')
        plt.ylabel('Frecuencia')
        plt.grid(True)
        
        if save and self.output_dir:
            output_path = self.output_dir / f"{coord_name}_error_dist.png"
            plt.savefig(str(output_path))
            plt.close()
            return str(output_path)
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return None

    def plot_search_path(self,
                        search_coordinates: List[Tuple[int, int]],
                        best_point: Tuple[int, int],
                        coord_name: str,
                        save: bool = False,
                        show: bool = False) -> Optional[str]:
        """
        Visualiza el camino de búsqueda para un punto.
        Mantiene formato (y,x) consistentemente.
        
        Args:
            search_coordinates: Lista de coordenadas de búsqueda (y,x)
            best_point: Coordenadas del mejor punto encontrado (y,x)
            coord_name: Nombre del punto anatómico
            save: Si se debe guardar la visualización
            show: Si se debe mostrar la visualización
            
        Returns:
            Ruta del archivo guardado si save=True
        """
        plt.figure(figsize=(10, 10))
        
        # Graficar coordenadas de búsqueda
        coords = np.array(search_coordinates)
        plt.scatter(coords[:, 1], coords[:, 0], 
                   c='blue', alpha=0.5, label='Puntos de búsqueda (y,x)')
        
        # Graficar punto óptimo manteniendo (y,x)
        best_y, best_x = best_point
        plt.scatter(best_x, best_y, 
                   c='red', s=100, marker='*', label='Mejor punto encontrado (y,x)')
        
        plt.title(f'Camino de Búsqueda - {coord_name}')
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')
        plt.gca().invert_yaxis()  # Y aumenta hacia abajo en imágenes
        plt.legend()
        plt.grid(True)
        
        if save and self.output_dir:
            output_path = self.output_dir / f"{coord_name}_search_path.png"
            plt.savefig(str(output_path))
            plt.close()
            return str(output_path)
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return None

    def visualize_results(self,
                         image: np.ndarray,
                         coord_config: Dict,
                         results: Dict,
                         pca_models: Dict,
                         save: bool = True) -> None:
        """
        Visualiza los resultados del análisis para una coordenada específica.
        Todo se maneja en formato (y,x).
        
        Args:
            image: Imagen analizada
            coord_config: Configuración de la coordenada
            results: Resultados del análisis
            pca_models: Modelos PCA utilizados
            save: Si True, guarda la visualización
        """
        for coord_name, result in results.items():
            if coord_name not in pca_models:
                print(f"Saltando visualización de {coord_name}: no hay modelo PCA")
                continue

            # Obtener datos del template y región
            template_data = coord_config[coord_name]
            template_bounds = template_data["template_bounds"]
            intersection_point = template_data["intersection_point"]
            region_bounds = template_data["region_bounds"]
            
            plt.figure(figsize=(20, 16))
            gs = GridSpec(1, 1)
            
            # Imagen original con resultados
            ax_main = plt.subplot(gs[0, 0])
            ax_main.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax_main.set_title(f"{coord_name} - Resultados de búsqueda")
            
            # Dibujar región de búsqueda
            region_rect = plt.Rectangle(
                (region_bounds['left'], region_bounds['sup']),
                region_bounds['width'],
                region_bounds['height'],
                fill=False, edgecolor='blue', linewidth=1, linestyle='--',
                label='Región de búsqueda'
            )
            ax_main.add_patch(region_rect)
            
            # Puntos de búsqueda
            if 'search_coordinates' in result:
                search_coords = np.array(result['search_coordinates'])
                if len(search_coords) > 0:
                    ax_main.scatter(search_coords[:, 1], search_coords[:, 0], 
                                  c='red', s=20, alpha=0.5,
                                  label='Puntos de búsqueda (y,x)')
            
            # Obtener coordenadas del mejor punto encontrado
            min_y, min_x = result['min_error_coords']
            
            # Calcular la esquina superior izquierda del template
            template_start_x = min_x - intersection_point['x']
            template_start_y = min_y - intersection_point['y']
            
            # Dibujar template en posición final
            template_rect = plt.Rectangle(
                (template_start_x, template_start_y),
                template_bounds['width'],
                template_bounds['height'],
                fill=False, edgecolor='green', linewidth=2,
                label='Template'
            )
            ax_main.add_patch(template_rect)
            
            # Dibujar punto óptimo (donde se movió el punto de intersección)
            ax_main.plot(min_x, min_y, 'g*', markersize=25,
                        label='Punto óptimo')
            
            # Dibujar punto de intersección en su posición relativa al template
            ax_main.plot(min_x, min_y, 'r+', markersize=15,
                        label='Punto de intersección')
            
            ax_main.set_xlim(0, 63)
            ax_main.set_ylim(63, 0)  # Y aumenta hacia abajo
            ax_main.grid(True, alpha=0.3)
            ax_main.legend()
            
            plt.tight_layout()
            
            output_path = self.output_dir / f"{coord_name}_results.png"
            plt.savefig(output_path)
            print(f"Visualización guardada en: {output_path}")
            plt.close()

    def plot_eigenfaces(self,
                       eigenfaces: np.ndarray,
                       mean_face: np.ndarray,
                       n_components: int = 5,
                       save: bool = False,
                       show: bool = False,
                       filename: Optional[str] = None) -> Optional[str]:
        """
        Visualiza los eigenfaces y la cara media.
        
        Args:
            eigenfaces: Array de eigenfaces
            mean_face: Cara media
            n_components: Número de eigenfaces a mostrar
            save: Si se debe guardar la visualización
            show: Si se debe mostrar la visualización
            
        Returns:
            Ruta del archivo guardado si save=True
        """
        n_row = 2  # Primera fila para mean_face, segunda para eigenfaces
        n_col = max(1, min(n_components, len(eigenfaces)))
        
        plt.figure(figsize=(4*n_col, 8))
        
        # Mostrar cara media
        plt.subplot(n_row, n_col, 1)
        plt.imshow(mean_face, cmap='gray')
        plt.title('Cara Media')
        plt.axis('off')
        
        # Mostrar eigenfaces
        for i in range(min(n_components, len(eigenfaces))):
            plt.subplot(n_row, n_col, n_col + i + 1)
            plt.imshow(eigenfaces[i], cmap='gray')
            plt.title(f'Eigenface {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        
        if save and self.output_dir:
            output_path = self.output_dir / (filename or "eigenfaces.png")
            plt.savefig(str(output_path))
            plt.close()
            return str(output_path)
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return None
