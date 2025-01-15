"""
Módulo para la visualización de resultados del análisis de imágenes pulmonares.

Este módulo proporciona clases y funciones para:
- Visualización de resultados del análisis PCA
- Visualización de regiones de búsqueda
- Presentación de errores y estadísticas
- Generación de gráficos comparativos
"""

import cv2
import numpy as np
import matplotlib
# Configurar backend no interactivo
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class Visualizer:
    """
    Clase para visualizar resultados del análisis de imágenes pulmonares.
    
    Esta clase maneja:
    - Visualización de resultados del análisis PCA
    - Visualización de regiones de búsqueda
    - Presentación de errores y estadísticas
    - Generación de gráficos comparativos
    
    Attributes:
        output_dir (Path): Directorio para guardar las visualizaciones
    """
    
    def __init__(self, output_dir: str = "visualization_results"):
        """
        Inicializa el visualizador.
        
        Args:
            output_dir (str): Directorio para guardar las visualizaciones
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_results(self,
                         image: np.ndarray,
                         coord_config: Dict,
                         results: Dict,
                         pca_models: Dict,
                         save: bool = True) -> None:
        """
        Visualiza los resultados del análisis para una coordenada específica.
        
        Args:
            image (np.ndarray): Imagen analizada
            coord_config (Dict): Configuración de la coordenada
            results (Dict): Resultados del análisis
            pca_models (Dict): Modelos PCA utilizados
            save (bool): Si True, guarda la visualización en un archivo
        """
        for coord_name, result in results.items():
            if coord_name not in pca_models:
                print(f"Saltando visualización de {coord_name}: no hay modelo PCA")
                continue

            config = coord_config[coord_name]
            pca_model = pca_models[coord_name]
            
            plt.figure(figsize=(20, 16))
            #gs = GridSpec(3, 3, height_ratios=[2, 1, 1])
            gs = GridSpec(1, 1)
            
            # Imagen original con resultados
            #ax_main = plt.subplot(gs[0, :])
            ax_main = plt.subplot(gs[0, 0])
            ax_main.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax_main.set_title(f"{coord_name} - Resultados de búsqueda")
            
            # Puntos de búsqueda
            if 'search_coordinates' in result:
                search_coords = np.array(result['search_coordinates'])
                if len(search_coords) > 0:
                    ax_main.scatter(search_coords[:, 1], search_coords[:, 0], 
                                  c='red', s=20, alpha=0.5,
                                  label='Puntos de búsqueda')
            
            # Resultado final
            min_x, min_y = result['min_error_coords']
            rect = plt.Rectangle(
                (min_x - config['left'], min_y - config['sup']),
                config['width'], config['height'],
                fill=False, edgecolor='green', linewidth=2,
                label='Región encontrada'
            )
            ax_main.add_patch(rect)
            ax_main.plot(min_x, min_y, 'g*', markersize=25,
                        label='Punto óptimo')
            
            ax_main.set_xlim(0, 64)
            ax_main.set_ylim(64, 0)
            ax_main.grid(True, alpha=0.3)
            ax_main.legend()
            
            # Información
            # ax_info = plt.subplot(gs[0, 2])
            # ax_info.axis('off')
            # info_text = [
            #     f"{coord_name} - Información",
            #     "-" * 30,
            #     f"Error mínimo: {result['min_error']:.4f}",
            #     f"Paso: {result['min_error_step']}",
            #     f"Coordenadas: ({min_x}, {min_y})",
            #     "",
            #     "Dimensiones:",
            #     f"Ancho: {config['width']}",
            #     f"Alto: {config['height']}",
            #     f"Superior: {config['sup']}",
            #     f"Inferior: {config['inf']}",
            #     f"Izquierda: {config['left']}",
            #     f"Derecha: {config['right']}",
            #     "",
            #     "PCA:",
            #     f"Componentes: {pca_model.n_components}",
            #     f"Varianza: {pca_model.pca.explained_variance_ratio_.sum():.2%}"
            # ]
            # ax_info.text(0, 1, '\n'.join(info_text), 
            #             va='top', ha='left', fontsize=8)
            
            # Rostro medio y eigenfaces
            #ax_mean = plt.subplot(gs[1, 0])
            #ax_mean.imshow(pca_model.mean_face, cmap='gray')
            #ax_mean.set_title("Rostro medio")
            #ax_mean.axis('off')
            
            #n_eigen = min(2, len(pca_model.eigenfaces))
            #for i in range(n_eigen):
            #    ax = plt.subplot(gs[1, i+1])
            #    ax.imshow(pca_model.eigenfaces[i], cmap='gray')
            #    ax.set_title(f"Eigenface {i+1}")
            #    ax.axis('off')
            
            #try:
            #    # Región recortada
            #    start_x = min_x - config['left']
            #    start_y = min_y - config['sup']
            #    cropped_region = image[
            #        max(0, start_y):min(64, start_y + config['height']),
            #        max(0, start_x):min(64, start_x + config['width'])
            #    ]
            #    
            #    cropped_region = cv2.resize(cropped_region, 
            #                              (config['width'], config['height']))
            #    if len(cropped_region.shape) == 3:
            #        cropped_region = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
            #    cropped_region = cropped_region.astype(float) / 255.0
            #    
            #    omega = pca_model.calculate_omega(cropped_region)
            #    reconstructed = pca_model.reconstruct_image(omega)
            #    reconstructed = reconstructed.reshape(cropped_region.shape)
            #    
            #    ax_crop = plt.subplot(gs[2, 0])
            #    ax_crop.imshow(cropped_region, cmap='gray')
            #    ax_crop.set_title("Región recortada")
            #    ax_crop.axis('off')
            #    
             #   ax_recon = plt.subplot(gs[2, 1])
             #   ax_recon.imshow(reconstructed, cmap='gray')
             #   ax_recon.set_title("Reconstrucción")
             #   ax_recon.axis('off')
             #   
             #   ax_error = plt.subplot(gs[2, 2])
             #   error_img = np.abs(cropped_region - reconstructed)
             #   ax_error.imshow(error_img, cmap='hot')
             #   ax_error.set_title(f"Error (Norma L2: {result['min_error']:.4f})")
             #   ax_error.axis('off')
             #   
            #except Exception as e:
            #    print(f"Error en visualización de región: {str(e)}")
            
            plt.tight_layout()
            
            output_path = self.output_dir / f"{coord_name}_results.png"
            plt.savefig(output_path)
            print(f"Visualización guardada en: {output_path}")
            plt.close()

    def plot_error_distribution(self, 
                              errors: List[float], 
                              coord_name: str,
                              save: bool = True) -> None:
        """
        Visualiza la distribución de errores para una coordenada.
        
        Args:
            errors (List[float]): Lista de errores
            coord_name (str): Nombre de la coordenada
            save (bool): Si True, guarda la visualización en un archivo
        """
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.75)
        plt.title(f"Distribución de errores - {coord_name}")
        plt.xlabel("Error")
        plt.ylabel("Frecuencia")
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"{coord_name}_error_distribution.png"
        plt.savefig(output_path)
        print(f"Distribución guardada en: {output_path}")
        plt.close()

    def plot_search_path(self,
                        search_coordinates: List[Tuple[int, int]],
                        min_error_coords: Tuple[int, int],
                        coord_name: str,
                        save: bool = True) -> None:
        """
        Visualiza el camino de búsqueda para una coordenada.
        
        Args:
            search_coordinates (List[Tuple[int, int]]): Coordenadas de búsqueda
            min_error_coords (Tuple[int, int]): Coordenadas del error mínimo
            coord_name (str): Nombre de la coordenada
            save (bool): Si True, guarda la visualización en un archivo
        """
        plt.figure(figsize=(8, 8))
        coords = np.array(search_coordinates)
        
        # Invertir el orden de las coordenadas para coincidir con la visualización principal
        plt.scatter(coords[:, 1], coords[:, 0], c='red', s=10, alpha=0.5,
                   label='Puntos de búsqueda')
        plt.plot(min_error_coords[0], min_error_coords[1], 'g*', markersize=15,
                label='Punto óptimo')
        
        plt.title(f"Camino de búsqueda - {coord_name}")
        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")
        plt.grid(True, alpha=0.3)
        plt.ylim(64, 0)  # Invertir el eje Y para coincidir con la visualización principal
        plt.legend()
        
        output_path = self.output_dir / f"{coord_name}_search_path.png"
        plt.savefig(output_path)
        print(f"Camino de búsqueda guardado en: {output_path}")
        plt.close()
