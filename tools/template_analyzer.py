"""
Programa para analizar y calcular templates de recorte y puntos de intersección.

Este programa:
1. Lee las coordenadas de búsqueda
2. Calcula las distancias y templates para cada coordenada
3. Determina los puntos de intersección
4. Guarda los resultados en un archivo JSON
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class TemplateAnalyzer:
    def __init__(self, search_coordinates_file: str, output_dir: str = "template_analysis"):
        """
        Inicializa el analizador de templates.
        
        Args:
            search_coordinates_file: Ruta al archivo JSON con coordenadas de búsqueda
            output_dir: Directorio donde se guardarán los resultados
        """
        self.search_coordinates_file = Path(search_coordinates_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Almacenamiento de datos
        self.search_coordinates = {}
        self.template_data = {}
        
    def load_search_coordinates(self) -> None:
        """Carga las coordenadas de búsqueda desde el archivo JSON."""
        try:
            with open(self.search_coordinates_file, 'r') as f:
                data = json.load(f)
                
            # Procesar cada coordenada
            for i in range(1, 16):
                coord_name = f"coord{i}"
                if coord_name in data:
                    self.search_coordinates[coord_name] = data[coord_name]
                else:
                    print(f"Advertencia: {coord_name} no encontrada en el archivo")
                    
        except Exception as e:
            raise ValueError(f"Error al cargar coordenadas de búsqueda: {str(e)}")
            
    def calculate_region_bounds(self, coord_points: List[List[int]]) -> Dict:
        """
        Calcula los límites de una región basado en las coordenadas de búsqueda.
        
        Args:
            coord_points: Lista de coordenadas [x,y]
            
        Returns:
            Diccionario con los límites calculados
        """
        if not coord_points:
            return None
            
        # Extraer x,y de los puntos
        x_coords = [p[0] for p in coord_points]
        y_coords = [p[1] for p in coord_points]
        
        # Calcular límites
        left = min(x_coords)
        right = max(x_coords)
        sup = min(y_coords)
        inf = max(y_coords)
        
        # Calcular dimensiones para coordenadas 0-based
        # Ejemplo: si tenemos puntos en x=5 y x=7
        # Necesitamos incluir todos los puntos: 5,6,7
        # right-left = 7-5 = 2 (espacios entre puntos)
        # width = 7-5+1 = 3 (número total de puntos)
        width = right - left + 1   # Número total de puntos en X
        height = inf - sup + 1     # Número total de puntos en Y
        
        return {
            "sup": sup,     # y mínima (0-based)
            "inf": inf,     # y máxima (0-based)
            "left": left,   # x mínima (0-based)
            "right": right, # x máxima (0-based)
            "width": width, # Número total de puntos en X (incluyendo extremos)
            "height": height # Número total de puntos en Y (incluyendo extremos)
        }
        
    def calculate_template_distances(self, search_region: np.ndarray, template_size: int = 64) -> Tuple[int, int, int, int]:
        """
        Calcula las distancias a,b,c,d desde la región de búsqueda al template original.
        
        Args:
            search_region: Matriz binaria con la región de búsqueda
            template_size: Tamaño del template original
            
        Returns:
            Tuple con las distancias (a,b,c,d)
        """
        # Validar tamaño del template
        if template_size != 64:
            raise ValueError("El tamaño del template debe ser 64x64")
            
        # Validar dimensiones de la región de búsqueda
        if search_region.shape != (64, 64):
            raise ValueError("La región de búsqueda debe ser 64x64")
            
        # Obtener límites de la región de búsqueda
        non_zero = np.nonzero(search_region)
        if len(non_zero[0]) == 0:
            raise ValueError("Región de búsqueda vacía")
            
        min_y, max_y = non_zero[0].min(), non_zero[0].max()
        min_x, max_x = non_zero[1].min(), non_zero[1].max()
        
        # Calcular distancias desde los límites de la región al template
        a = min_y  # Distancia desde el borde superior
        b = 63 - max_x  # Distancia desde el borde derecho 
        c = 63 - max_y  # Distancia desde el borde inferior 
        d = min_x  # Distancia desde el borde izquierdo 
        
        # Validar que las distancias son válidas
        if a + c >= template_size:
            raise ValueError(f"Las distancias verticales a({a})+c({c}) suman más que el tamaño del template")
        if b + d >= template_size:
            raise ValueError(f"Las distancias horizontales b({b})+d({d}) suman más que el tamaño del template")
            
        return a, b, c, d
        
    def create_search_region(self, coord_points: List[List[int]]) -> np.ndarray:
        """
        Crea una matriz binaria 64x64 con la región de búsqueda.
        
        Args:
            coord_points: Lista de coordenadas [x,y] en formato 0-based (0-63)
            
        Returns:
            Matriz binaria con la región de búsqueda
        """
        search_region = np.zeros((64, 64))
        for x, y in coord_points:
            search_region[x, y] = 1  # Coordenadas ya están en 0-based
        return search_region
        
    def create_cutting_template(self, a: int, b: int, c: int, d: int) -> np.ndarray:
        """
        Crea el template de recorte basado en las distancias calculadas.
        
        Args:
            a,b,c,d: Distancias calculadas
            
        Returns:
            Matriz binaria con el template de recorte
        """
        template = np.zeros((64, 64))
        
        # Calcular dimensiones del cuadrilátero
        height = c + a  # Suma de las distancias verticales
        width = b + d   # Suma de las distancias horizontales
        
        # Validar dimensiones
        if height <= 0 or width <= 0:
            raise ValueError(f"Dimensiones inválidas: {width}x{height}")
            
        # Crear template desde el punto (d,a)
        template[a:a+height, d:d+width] = 1
        
        return template
        
    def find_intersection_point(self, a: int, d: int) -> Tuple[int, int]:
        """
        Encuentra el punto de intersección en coordenadas globales.
        
        Args:
            a: Distancia desde el borde superior
            d: Distancia desde el borde izquierdo
            
        Returns:
            Tuple con las coordenadas (x,y) del punto de intersección
        """
        return d, a
        
    def visualize_template(self, 
                          search_region: np.ndarray,
                          template: np.ndarray,
                          intersection_point: Tuple[int, int],
                          coord_name: str) -> None:
        """
        Genera una visualización del template y punto de intersección.
        
        Args:
            search_region: Región de búsqueda
            template: Template de recorte
            intersection_point: Punto de intersección (x,y)
            coord_name: Nombre de la coordenada
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Región de búsqueda
        sns.heatmap(search_region, ax=ax1, cmap='RdYlBu_r', cbar=False)
        ax1.set_title(f'Región de Búsqueda - {coord_name}')
        
        # Template y punto de intersección
        ax2.imshow(template, cmap='binary')
        ax2.plot(intersection_point[0], intersection_point[1], 'r*', 
                markersize=15, label=f'Intersección ({intersection_point[0]},{intersection_point[1]})')
        ax2.set_title(f'Template de Recorte - {coord_name}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'template_analysis_{coord_name}.png')
        plt.close()
        
    def analyze_templates(self) -> None:
        """Analiza los templates para todas las coordenadas."""
        try:
            # Cargar coordenadas si no se han cargado
            if not self.search_coordinates:
                self.load_search_coordinates()
                
            # Procesar cada coordenada
            for coord_name, coord_points in self.search_coordinates.items():
                print(f"\nAnalizando {coord_name}...")
                
                # Calcular límites de la región
                region_bounds = self.calculate_region_bounds(coord_points)
                if not region_bounds:
                    print(f"No hay puntos para {coord_name}")
                    continue
                    
                # Crear región de búsqueda
                search_region = self.create_search_region(coord_points)
                
                # Calcular distancias
                a, b, c, d = self.calculate_template_distances(search_region)
                
                # Calcular dimensiones del template
                height = 63 - (c + a)  # Suma de las distancias verticales
                width = 63 - (b + d)   # Suma de las distancias horizontales
                
                # Crear template de recorte
                template = self.create_cutting_template(a, b, c, d)
                
                # Encontrar punto de intersección
                intersection_point = self.find_intersection_point(a, d)
                
                # Guardar datos
                self.template_data[coord_name] = {
                    "region_bounds": region_bounds,
                    "distances": {
                        "a": int(a),
                        "b": int(b),
                        "c": int(c),
                        "d": int(d)
                    },
                    "template_bounds": {
                        "min_x": int(d),
                        "max_x": int(d + width),
                        "min_y": int(a),
                        "max_y": int(a + height),
                        "width": int(d+b),
                        "height": int(a+c)
                    },
                    "intersection_point": {
                        "x": int(intersection_point[0]),
                        "y": int(intersection_point[1])
                    }
                }
                
                # Generar visualización
                self.visualize_template(
                    search_region,
                    template,
                    intersection_point,
                    coord_name
                )
                
            # Guardar resultados
            self.save_results()
            
        except Exception as e:
            print(f"Error durante el análisis: {str(e)}")
            raise
            
    def save_results(self) -> None:
        """Guarda los resultados del análisis en un archivo JSON."""
        try:
            output_file = self.output_dir / "template_analysis_results.json"
            with open(output_file, 'w') as f:
                json.dump(self.template_data, f, indent=2)
            print(f"\nResultados guardados en: {output_file}")
            
        except Exception as e:
            print(f"Error al guardar resultados: {str(e)}")
            raise

def main():
    # Configurar rutas
    tesis_root = Path("/home/donrobot/projects/Tesis")
    search_coordinates_file = tesis_root / "all_search_coordinates.json"
    output_dir = tesis_root / "tools/template_analysis"
    
    # Crear y ejecutar analizador
    analyzer = TemplateAnalyzer(
        search_coordinates_file=str(search_coordinates_file),
        output_dir=str(output_dir)
    )
    
    print("Iniciando análisis de templates...")
    analyzer.analyze_templates()
    print("\nAnálisis completado!")

if __name__ == "__main__":
    main()
