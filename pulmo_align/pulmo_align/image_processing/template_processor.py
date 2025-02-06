"""
Módulo para el procesamiento de templates y visualizaciones de recorte de imágenes.

Este módulo proporciona funciones para:
- Cálculo de distancias de template
- Creación de templates de recorte
- Visualización de cada paso del proceso
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Tuple, Dict, Optional
"""
Módulo para el procesamiento de templates y visualizaciones de recorte de imágenes.

Este módulo proporciona funciones para:
- Cálculo de distancias de template
- Creación de templates de recorte
- Visualización de cada paso del proceso
"""


class TemplateProcessor:
    """
    Clase para el procesamiento de templates y visualizaciones.
    
    Esta clase maneja:
    - Cálculo de distancias desde regiones de búsqueda
    - Creación de templates de recorte
    - Generación de visualizaciones para cada paso
    """
    
    def __init__(self, visualization_dir: str = "visualization_results"):
        """
        Inicializa el procesador de templates.
        
        Args:
            visualization_dir: Directorio donde se guardarán las visualizaciones
        """
        self.visualization_dir = Path(visualization_dir)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        
        # Ruta al archivo de datos pre-calculados
        self.template_data_path = Path(__file__).parent.parent.parent.parent / "tools" / "template_analysis" / "template_analysis_results.json"
        self.template_data = self._load_template_data_file()
        
    def _load_template_data_file(self) -> Dict:
        """
        Carga el archivo JSON con los datos pre-calculados de los templates.
        
        Returns:
            Dict con los datos de todos los templates
        """
        try:
            if not self.template_data_path.exists():
                print(f"Advertencia: No se encontró el archivo de datos pre-calculados en {self.template_data_path}")
                return {}
                
            with open(self.template_data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error cargando archivo de datos pre-calculados: {str(e)}")
            return {}
    
    def validate_coord_name(self, coord_name: str) -> None:
        """
        Valida que el nombre de coordenada sea coord1 o coord2.
        
        Args:
            coord_name: Nombre de la coordenada a validar
            
        Raises:
            ValueError: Si el nombre de coordenada no es válido
        """
        if coord_name.lower() not in ["coord1", "coord2"]:
            raise ValueError(f"Solo se permiten coord1 y coord2. Recibido: {coord_name}")

    def load_template_data(self, coord_name: str) -> Optional[Dict]:
        """
        Obtiene los datos pre-calculados para una coordenada específica.
        
        Args:
            coord_name: Nombre de la coordenada (e.g., "coord1")
            
        Returns:
            Dict con los datos del template o None si no existe
            
        Raises:
            ValueError: Si el nombre de coordenada no es válido
        """
        self.validate_coord_name(coord_name)
        return self.template_data.get(coord_name)
        
    def validate_intersection_point(self, x: int, y: int, a: int, b: int, c: int, d: int) -> bool:
        """
        Valida que el punto de intersección está dentro de los límites válidos.
        
        Args:
            x: Coordenada x del punto de intersección (d)
            y: Coordenada y del punto de intersección (c)
            a,b,c,d: Distancias calculadas
            
        Returns:
            bool: True si el punto es válido, False en caso contrario
        """
        # El punto (d,c) siempre es válido si las distancias son válidas
        return True

    def calculate_template_distances(self, 
                                   search_region: np.ndarray, 
                                   template_size: int = 64) -> Tuple[int, int, int, int]:
        """
        Calcula las distancias a,b,c,d desde la región de búsqueda al template original.
        
        Args:
            search_region: Matriz binaria con la región de búsqueda
            template_size: Tamaño del template original
            
        Returns:
            Tuple con las distancias (a,b,c,d)
            
        Raises:
            ValueError: Si las dimensiones son inválidas o la región está vacía
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
        
        # Calcular distancias con límites
        a = min_y  # Distancia desde el borde superior al inicio de la región
        d = min_x  # Distancia desde el borde izquierdo al inicio de la región
        
        # Calcular ancho y alto de la región
        region_width = max_x - min_x + 1
        region_height = max_y - min_y + 1
        
        # Calcular distancias al final de la región
        b = 63 - max_x  # Distancia desde el final de la región al borde derecho
        c = 63 - max_y  # Distancia desde el final de la región al borde inferior
        
        # Validar que las distancias son válidas
        if a + c >= 64:
            raise ValueError(f"Las distancias verticales a({a})+c({c}) suman más que el tamaño del template")
        if b + d >= 64:
            raise ValueError(f"Las distancias horizontales b({b})+d({d}) suman más que el tamaño del template")
            
        return a, b, c, d
    
    def create_cutting_template(self, 
                              a: int, 
                              b: int, 
                              c: int, 
                              d: int, 
                              template_size: int = 64) -> np.ndarray:
        """
        Crea el template de recorte basado en las distancias calculadas.
        
        Args:
            a: Distancia al borde superior
            b: Distancia al borde derecho
            c: Distancia al borde inferior
            d: Distancia al borde izquierdo
            template_size: Tamaño del template
            
        Returns:
            Matriz binaria con el template de recorte
            
        Raises:
            ValueError: Si las dimensiones son inválidas
        """
        template = np.zeros((template_size, template_size))
        
        # Calcular dimensiones iniciales del cuadrilátero
        height = c + a  # Suma de las distancias verticales
        width = b + d   # Suma de las distancias horizontales
        
        # Validar dimensiones básicas
        if height <= 0:
            raise ValueError(f"Altura inválida: {height} (a={a}, c={c})")
        if width <= 0:
            raise ValueError(f"Ancho inválido: {width} (b={b}, d={d})")
        
        # Validar y ajustar coordenadas para mantener dentro del template
        if a < 0:
            a = 0
        if d < 0:
            d = 0
            
        # Ajustar dimensiones si exceden los límites
        if a + height > template_size:
            # Reducir altura manteniendo proporción entre a y c
            total = a + c
            if total > 0:
                ratio_a = a / total
                ratio_c = c / total
                new_height = template_size - a
                c = int(new_height * ratio_c)
                height = c + a
        
        if d + width > template_size:
            # Reducir ancho manteniendo proporción entre b y d
            total = b + d
            if total > 0:
                ratio_d = d / total
                ratio_b = b / total
                new_width = template_size - d
                b = int(new_width * ratio_b)
                width = b + d
        
        # Verificar dimensiones finales
        if a >= template_size or d >= template_size:
            raise ValueError(f"Coordenadas fuera de rango: a={a}, d={d}")
        if height <= 0 or width <= 0:
            raise ValueError(f"Dimensiones inválidas después de ajuste: {width}x{height}")
        if a + height > template_size or d + width > template_size:
            raise ValueError(f"Template excede límites después de ajuste: ({d},{a}) + {width}x{height}")
        
        # Crear template desde el punto (d,a) con las dimensiones ajustadas
        template[a:a+height, d:d+width] = 1
        
        return template
    
    def transform_intersection_point(self,
                                   local_point: Tuple[int, int],
                                   template: np.ndarray) -> Tuple[int, int]:
        """
        Transforma el punto de intersección del sistema local al sistema 64x64.
        
        Args:
            local_point: Punto (d,a) en coordenadas del template recortado
            template: Template completo 64x64
            
        Returns:
            Punto transformado al sistema 64x64
        """
        # Obtener los límites del template
        non_zero = np.nonzero(template)
        min_y = non_zero[0].min()  # Offset vertical
        min_x = non_zero[1].min()  # Offset horizontal
        
        # El punto (d,a) en el sistema local se transforma sumando los offsets
        x = min_x + local_point[0]  # Añadir offset horizontal a d
        y = min_y + local_point[1]  # Añadir offset vertical a a
        
        return (x, y)
    
    def find_intersection_point(self, 
                              cutting_template: np.ndarray,
                              a: int = None,
                              b: int = None,
                              c: int = None,
                              d: int = None) -> Tuple[int, int]:
        """
        Encuentra el punto de intersección en el template de recorte.
        
        Args:
            cutting_template: Template de recorte
            a,b,c,d: Distancias calculadas
            
        Returns:
            Tuple con las coordenadas (x,y) del punto de intersección en sistema local
            
        Raises:
            ValueError: Si el template está vacío o faltan distancias
        """
        if any(x is None for x in [a, b, c, d]):
            raise ValueError("Se requieren todas las distancias (a,b,c,d)")
            
        # Validar que el template no está vacío
        non_zero = np.nonzero(cutting_template)
        if len(non_zero[0]) == 0:
            raise ValueError("Template de recorte vacío")
            
        # El punto de intersección es (d,a) en el sistema de coordenadas local del template
        x = d  # Coordenada x es d (distancia desde el borde izquierdo)
        y = a  # Coordenada y es a (distancia desde el borde superior)
        
        return x, y
    
    def validate_coord_num(self, coord_num: int) -> None:
        """
        Valida que el número de coordenada sea 1 o 2.
        
        Args:
            coord_num: Número de coordenada a validar
            
        Raises:
            ValueError: Si el número de coordenada no es 1 o 2
        """
        if coord_num not in [1, 2]:
            raise ValueError(f"Solo se permiten los puntos 1 y 2. Recibido: {coord_num}")

    def visualize_search_region(self, 
                              search_region: np.ndarray, 
                              coord_num: int) -> None:
        """
        Visualiza la región de búsqueda.
        
        Args:
            search_region: Matriz binaria con la región de búsqueda
            coord_num: Número de coordenada (1 o 2)
            
        Raises:
            ValueError: Si el número de coordenada no es 1 o 2
        """
        self.validate_coord_num(coord_num)
        plt.figure(figsize=(10, 10))
        sns.heatmap(search_region, cmap='RdYlBu_r', cbar=False, square=True)
        plt.title(f'Región de Búsqueda - Coord{coord_num}')
        plt.savefig(self.visualization_dir / f'step1_search_region_coord{coord_num}.png')
        plt.close()
    
    def visualize_distances(self, 
                          search_region: np.ndarray,
                          a: int, 
                          b: int, 
                          c: int, 
                          d: int, 
                          coord_num: int) -> None:
        """
        Visualiza las distancias calculadas sobre la región de búsqueda.
        
        Args:
            search_region: Matriz binaria con la región de búsqueda
            a,b,c,d: Distancias calculadas
            coord_num: Número de coordenada (1 o 2)
            
        Raises:
            ValueError: Si el número de coordenada no es 1 o 2
        """
        self.validate_coord_num(coord_num)
        plt.figure(figsize=(10, 10))
        plt.imshow(search_region, cmap='RdYlBu_r')
        
        # Calcular dimensiones del cuadrilátero
        height = c + a  # Suma de las distancias verticales
        width = b + d   # Suma de las distancias horizontales
        
        # Crear template temporal para transformación
        template = np.zeros_like(search_region)
        template[a:a+height, d:d+width] = 1
        
        # Obtener punto de intersección en coordenadas locales y transformarlo
        local_point = (d, a)
        global_point = self.transform_intersection_point(local_point, template)
        
        # Dibujar el cuadrilátero
        rect = plt.Rectangle((d, a), width, height, 
                           fill=False, color='r', linestyle='--', 
                           label=f'Template {width}x{height}')
        plt.gca().add_patch(rect)
        
        # Dibujar punto de intersección transformado
        plt.plot(global_point[0], global_point[1], 'r*', markersize=15, 
                label=f'Punto de intersección (d={d},a={a})\n' +
                      f'Transformado a ({global_point[0]},{global_point[1]})')
        
        # Dibujar líneas de distancia
        plt.annotate(f'a={a}', xy=(d+width/2, a), xytext=(d+width/2, a-10),
                    ha='center', va='top', color='blue',
                    arrowprops=dict(arrowstyle='->'))
        plt.annotate(f'b={b}', xy=(d+width, a+height/2), xytext=(d+width+10, a+height/2),
                    ha='left', va='center', color='blue',
                    arrowprops=dict(arrowstyle='->'))
        plt.annotate(f'c={c}', xy=(d+width/2, a+height), xytext=(d+width/2, a+height+10),
                    ha='center', va='bottom', color='blue',
                    arrowprops=dict(arrowstyle='->'))
        plt.annotate(f'd={d}', xy=(d, a+height/2), xytext=(d-10, a+height/2),
                    ha='right', va='center', color='blue',
                    arrowprops=dict(arrowstyle='->'))
        
        plt.legend()
        plt.title(f'Distancias Template - Coord{coord_num}')
        plt.savefig(self.visualization_dir / f'step2_distances_coord{coord_num}.png')
        plt.close()
    
    def visualize_cutting_template(self, 
                                 template: np.ndarray, 
                                 coord_num: int) -> None:
        """
        Visualiza el template de recorte.
        
        Args:
            template: Template de recorte
            coord_num: Número de coordenada (1 o 2)
            
        Raises:
            ValueError: Si el número de coordenada no es 1 o 2
        """
        self.validate_coord_num(coord_num)
        plt.figure(figsize=(10, 10))
        plt.imshow(template, cmap='binary')
        plt.title(f'Template de Recorte - Coord{coord_num}')
        plt.colorbar(label='Área de recorte')
        plt.savefig(self.visualization_dir / f'step3_template_coord{coord_num}.png')
        plt.close()
    
    def visualize_intersection(self, 
                             template: np.ndarray,
                             intersection_point: Tuple[int, int],
                             coord_num: int) -> None:
        """
        Visualiza el punto de intersección sobre el template de recorte.
        
        Args:
            template: Template de recorte
            intersection_point: Coordenadas del punto de intersección (d,a)
            coord_num: Número de coordenada (1 o 2)
            
        Raises:
            ValueError: Si el número de coordenada no es 1 o 2
        """
        self.validate_coord_num(coord_num)
        # Obtener solo la región del template de recorte
        non_zero = np.nonzero(template)
        min_y, max_y = non_zero[0].min(), non_zero[0].max()
        min_x, max_x = non_zero[1].min(), non_zero[1].max()
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        
        # Extraer solo el template de recorte
        cutting_template = template[min_y:max_y+1, min_x:max_x+1]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(cutting_template, cmap='binary')
        
        # El punto de intersección (d,a) ya está en coordenadas del template
        x, y = intersection_point
        
        # Dibujar el punto de intersección
        plt.plot(x, y, 'r*', 
                markersize=15, label=f'Punto de intersección (d={x},a={y})')
        
        # Dibujar líneas de referencia
        plt.axhline(y=y, color='r', linestyle='--', alpha=0.3)
        plt.axvline(x=x, color='r', linestyle='--', alpha=0.3)
        
        # Agregar dimensiones del template
        plt.title(f'Template de Recorte ({width}x{height}) - Coord{coord_num}')
        plt.xlabel('Ancho del template (d)')
        plt.ylabel('Alto del template (a)')
        
        # Agregar anotaciones
        plt.text(width/2, -0.5, f'Ancho: {width}px', 
                ha='center', va='top')
        plt.text(-0.5, height/2, f'Alto: {height}px', 
                ha='right', va='center', rotation=90)
        
        plt.legend()
        plt.savefig(self.visualization_dir / f'step4_intersection_coord{coord_num}.png')
        plt.close()
    
    def validate_template_bounds(self, template, labeled_point, intersection_point):
        """
        Valida que el template mantenga sus dimensiones dentro de los límites.
        
        Args:
            template: Template de recorte
            labeled_point: Coordenadas del punto etiquetado
            intersection_point: Coordenadas del punto de intersección
            
        Returns:
            Dict con información de dimensiones y límites validados
            
        Raises:
            ValueError: Si las dimensiones o coordenadas son inválidas
        """
        # Obtener dimensiones del template
        non_zero = np.nonzero(template)
        min_y, max_y = non_zero[0].min(), non_zero[0].max()
        min_x, max_x = non_zero[1].min(), non_zero[1].max()
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        
        # Verificar dimensiones originales
        if width > 64 or height > 64:
            raise ValueError(f"Template original excede límites: {width}x{height}")
        
        # Verificar punto de intersección
        if not (0 <= intersection_point[0] < 64 and 0 <= intersection_point[1] < 64):
            raise ValueError(f"Punto de intersección fuera de límites: {intersection_point}")
        
        # Verificar punto etiquetado
        if not (0 <= labeled_point[0] < 64 and 0 <= labeled_point[1] < 64):
            raise ValueError(f"Punto etiquetado fuera de límites: {labeled_point}")
        
        return {
            'width': width,
            'height': height,
            'original_bounds': {
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y
            }
        }

    def visualize_alignment(self,
                          image: np.ndarray,
                          template: np.ndarray,
                          labeled_point: Tuple[int, int],
                          intersection_point: Tuple[int, int],
                          coord_num: int) -> None:
        """
        Visualiza la alineación del template con el punto etiquetado.
        
        Args:
            image: Imagen original
            template: Template de recorte
            labeled_point: Coordenadas del punto etiquetado
            intersection_point: Coordenadas del punto de intersección local (d,a)
            coord_num: Número de coordenada (1 o 2)
            
        Raises:
            ValueError: Si el número de coordenada no es 1 o 2
        """
        self.validate_coord_num(coord_num)
        try:
            # 1. Logging inicial
            print(f"\nDEBUG - Coord{coord_num} Alignment:")
            print(f"1. Template shape: {template.shape}")
            print(f"2. Labeled point: {labeled_point}")
            print(f"3. Intersection point: {intersection_point}")
            
            # 2. Validar datos de entrada
            validation = self.validate_template_bounds(template, labeled_point, intersection_point)
            width = validation['width']
            height = validation['height']
            bounds = validation['original_bounds']
            
            print(f"4. Original template bounds:")
            print(f"   - X: {bounds['min_x']} to {bounds['max_x']} (width: {width})")
            print(f"   - Y: {bounds['min_y']} to {bounds['max_y']} (height: {height})")
            
            # 3. Transformar punto de intersección a coordenadas globales
            global_intersection = (
                bounds['min_x'] + intersection_point[0],
                bounds['min_y'] + intersection_point[1]
            )
            
            print(f"5. Coordinate systems:")
            print(f"   Local intersection point: {intersection_point}")
            print(f"   Template bounds: ({bounds['min_x']}, {bounds['min_y']}) to ({bounds['max_x']}, {bounds['max_y']})")
            print(f"   Global intersection point: {global_intersection}")
            
            # 4. Calcular desplazamiento necesario
            dx = labeled_point[0] - global_intersection[0]
            dy = labeled_point[1] - global_intersection[1]
            
            print(f"6. Initial displacement:")
            print(f"   - dx: {dx}")
            print(f"   - dy: {dy}")
            
            # 5. Calcular coordenadas finales directamente
            final_min_x = bounds['min_x'] + dx
            final_max_x = final_min_x + width
            final_min_y = bounds['min_y'] + dy
            final_max_y = final_min_y + height
            
            # 6. Verificar y ajustar si es necesario mantener dentro de 64x64
            if final_min_x < 0 or final_max_x > 64 or final_min_y < 0 or final_max_y > 64:
                print("Advertencia: Ajustando template para mantener dentro de 64x64")
                # Calcular el centro del template
                template_center_x = (final_min_x + final_max_x) // 2
                template_center_y = (final_min_y + final_max_y) // 2
                
                # Ajustar manteniendo el centro lo más cerca posible al punto etiquetado
                if final_min_x < 0:
                    shift_x = -final_min_x
                    final_min_x = 0
                    final_max_x = width
                elif final_max_x > 64:
                    shift_x = 64 - final_max_x
                    final_max_x = 64
                    final_min_x = 64 - width
                    
                if final_min_y < 0:
                    shift_y = -final_min_y
                    final_min_y = 0
                    final_max_y = height
                elif final_max_y > 64:
                    shift_y = 64 - final_max_y
                    final_max_y = 64
                    final_min_y = 64 - height
                
                print(f"   Ajuste aplicado:")
                print(f"   - Nuevos límites X: {final_min_x} to {final_max_x}")
                print(f"   - Nuevos límites Y: {final_min_y} to {final_max_y}")
            
            # 7. Calcular punto de intersección final
            final_intersection = (
                final_min_x + intersection_point[0],
                final_min_y + intersection_point[1]
            )
            
            print(f"8. Final coordinates:")
            print(f"   Template bounds: X: {final_min_x} to {final_max_x}")
            print(f"   Template bounds: Y: {final_min_y} to {final_max_y}")
            print(f"   Final intersection point: {final_intersection}")
            
            # 9. Verificación final
            assert 0 <= final_min_x < final_max_x <= 64, "Error en límites horizontales"
            assert 0 <= final_min_y < final_max_y <= 64, "Error en límites verticales"
            assert final_max_x - final_min_x == width, "Error en ancho del template"
            assert final_max_y - final_min_y == height, "Error en alto del template"
            
            # 7. Crear visualización
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Imagen original con punto etiquetado
            ax1.imshow(image, cmap='gray')
            ax1.plot(labeled_point[0], labeled_point[1], 'r*', 
                    markersize=15, label=f'Punto etiquetado ({labeled_point[0]},{labeled_point[1]})')
            ax1.axhline(y=labeled_point[1], color='r', linestyle='--', alpha=0.3)
            ax1.axvline(x=labeled_point[0], color='r', linestyle='--', alpha=0.3)
            ax1.set_title('Imagen Original con Punto Etiquetado')
            ax1.legend()
            
            # Template alineado
            ax2.imshow(image, cmap='gray', label='Imagen Original')
            aligned_template = np.zeros_like(template)
            aligned_template[final_min_y:final_max_y, final_min_x:final_max_x] = 1
            template_overlay = ax2.imshow(aligned_template, cmap='binary', alpha=0.3, label='Template Alineado')
            
            # Punto de intersección alineado
            intersection_marker = ax2.plot(final_intersection[0], final_intersection[1], 'r*', 
                                        markersize=15, label=f'Punto de intersección\n({final_intersection[0]},{final_intersection[1]})')
            
            # Punto etiquetado
            labeled_marker = ax2.plot(labeled_point[0], labeled_point[1], 'b*',
                                    markersize=15, label=f'Punto etiquetado\n({labeled_point[0]},{labeled_point[1]})')
            
            # Líneas de referencia
            h_line = ax2.axhline(y=labeled_point[1], color='r', linestyle='--', alpha=0.3, label='Líneas de referencia')
            ax2.axvline(x=labeled_point[0], color='r', linestyle='--', alpha=0.3)
            
            ax2.set_title(f'Template de Recorte Alineado ({width}x{height})')
            
            # Crear leyenda con elementos relevantes
            legend_elements = [
                template_overlay,
                intersection_marker[0],
                labeled_marker[0],
                h_line
            ]
            ax2.legend(handles=legend_elements)
            
            plt.savefig(self.visualization_dir / f'step5_alignment_coord{coord_num}.png')
            plt.close()
            
            print("8. Visualización completada exitosamente")
            
        except Exception as e:
            print(f"Error en visualize_alignment: {str(e)}")
            raise
    
    def crop_aligned_image(self,
                          image: np.ndarray,
                          template: np.ndarray,
                          labeled_point: Tuple[int, int],
                          intersection_point: Tuple[int, int]) -> np.ndarray:
        """
        Recorta la imagen usando el template alineado con el punto etiquetado.
        
        Args:
            image: Imagen original
            template: Template de recorte
            labeled_point: Coordenadas del punto etiquetado
            intersection_point: Coordenadas del punto de intersección local (d,a)
            
        Returns:
            Imagen recortada del tamaño del template de recorte
        """
        # Obtener dimensiones del template
        non_zero = np.nonzero(template)
        min_y, max_y = non_zero[0].min(), non_zero[0].max()
        min_x, max_x = non_zero[1].min(), non_zero[1].max()
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        
        # Calcular desplazamiento
        dx = labeled_point[0] - (min_x + intersection_point[0])
        dy = labeled_point[1] - (min_y + intersection_point[1])
        
        # Calcular coordenadas finales con límites
        final_min_x = np.clip(min_x + dx, 0, 64 - width)
        final_min_y = np.clip(min_y + dy, 0, 64 - height)
        final_max_x = final_min_x + width
        final_max_y = final_min_y + height
        
        # Recortar y retornar
        return image[final_min_y:final_max_y, final_min_x:final_max_x]
    
    def visualize_final_result(self,
                             image: np.ndarray,
                             template: np.ndarray,
                             labeled_point: Tuple[int, int],
                             intersection_point: Tuple[int, int],
                             coord_num: int) -> None:
        """
        Visualiza el resultado final del recorte.
        
        Args:
            image: Imagen original
            template: Template de recorte
            labeled_point: Coordenadas del punto etiquetado
            intersection_point: Coordenadas del punto de intersección local (d,a)
            coord_num: Número de coordenada (1 o 2)
            
        Raises:
            ValueError: Si el número de coordenada no es 1 o 2
        """
        self.validate_coord_num(coord_num)
        try:
            # Obtener la imagen recortada
            cropped_image = self.crop_aligned_image(image, template, labeled_point, intersection_point)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Imagen original con punto etiquetado y template
            ax1.imshow(image, cmap='gray')
            ax1.plot(labeled_point[0], labeled_point[1], 'r*', 
                    markersize=15, label=f'Punto etiquetado ({labeled_point[0]},{labeled_point[1]})')
            
            # Validar y obtener dimensiones
            validation = self.validate_template_bounds(template, labeled_point, intersection_point)
            width = validation['width']
            height = validation['height']
            bounds = validation['original_bounds']
            
            # Transformar punto de intersección a coordenadas globales
            global_intersection = (
                bounds['min_x'] + intersection_point[0],
                bounds['min_y'] + intersection_point[1]
            )
            
            # Calcular desplazamiento necesario
            dx = labeled_point[0] - global_intersection[0]
            dy = labeled_point[1] - global_intersection[1]
            
            # Ajustar desplazamiento si es necesario
            new_min_x = bounds['min_x'] + dx
            new_min_y = bounds['min_y'] + dy
            
            if new_min_x < 0:
                dx = -bounds['min_x']
            elif new_min_x + width > 64:
                dx = 63 - (bounds['min_x'] + width)
                
            if new_min_y < 0:
                dy = -bounds['min_y']
            elif new_min_y + height > 64:
                dy = 63 - (bounds['min_y'] + height)
            
            # Calcular coordenadas finales
            final_min_x = bounds['min_x'] + dx
            final_max_x = final_min_x + width
            final_min_y = bounds['min_y'] + dy
            final_max_y = final_min_y + height
            
            # Calcular punto de intersección final
            final_intersection = (
                final_min_x + intersection_point[0],
                final_min_y + intersection_point[1]
            )
            
            # Dibujar el template alineado
            aligned_template = np.zeros_like(template)
            aligned_template[final_min_y:final_max_y, final_min_x:final_max_x] = 1
            
            # Imagen original con template y puntos
            ax1.imshow(image, cmap='gray')
            ax1.imshow(aligned_template, cmap='binary', alpha=0.3, label='Template Alineado')
            ax1.plot(labeled_point[0], labeled_point[1], 'b*',
                    markersize=15, label=f'Punto etiquetado\n({labeled_point[0]},{labeled_point[1]})')
            ax1.plot(final_intersection[0], final_intersection[1], 'r*',
                    markersize=15, label=f'Punto de intersección\n({final_intersection[0]},{final_intersection[1]})')
            ax1.set_title('Imagen Original con Template Alineado')
            ax1.legend()
            
            # Imagen recortada
            ax2.imshow(cropped_image, cmap='gray')
            ax2.plot(width//2, height//2, 'r*',
                    markersize=15, label=f'Centro de la región recortada')
            ax2.set_title(f'Imagen Recortada ({width}x{height})')
            ax2.legend()
            
            plt.savefig(self.visualization_dir / f'step6_final_result_coord{coord_num}.png')
            plt.close()
            
        except ValueError as e:
            print(f"Error al recortar la imagen: {str(e)}")
