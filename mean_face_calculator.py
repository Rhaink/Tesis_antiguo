#!/usr/bin/env python3
"""
Programa para calcular el rostro medio de los puntos anatómicos 1 y 2 en radiografías pulmonares.
"""

import os
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Usar backend que no requiere GUI
import matplotlib.pyplot as plt

class ContrastEnhancer:
    """
    Implementación del algoritmo SAHS para mejoramiento de contraste.
    """
    
    @staticmethod
    def enhance_contrast_sahs(image):
        """
        Aplica el algoritmo SAHS para mejorar el contraste de la imagen.
        
        Args:
            image (np.ndarray): Imagen de entrada en escala de grises
            
        Returns:
            np.ndarray: Imagen con contraste mejorado
        """
        try:
            if image is None:
                raise ValueError("La imagen de entrada es None")
            
            # Asegurar que la imagen está en escala de grises
            if len(image.shape) > 2:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image.copy()
            
            # Calcular la media de los niveles de gris
            gray_mean = np.mean(gray_image)
            
            # Separar píxeles por encima y debajo de la media
            above_mean = gray_image[gray_image > gray_mean]
            below_or_equal_mean = gray_image[gray_image <= gray_mean]
            
            # Calcular límites usando desviación estándar asimétrica
            max_value = gray_mean
            min_value = gray_mean
            
            if above_mean.size > 0:
                # Factor 2.5 para el límite superior
                std_above = np.sqrt(np.mean((above_mean - gray_mean) ** 2))
                max_value = gray_mean + 2.5 * std_above
                
            if below_or_equal_mean.size > 0:
                # Factor 2.0 para el límite inferior
                std_below = np.sqrt(np.mean((below_or_equal_mean - gray_mean) ** 2))
                min_value = gray_mean - 2.0 * std_below
            
            # Normalizar al rango [0, 255]
            if max_value != min_value:
                enhanced_image = np.clip(
                    (255 / (max_value - min_value)) * (gray_image - min_value),
                    0, 255
                ).astype(np.uint8)
            else:
                enhanced_image = gray_image
                
            return enhanced_image
            
        except Exception as e:
            print(f"Error en enhance_contrast_sahs: {str(e)}")
            return None

class MeanFaceCalculator:
    def __init__(self, coordinates_file, indices_file, base_path):
        """
        Inicializa el calculador de rostro medio.
        
        Args:
            coordinates_file (str): Ruta al archivo CSV con las coordenadas
            indices_file (str): Ruta al archivo CSV con los índices
            base_path (str): Ruta base al dataset de imágenes
        """
        # Cargar coordenadas
        self.coordinates = pd.read_csv(coordinates_file, header=None)
        # Asignar nombres a las columnas: índice, coordenadas (x1,y1,...,x15,y15), ImageId
        coord_columns = [f'{"x" if i%2==0 else "y"}{(i//2)+1}' for i in range(30)]
        self.coordinates.columns = ['index'] + coord_columns + ['ImageId']
        
        # Cargar índices y establecer el índice del DataFrame
        self.indices = pd.read_csv(indices_file, header=None, names=['index', 'category', 'image_number'])
        self.indices.set_index('index', inplace=True)
        
        self.base_path = Path(base_path)
        
        # Definición de las regiones de interés para puntos 1 y 2 (según all_search_coordinates.json)
        self.roi_params = {
            1: {"width": 17, "height": 17},  # Punto 1: coordenadas x[1-17], y[23-39]
            2: {"width": 25, "height": 13}   # Punto 2: coordenadas x[39-63], y[27-39]
        }

    def extract_roi(self, image, center_x, center_y, point_num):
        """
        Extrae una región de interés (ROI) de la imagen usando los parámetros del punto específico.
        
        Args:
            image (np.ndarray): Imagen de entrada
            center_x (int): Coordenada x del centro
            center_y (int): Coordenada y del centro
            point_num (int): Número del punto (1 o 2)
            
        Returns:
            np.ndarray: ROI extraída y normalizada
        """
        params = self.roi_params[point_num]
        width = params["width"]
        height = params["height"]
        
        # Calculamos el centro de la ROI
        width = params["width"]
        height = params["height"]
        
        # Calculamos las coordenadas para el recorte centrado en el punto
        crop_start_x = max(int(center_x - width/2), 0)
        crop_end_x = min(int(center_x + width/2), image.shape[1])
        crop_start_y = max(int(center_y - height/2), 0)
        crop_end_y = min(int(center_y + height/2), image.shape[0])
        
        # Extracción de ROI
        roi = image[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
        
        # Asegurar dimensiones consistentes
        roi = cv2.resize(roi, (width, height))
        
        # Normalización
        roi = roi.astype(float) / 255.0
        
        return roi

    def calculate_mean_face(self, point_num):
        """
        Calcula el rostro medio para un punto específico.
        
        Args:
            point_num (int): Número del punto (1 o 2)
            
        Returns:
            np.ndarray: Rostro medio calculado
        """
        roi_params = self.roi_params[point_num]
        width = roi_params["width"]
        height = roi_params["height"]
        
        # Lista para almacenar todas las ROIs
        all_rois = []
        
        # Extraer ROIs de todas las imágenes
        for _, row in tqdm(self.coordinates.iterrows(), desc=f"Procesando punto {point_num}"):
            try:
                # Obtener información de la imagen desde el archivo de índices
                index = int(row['index'])  # Asegurar que el índice es un entero
                image_info = self.indices.loc[index]  # Acceder directamente por índice
                
                # Determinar la categoría
                if image_info['category'] == 1:
                    category = "COVID"
                elif image_info['category'] == 2:
                    category = "Normal"
                elif image_info['category'] == 3:
                    category = "Viral Pneumonia"
                else:
                    print(f"Categoría no válida para índice {index}")
                    continue
                
                # Construir el nombre del archivo
                image_number = str(image_info['image_number'])
                image_name = f"{category}-{image_number}.png"
            except KeyError:
                print(f"No se encontró el índice {index} en el archivo de índices")
                continue
            except Exception as e:
                print(f"Error procesando índice {index}: {str(e)}")
                continue
            
            # Construir la ruta completa
            image_path = self.base_path / category / "images" / image_name
            if not image_path.exists():
                print(f"No se encontró la imagen: {image_path}")
                continue
                
            # Leer y mejorar contraste de la imagen
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
                
            # Aplicar mejora de contraste SAHS
            enhanced_image = ContrastEnhancer.enhance_contrast_sahs(image)
            if enhanced_image is None:
                print(f"Advertencia: No se pudo mejorar el contraste de {image_path}")
                enhanced_image = image
                
            # Extraer ROI
            x = row[f'x{point_num}']
            y = row[f'y{point_num}']
            roi = self.extract_roi(enhanced_image, x, y, point_num)
            
            all_rois.append(roi)
        
        # Calcular la media
        if all_rois:
            mean_face = np.mean(all_rois, axis=0)
            return mean_face
        else:
            raise ValueError(f"No se pudieron procesar ROIs para el punto {point_num}")

    def save_mean_faces(self, output_dir):
        """
        Calcula y guarda los rostros medios para los puntos 1 y 2.
        
        Args:
            output_dir (str): Directorio donde guardar los resultados
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calcular rostros medios
        mean_faces = {}
        for point_num in [1, 2]:
            try:
                mean_face = self.calculate_mean_face(point_num)
                mean_faces[point_num] = mean_face
                
                # Guardar imagen
                plt.figure(figsize=(5, 5))
                plt.imshow(mean_face, cmap='gray')
                plt.title(f'Rostro Medio - Punto {point_num}')
                plt.colorbar()
                plt.savefig(output_path / f'mean_face_point_{point_num}.png')
                plt.close()
                
                # Guardar array
                np.save(output_path / f'mean_face_point_{point_num}.npy', mean_face)
                
            except Exception as e:
                print(f"Error procesando punto {point_num}: {str(e)}")
        
        # Calcular y guardar la diferencia entre rostros medios
        if len(mean_faces) == 2:
            # Redimensionar el rostro medio 2 al tamaño del rostro medio 1 antes de calcular la diferencia
            face1 = mean_faces[1]
            face2_resized = cv2.resize(mean_faces[2], (face1.shape[1], face1.shape[0]))
            
            # Calcular y guardar la diferencia
            diff = face1 - face2_resized
            plt.figure(figsize=(5, 5))
            plt.imshow(diff, cmap='RdBu')
            plt.title('Diferencia entre Rostros Medios (Punto 1 - Punto 2)')
            plt.colorbar()
            plt.savefig(output_path / 'mean_faces_difference.png')
            plt.close()
            
            # Guardar también las dimensiones originales
            with open(output_path / 'dimensions.txt', 'w') as f:
                f.write(f"Punto 1: {face1.shape}\n")
                f.write(f"Punto 2: {mean_faces[2].shape}\n")

def main():
    """Función principal del programa."""
    try:
        # Configurar rutas
        base_dir = Path(__file__).parent
        coordinates_file = base_dir / "coordenadas_64x64.csv"
        
        # Solicitar al usuario la ubicación de las imágenes
        print("Iniciando cálculo de rostros medios...")
        print(f"Usando archivo de coordenadas: {coordinates_file}")
        
        # Configurar rutas
        base_path = base_dir / "COVID-19_Radiography_Dataset"
        if not base_path.exists():
            raise FileNotFoundError(f"No se encontró el directorio base: {base_path}")
            
        print(f"Usando directorio base: {base_path}")
        
        # Crear directorio de salida
        output_dir = base_dir / "mean_faces_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear y ejecutar el calculador
        # Configurar rutas de archivos
        indices_file = base_dir / "indices.csv"
        if not indices_file.exists():
            raise FileNotFoundError(f"No se encontró el archivo de índices: {indices_file}")
            
        # Crear y ejecutar el calculador
        calculator = MeanFaceCalculator(coordinates_file, indices_file, base_path)
        calculator.save_mean_faces(output_dir)
        
        print(f"\nResultados guardados en: {output_dir}")
        
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")

if __name__ == "__main__":
    main()
