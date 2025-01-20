#!/usr/bin/env python3
"""
Script principal para ejecutar el programa de etiquetado.
"""
import os
import pandas as pd
import numpy as np
from etiquetador.main import ImageAnnotator

def main():
    """Función principal del programa."""
    # Obtener directorio base
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Configuración de archivos con rutas absolutas
    archivo_coordenadas_base = os.path.join(base_dir, "coordenadas")  # Archivo base para guardar coordenadas
    archivo_indices = os.path.join(base_dir, "indices.csv")  # Archivo con índices de imágenes
    
    # Leer índices de imágenes
    data_indices = pd.read_csv(archivo_indices, header=None)
    indices = np.array(data_indices)
    
    # Generar rutas de imágenes
    image_paths = []
    for i in range(100):  # Procesar primeras 100 imágenes
        if indices[i,1] == 1:
            path = os.path.join(base_dir, "COVID-19_Radiography_Dataset/COVID/images", f"COVID-{indices[i,2]}.png")
        elif indices[i,1] == 2:
            path = os.path.join(base_dir, "COVID-19_Radiography_Dataset/Normal/images", f"Normal-{indices[i,2]}.png")
        elif indices[i,1] == 3:
            path = os.path.join(base_dir, "COVID-19_Radiography_Dataset/Viral Pneumonia/images", f"Viral Pneumonia-{indices[i,2]}.png")
        image_paths.append(path)
    
    # Iniciar programa de etiquetado
    print("\nSe guardarán las coordenadas en las siguientes resoluciones:")
    print("- 64x64 pixels (coordenadas_64x64.csv)")
    print("- 128x128 pixels (coordenadas_128x128.csv)")
    print("- 256x256 pixels (coordenadas_256x256.csv)")
    print("\nIniciando programa de etiquetado...\n")
    
    annotator = ImageAnnotator(image_paths, archivo_coordenadas_base)
    annotator.run()

if __name__ == "__main__":
    main()
