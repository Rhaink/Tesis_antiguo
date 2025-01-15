#!/usr/bin/env python3
"""
Script principal para ejecutar el programa de etiquetado.
"""
import os
import pandas as pd
import numpy as np
from etiquetador.main import ImageAnnotator

def main():
    """Función principal del programa."""#prueba2
    # Configuración de archivos
    archivo_coordenadas = "Tesis/coordenadas.csv"  # Archivo para guardar coordenadas
    archivo_indices = "Tesis/indices.csv"  # Archivo con índices de imágenes
    
    # Leer índices de imágenes
    data_indices = pd.read_csv(archivo_indices, header=None)
    indices = np.array(data_indices)
    
    # Generar rutas de imágenes
    image_paths = []
    for i in range(100):  # Procesar primeras 100 imágenes
        if indices[i,1] == 1:
            path = f"Tesis/COVID-19_Radiography_Dataset/COVID/images/COVID-{indices[i,2]}.png"
        elif indices[i,1] == 2:
            path = f"Tesis/COVID-19_Radiography_Dataset/Normal/images/Normal-{indices[i,2]}.png"
        elif indices[i,1] == 3:
            path = f"Tesis/COVID-19_Radiography_Dataset/Viral Pneumonia/images/Viral Pneumonia-{indices[i,2]}.png"
        image_paths.append(path)
    
    # Iniciar programa de etiquetado
    annotator = ImageAnnotator(image_paths, archivo_coordenadas)
    annotator.run()

if __name__ == "__main__":
    main()
