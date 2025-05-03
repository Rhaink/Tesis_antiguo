#!/usr/bin/env python3
"""
Script principal para ejecutar el programa de etiquetado.
MODIFICADO PARA ETIQUETAR IMÁGENES NUEVAS.
"""
import os
import sys # Necesario si se ajusta el path
import pandas as pd
import numpy as np

# --- Ajustar sys.path si es necesario para encontrar 'etiquetador' ---
# Obtener la ruta absoluta del directorio del script actual (pulmon)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Obtener ruta del directorio padre (Tesis)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 
# Agregar el directorio que contiene 'etiquetador' al path si no está en 'pulmon'
# Si 'etiquetador' está directamente dentro de 'Tesis', esta línea es necesaria:
# sys.path.append(PROJECT_ROOT) 
# Si 'etiquetador' está dentro de 'pulmon', el import relativo podría funcionar mejor.
try:
    from etiquetador.main import ImageAnnotator
except ImportError:
    print("Error: No se pudo importar ImageAnnotator.")
    print("Asegúrate de que el paquete 'etiquetador' sea accesible desde:", SCRIPT_DIR)
    print("Puede que necesites ajustar sys.path o usar imports relativos.")
    sys.exit(1)
# --- Fin ajuste sys.path ---

def main():
    """Función principal del programa."""
    # Obtener directorio base del proyecto (Tesis)
    base_dir = PROJECT_ROOT # Usamos el directorio padre del script
    
    # --- CONFIGURACIÓN PARA NUEVAS IMÁGENES ---
    # 1. Archivo con los índices de las imágenes a etiquetar
    archivo_indices = os.path.join(base_dir, "indices/indices_nuevas_500.csv") 
    
    # 2. Nombre base para los NUEVOS archivos de coordenadas (se guardarán en Tesis/)
    archivo_coordenadas_base = os.path.join(base_dir, "coordenadas/coordenadas_nuevas_500") 
    # --- FIN CONFIGURACIÓN ---

    print(f"Intentando leer índices de: {archivo_indices}")
    print(f"Las coordenadas nuevas se guardarán con base: {os.path.basename(archivo_coordenadas_base)}")

    # Leer índices de imágenes
    try:
        # Especificar dtype puede ayudar si hay IDs grandes o mezcla de tipos
        data_indices = pd.read_csv(archivo_indices, header=None, dtype={0: int, 1: int, 2: object}) 
        # Convertir la columna de ID (col 2) a int después de cargarla como objeto para manejar posibles errores
        data_indices[2] = pd.to_numeric(data_indices[2], errors='coerce').astype('Int64') # Permite NaN si hay error
        # Eliminar filas donde el ID no pudo ser convertido
        original_len = len(data_indices)
        data_indices.dropna(subset=[2], inplace=True)
        if len(data_indices) < original_len:
             print(f"Advertencia: Se eliminaron {original_len - len(data_indices)} filas del archivo de índices debido a IDs no numéricos.")
        
    except FileNotFoundError:
        print(f"Error CRÍTICO: No se encontró el archivo de índices '{archivo_indices}'.")
        print("Asegúrate de haber ejecutado el script para crear 'indices_nuevas_500.csv' y que esté en el directorio correcto.")
        return
    except Exception as e:
        print(f"Error al leer el archivo de índices '{archivo_indices}': {e}")
        return
        
    indices = data_indices.to_numpy()
    
    # 3. Usar la longitud real del archivo leído
    num_imagenes_a_etiquetar = len(indices) 
    print(f"Se procesarán {num_imagenes_a_etiquetar} imágenes.")

    if num_imagenes_a_etiquetar == 0:
        print("El archivo de índices está vacío o no contiene IDs válidos.")
        return

    # Generar rutas de imágenes
    image_paths = []
    dataset_base_path = os.path.join(base_dir, "COVID-19_Radiography_Dataset") # Ruta base del dataset

    for i in range(num_imagenes_a_etiquetar): 
        # Acceder a las columnas por índice numérico (0=indice_nuevo, 1=categoria, 2=id_imagen)
        categoria = indices[i, 1]
        img_id = indices[i, 2] 
        
        # Construir el nombre del archivo y la ruta completa
        path = None
        if categoria == 1:
            filename = f"COVID-{img_id}.png"
            path = os.path.join(dataset_base_path, "COVID/images", filename)
        elif categoria == 2:
            filename = f"Normal-{img_id}.png"
            path = os.path.join(dataset_base_path, "Normal/images", filename)
        elif categoria == 3:
            filename = f"Viral Pneumonia-{img_id}.png"
            path = os.path.join(dataset_base_path, "Viral Pneumonia/images", filename)
        else:
            print(f"Advertencia: Categoría desconocida '{categoria}' en la fila {i} del archivo de índices. Omitiendo.")
            continue 
        
        # Verificar si el archivo de imagen realmente existe antes de añadirlo
        if os.path.exists(path):
            image_paths.append(path)
        else:
            print(f"Advertencia: No se encontró el archivo de imagen esperado: {path}. Omitiendo.")

    # Verificar si se generaron rutas válidas
    if not image_paths:
        print("Error: No se generaron rutas de imágenes válidas. Revisa el archivo de índices, las categorías y la existencia de los archivos de imagen.")
        return

    print(f"\nSe encontraron {len(image_paths)} imágenes válidas para etiquetar.")

    # Iniciar programa de etiquetado
    print("\nSe guardarán las coordenadas en las siguientes resoluciones:")
    print(f"- 64x64 pixels ({os.path.basename(archivo_coordenadas_base)}_64x64.csv)")
    print(f"- 128x128 pixels ({os.path.basename(archivo_coordenadas_base)}_128x128.csv)")
    print(f"- 256x256 pixels ({os.path.basename(archivo_coordenadas_base)}_256x256.csv)")
    print("\nIniciando programa de etiquetado...\n")
    
    # Crear e iniciar el anotador (Asegúrate que el import al inicio sea correcto)
    annotator = ImageAnnotator(image_paths, archivo_coordenadas_base)
    annotator.run()

    print("\nProceso de etiquetado finalizado.")

if __name__ == "__main__":
    main()