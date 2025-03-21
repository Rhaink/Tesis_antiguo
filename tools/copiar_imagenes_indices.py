import os
import pandas as pd
import shutil

# Definir rutas
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
indices_file = os.path.join(base_dir, "indices/indices.csv")
source_dir = os.path.join(base_dir, "COVID-19_Radiography_Dataset")
dest_dir = os.path.join(base_dir, "dataset/puntos_interes_indices_prueba_2")

# Crear la carpeta destino si no existe
os.makedirs(dest_dir, exist_ok=True)

# Leer los índices
data_indices = pd.read_csv(indices_file, header=None)

# Recorrer los índices y copiar imágenes
for _, row in data_indices.iterrows():
    categoria = row[1]
    indice_img = row[2]
    
    if categoria == 1:
        source_path = os.path.join(source_dir, "COVID/images", f"COVID-{indice_img}.png")
    elif categoria == 2:
        source_path = os.path.join(source_dir, "Normal/images", f"Normal-{indice_img}.png")
    elif categoria == 3:
        source_path = os.path.join(source_dir, "Viral Pneumonia/images", f"Viral Pneumonia-{indice_img}.png")
    else:
        continue  # Si la categoría no es válida, pasar a la siguiente
    
    # Definir la ruta de destino
    dest_path = os.path.join(dest_dir, os.path.basename(source_path))
    
    # Copiar la imagen si existe
    if os.path.exists(source_path):
        shutil.copy2(source_path, dest_path)
        print(f"Copiada: {source_path} -> {dest_path}")
    else:
        print(f"No encontrada: {source_path}")

print("Proceso de copiado completado.")
