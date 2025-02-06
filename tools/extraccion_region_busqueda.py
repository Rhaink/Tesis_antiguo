import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def generate_search_zone(csv_file, coord_num):
    # Cargar datos del CSV
    df = pd.read_csv(csv_file, header=None)
    x_col = (coord_num - 1) * 2 + 1
    y_col = x_col + 1
    
    # Obtener coordenadas y corregir 0s y valores fuera de rango
    coord_x = df.iloc[:, x_col].values
    coord_y = df.iloc[:, y_col].values
    
    # Convertir 0 → 1 y valores >64 → 64 (1-based)
    coord_x = np.clip(coord_x, 1, 64)
    coord_y = np.clip(coord_y, 1, 64)

    print(f"\nDiagnósticos para Coord{coord_num}:")
    print(f"Número total de puntos: {len(coord_x)}")
    print(f"Rango X : {coord_x.min()} - {coord_x.max()}")
    print(f"Rango Y : {coord_y.min()} - {coord_y.max()}")

    # Crear heatmap 64x64
    heatmap = np.zeros((64, 64))
    for x, y in zip(coord_x, coord_y):
        x_adj, y_adj = x - 1, y - 1  # Convertir a 0-based (0-63)
        heatmap[y_adj, x_adj] += 1  # Ya no hay valores fuera de rango

    # Calcular área de interés
    non_zero = np.nonzero(heatmap)
    if len(non_zero[0]) == 0:
        print("Error: No hay puntos válidos en el heatmap")
        return

    min_y, max_y = non_zero[0].min(), non_zero[0].max()
    min_x, max_x = non_zero[1].min(), non_zero[1].max()

    # Crear zona de búsqueda
    search_zone = np.zeros((64, 64))
    search_zone[min_y:max_y+1, min_x:max_x+1] = 1

    # Obtener coordenadas (1-based)
    search_coordinates = [
        (int(y+1), int(x+1)) 
        for y, x in np.argwhere(search_zone == 1)
    ]

    print(f"\nNúmero de píxeles en zona de búsqueda: {len(search_coordinates)}")

    # Visualización
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        search_zone, 
        cmap='RdYlBu_r', 
        cbar=False, 
        square=True, 
        linewidths=0.5, 
        linecolor='black'
    )
    
    # Ajustar ejes
    xticks_pos = np.arange(0, 65, 8)
    xticks_labels = np.arange(1, 65, 8)
    xticks_labels = np.append(xticks_labels, 65)
    
    plt.xticks(xticks_pos, labels=xticks_labels)
    plt.yticks(xticks_pos, labels=xticks_labels)
    plt.ylim(64, 0)
    
    # Cuadrícula
    for i in range(0, 65, 8):
        plt.axhline(y=i, color='black', linewidth=1.5)
        plt.axvline(x=i, color='black', linewidth=1.5)
    
    plt.title(f'Zona de Búsqueda para Coord{coord_num}')
    plt.xlabel('Coordenada X (1-based)')
    plt.ylabel('Coordenada Y (1-based)')
    
    # Guardar y cerrar
    plt.tight_layout()
    plt.savefig(f'coord{coord_num}_search_zone.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return search_coordinates

# Ejecutar análisis
csv_file = 'Tesis/coordenadas_64x64.csv'
all_search_zones = {}

for coord_num in range(1, 16):
    search_coordinates = generate_search_zone(csv_file, coord_num)
    if search_coordinates:
        all_search_zones[f'coord{coord_num}'] = search_coordinates

# Guardar resultados
with open('all_search_coordinates_prueba2.json', 'w') as f:
    json.dump(all_search_zones, f, indent=2)

print("\n¡Proceso completado! Coordenadas guardadas en 'all_search_coordinates_prueba2.json'")