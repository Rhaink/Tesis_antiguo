import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_coordinates(csv_path):
    # Leer el archivo CSV
    df = pd.read_csv(csv_path, header=None)
    
    # Crear figura y eje
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Obtener regiones únicas
    regions = df[0].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(regions)))
    
    for region_idx, region in enumerate(regions):
        # Filtrar coordenadas para esta región
        region_data = df[df[0] == region].iloc[:, 1:-1]  # Excluir la última columna (ID)
        coords = []
        
        # Convertir pares de coordenadas a puntos
        for i in range(0, len(region_data.columns), 2):
            x = region_data.iloc[:, i]
            y = region_data.iloc[:, i+1]
            coords.extend(zip(x, y))
        
        coords = np.array(coords)
        
        # Plotear puntos
        ax.scatter(coords[:, 0], coords[:, 1],
                  alpha=0.5,
                  s=30,  # Tamaño de los puntos
                  c=[colors[region_idx]],
                  label=f'Region {region}')
    
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.set_title('Distribución en el Espacio de Coordenadas Reales')
    ax.set_xlim(-1, 65)
    ax.set_ylim(-1, 65)
    ax.grid(True, alpha=0.3)
    
    plt.savefig('coordinate_space_real.png', dpi=1200, bbox_inches='tight', pad_inches=0.2)
    plt.close()

if __name__ == "__main__":
    plot_coordinates('Tesis/coordenadas.csv')
    print("Visualization saved as 'coordinate_space_real.png'")
