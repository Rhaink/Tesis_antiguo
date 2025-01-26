import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurar el estilo de las gráficas
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['axes.grid'] = True

def load_coordinates(file_path):
    """Cargar archivo CSV y estructurar las coordenadas."""
    df = pd.read_csv(file_path, header=None)
    # Excluir la primera y última columna (índice e identificador)
    coordinates = df.iloc[:, 1:-1].values
    return coordinates

def calculate_errors(real_coords, pred_coords):
    """Calcular distancia euclidiana entre puntos reales y predichos."""
    # Reshape para tener formato (n_samples, n_points, 2)
    real = real_coords.reshape(-1, 15, 2)
    pred = pred_coords.reshape(-1, 15, 2)
    
    # Calcular distancia euclidiana para cada punto
    euclidean_distances = np.sqrt(np.sum((real - pred)**2, axis=2))
    
    # Calcular estadísticas por punto
    mean_distances = np.mean(euclidean_distances, axis=0)
    std_distances = np.std(euclidean_distances, axis=0)
    max_distances = np.max(euclidean_distances, axis=0)
    
    return mean_distances, std_distances, max_distances, euclidean_distances

def plot_scatter_comparison(real_coords, pred_coords, point_idx, save_path):
    """Crear gráfico de dispersión para un punto específico."""
    plt.figure()
    plt.scatter(real_coords[:, point_idx*2], real_coords[:, point_idx*2+1], 
               alpha=0.5, label='Real')
    plt.scatter(pred_coords[:, point_idx*2], pred_coords[:, point_idx*2+1], 
               alpha=0.5, label='Predicho')
    plt.title(f'Comparación de coordenadas para el punto {point_idx+1}')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.legend()
    plt.savefig(save_path / f'scatter_point_{point_idx+1}.png')
    plt.close()

def plot_error_bars(mean_distances, std_distances, save_path):
    """Crear gráfico de barras de distancia euclidiana media por punto."""
    plt.figure(figsize=(15, 6))
    x = np.arange(15)
    
    plt.bar(x, mean_distances, yerr=std_distances, capsize=5)
    plt.xlabel('Número de punto')
    plt.ylabel('Distancia Euclidiana (píxeles)')
    plt.title('Distancia media entre puntos reales y predichos')
    plt.xticks(x, [f'P{i+1}' for i in range(15)])
    
    # Añadir valores sobre las barras
    for i, v in enumerate(mean_distances):
        plt.text(i, v + std_distances[i], f'{v:.2f}±{std_distances[i]:.2f}', 
                ha='center', va='bottom')
    
    plt.savefig(save_path / 'euclidean_distances.png')
    plt.close()

def plot_distance_histogram(euclidean_distances, save_path):
    """Crear histograma de distancias euclidianas."""
    plt.figure(figsize=(12, 6))
    
    # Aplanar todas las distancias
    all_distances = euclidean_distances.flatten()
    
    plt.hist(all_distances, bins=50, edgecolor='black')
    plt.xlabel('Distancia Euclidiana (píxeles)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de distancias entre puntos reales y predichos')
    
    # Añadir líneas verticales para estadísticas
    plt.axvline(np.mean(all_distances), color='r', linestyle='dashed', 
                label=f'Media: {np.mean(all_distances):.2f}')
    plt.axvline(np.median(all_distances), color='g', linestyle='dashed', 
                label=f'Mediana: {np.median(all_distances):.2f}')
    
    plt.legend()
    plt.savefig(save_path / 'distance_histogram.png')
    plt.close()

def main():
    # Crear directorio para resultados
    save_path = Path('visualization_results')
    save_path.mkdir(exist_ok=True)
    
    # Cargar datos
    real_coords = load_coordinates('Tesis/coordenadas_64x64.csv')
    pred_coords = load_coordinates('coordinate_predictor/predicted_coordinates_test.csv')
    
    # Calcular distancias euclidianas
    mean_distances, std_distances, max_distances, euclidean_distances = calculate_errors(real_coords, pred_coords)
    
    # Generar visualizaciones
    for i in range(15):
        plot_scatter_comparison(real_coords, pred_coords, i, save_path)
    
    plot_error_bars(mean_distances, std_distances, save_path)
    plot_distance_histogram(euclidean_distances, save_path)
    
    # Imprimir estadísticas
    print("\nEstadísticas de distancia por punto (en píxeles):")
    print("\nFormato: Media ± Desviación estándar (Máximo)")
    for i in range(15):
        print(f"Punto {i+1}: {mean_distances[i]:.2f} ± {std_distances[i]:.2f} (max: {max_distances[i]:.2f})")
    
    # Calcular estadísticas globales
    all_distances = euclidean_distances.flatten()
    print("\nEstadísticas globales (en píxeles):")
    print(f"Distancia media: {np.mean(all_distances):.2f}")
    print(f"Desviación estándar: {np.std(all_distances):.2f}")
    print(f"Mediana: {np.median(all_distances):.2f}")
    print(f"Distancia máxima: {np.max(all_distances):.2f}")
    print(f"Distancia mínima: {np.min(all_distances):.2f}")

if __name__ == "__main__":
    main()
