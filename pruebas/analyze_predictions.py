#!/usr/bin/env python3
"""
Script para analizar y comparar las predicciones con las coordenadas reales.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def load_indices_mapping() -> Dict[int, int]:
    """
    Carga el mapeo de número de imagen a tipo desde indices.csv
    
    Returns:
        Diccionario que mapea número de imagen a su tipo
    """
    mapping = {}
    indices_path = Path(__file__).parent / "indices.csv"
    with open(indices_path, 'r') as f:
        for line in f:
            _, tipo, num = line.strip().split(',')
            mapping[int(num)] = int(tipo)
    return mapping

def load_ground_truth(csv_path: str) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """
    Carga las coordenadas reales del archivo CSV.
    
    Args:
        csv_path: Ruta al archivo de coordenadas
        
    Returns:
        Diccionario con las coordenadas por imagen y punto
    """
    # Leer CSV sin encabezados
    df = pd.read_csv(csv_path, header=None)
    
    # Mapear tipos de imágenes
    type_map = {
        "Normal": 2,
        "COVID": 1,
        "Viral Pneumonia": 3
    }
    
    ground_truth = {}
    
    for _, row in df.iterrows():
        # El último valor es el ID de la imagen
        image_id = str(row.iloc[-1])
        
        # Determinar tipo y número de imagen
        if image_id.startswith('Normal-'):
            tipo = 2
            num = int(image_id.split('-')[1])
        elif image_id.startswith('COVID-'):
            tipo = 1
            num = int(image_id.split('-')[1])
        else:  # Es un número directo
            num = int(image_id)
            # Usar el mapeo de índices para determinar el tipo
            indices_mapping = load_indices_mapping()
            tipo = indices_mapping.get(num)
            if tipo is None:
                print(f"Advertencia: No se encontró tipo para imagen {num}")
                continue
        
        image_key = f"tipo{tipo}_img{num}"
        
        # Extraer coordenadas de los primeros dos puntos
        # Punto 1: columnas 0,1; Punto 2: columnas 2,3
        ground_truth[image_key] = {
            "coord1": (int(row[0]), int(row[1])),
            "coord2": (int(row[2]), int(row[3]))
        }
    
    return ground_truth

def load_predictions(json_path: str) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """
    Carga las predicciones del archivo JSON.
    
    Args:
        json_path: Ruta al archivo de predicciones
        
    Returns:
        Diccionario con las coordenadas predichas por imagen y punto
    """
    with open(json_path, 'r') as f:
        predictions_data = json.load(f)
    
    predictions = {}
    for image_key, data in predictions_data.items():
        predictions[image_key] = {}
        for coord_name, coord_data in data['predictions'].items():
            predictions[image_key][coord_name] = tuple(coord_data['min_error_coords'])
    
    return predictions

def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """
    Calcula la distancia euclidiana entre dos puntos.
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def analyze_results(ground_truth: Dict, 
                   predictions: Dict,
                   output_dir: str):
    """
    Analiza las diferencias entre predicciones y valores reales.
    
    Args:
        ground_truth: Coordenadas reales
        predictions: Coordenadas predichas
        output_dir: Directorio para guardar resultados
    """
    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Almacenar distancias por punto y tipo
    distances = {
        "coord1": {"total": [], 1: [], 2: [], 3: []},
        "coord2": {"total": [], 1: [], 2: [], 3: []}
    }
    
    # Calcular distancias
    results = {}
    for image_key in ground_truth.keys():
        if image_key not in predictions:
            print(f"Advertencia: No hay predicción para {image_key}")
            continue
            
        tipo = int(image_key.split('_')[0][-1])  # Extraer tipo del image_key
        
        results[image_key] = {}
        for coord_name in ['coord1', 'coord2']:
            if coord_name not in predictions[image_key]:
                print(f"Advertencia: No hay predicción para {coord_name} en {image_key}")
                continue
                
            gt_point = ground_truth[image_key][coord_name]
            pred_point = predictions[image_key][coord_name]
            distance = calculate_distance(gt_point, pred_point)
            
            results[image_key][coord_name] = {
                'ground_truth': gt_point,
                'prediction': pred_point,
                'distance': distance
            }
            
            distances[coord_name]['total'].append(distance)
            distances[coord_name][tipo].append(distance)
    
    # Calcular estadísticas
    stats = {}
    for coord_name in ['coord1', 'coord2']:
        stats[coord_name] = {
            'total': {
                'mean': np.mean(distances[coord_name]['total']),
                'std': np.std(distances[coord_name]['total']),
                'median': np.median(distances[coord_name]['total']),
                'min': np.min(distances[coord_name]['total']),
                'max': np.max(distances[coord_name]['total'])
            }
        }
        
        # Estadísticas por tipo
        for tipo in [1, 2, 3]:
            if distances[coord_name][tipo]:  # Si hay datos para este tipo
                stats[coord_name][f'tipo_{tipo}'] = {
                    'mean': np.mean(distances[coord_name][tipo]),
                    'std': np.std(distances[coord_name][tipo]),
                    'median': np.median(distances[coord_name][tipo]),
                    'min': np.min(distances[coord_name][tipo]),
                    'max': np.max(distances[coord_name][tipo])
                }
    
    # Guardar resultados
    with open(output_path / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    with open(output_path / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Generar visualizaciones
    for coord_name in ['coord1', 'coord2']:
        # Histograma de distancias
        plt.figure(figsize=(10, 6))
        plt.hist(distances[coord_name]['total'], bins=30)
        plt.title(f'Distribución de Distancias - {coord_name}')
        plt.xlabel('Distancia (píxeles)')
        plt.ylabel('Frecuencia')
        plt.savefig(output_path / f'{coord_name}_distances_hist.png')
        plt.close()
        
        # Box plot por tipo
        try:
            plt.figure(figsize=(10, 6))
            # Filtrar tipos que tienen datos
            valid_types = [tipo for tipo in [1, 2, 3] if len(distances[coord_name][tipo]) > 0]
            data = [distances[coord_name][tipo] for tipo in valid_types]
            if data:  # Solo crear el plot si hay datos
                tipo_labels = {1: 'COVID', 2: 'Normal', 3: 'Viral Pneumonia'}
                labels = [tipo_labels[t] for t in valid_types]
                plt.boxplot(data, labels=labels)
                plt.title(f'Distribución de Distancias por Tipo - {coord_name}')
                plt.ylabel('Distancia (píxeles)')
                plt.savefig(output_path / f'{coord_name}_distances_boxplot.png')
            plt.close()
        except Exception as e:
            print(f"Error creando boxplot para {coord_name}: {e}")

def main():
    """Función principal del script."""
    import argparse
    parser = argparse.ArgumentParser(description="Análisis de predicciones vs. ground truth")
    
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=str(Path(__file__).parent.parent / "coordenadas.csv"),
        help="Archivo CSV con coordenadas reales"
    )
    
    parser.add_argument(
        "--predictions",
        type=str,
        default=str(Path(__file__).parent / "batch_results/batch_results.json"),
        help="Archivo JSON con predicciones"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent / "analysis_results"),
        help="Directorio para resultados del análisis"
    )
    
    args = parser.parse_args()
    
    print("Cargando coordenadas reales...")
    ground_truth = load_ground_truth(args.ground_truth)
    
    print("Cargando predicciones...")
    predictions = load_predictions(args.predictions)
    
    print("Analizando resultados...")
    analyze_results(ground_truth, predictions, args.output_dir)
    
    print("\nAnálisis completado.")
    print(f"Resultados guardados en: {args.output_dir}")

if __name__ == "__main__":
    main()
