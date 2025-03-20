import pandas as pd
import json
import numpy as np

def calcular_distancia(x1, y1, x2, y2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def comparar_coordenadas():
    # Leer el archivo CSV con las coordenadas verdaderas
    df_verdaderos = pd.read_csv('Tesis/coordenadas_1.csv', header=None)
    # El nombre de la imagen está en la última columna
    df_verdaderos.columns = [*range(df_verdaderos.shape[1]-1), 'nombre_imagen']
    
    # Leer el archivo JSON con las predicciones
    with open('Tesis/resultados/prediccion/lote/json/results.json', 'r') as f:
        predicciones = json.load(f)
    
    resultados = []
    no_encontrados = []
    
    # Iterar sobre cada par de coordenadas verdaderas
    for index, row in df_verdaderos.iterrows():
        # Obtener el nombre de la imagen y agregarle la extensión .png si no la tiene
        nombre_imagen = row['nombre_imagen']
        if not nombre_imagen.endswith('.png'):
            nombre_imagen = f"{nombre_imagen}.png"
        
        # Obtener coordenadas verdaderas (están en las columnas 1,2 para coord1 y 3,4 para coord2)
        coord1_verdadero = (row[1], row[2])
        coord2_verdadero = (row[3], row[4])
        
        # Verificar si la imagen está en las predicciones
        if nombre_imagen not in predicciones:
            no_encontrados.append(nombre_imagen)
            continue
        
        # Obtener coordenadas predichas del JSON
        pred_imagen = predicciones[nombre_imagen]
        
        # Obtener coordenadas del modelo MSE
        coord1_predicho = pred_imagen["coord1_mse"]["coordinate"]
        coord2_predicho = pred_imagen["coord2_mse"]["coordinate"]
        
        # Calcular distancias
        distancia_coord1 = calcular_distancia(
            coord1_verdadero[0], coord1_verdadero[1],
            coord1_predicho[0], coord1_predicho[1]
        )
        
        distancia_coord2 = calcular_distancia(
            coord2_verdadero[0], coord2_verdadero[1],
            coord2_predicho[0], coord2_predicho[1]
        )
        
        resultados.append({
            'imagen': nombre_imagen,
            'distancia_coord1': distancia_coord1,
            'distancia_coord2': distancia_coord2,
            'error_mse_coord1': pred_imagen["coord1_mse"]["error"],
            'error_euclidean_coord1': pred_imagen["coord1_euclidean"]["error"],
            'error_mse_coord2': pred_imagen["coord2_mse"]["error"],
            'error_euclidean_coord2': pred_imagen["coord2_euclidean"]["error"]
        })
    
    # Verificar si tenemos resultados
    if not resultados:
        print("Error: No se encontraron coincidencias entre el CSV y el JSON")
        return None
        
    # Mostrar resumen de imágenes no encontradas
    if no_encontrados:
        print("\nImágenes no encontradas en el JSON:")
        for img in no_encontrados:
            print(f"- {img}")
        print(f"Total de imágenes no encontradas: {len(no_encontrados)}")
    
    # Convertir resultados a DataFrame para mejor visualización
    df_resultados = pd.DataFrame(resultados)
    
    # Guardar resultados en CSV
    df_resultados.to_csv('resultados_comparacion.csv', index=False)
    
    # Calcular y mostrar estadísticas
    print(f"\nTotal de imágenes procesadas exitosamente: {len(resultados)}")
    print("\nEstadísticas de las distancias (en píxeles):")
    print("\nPara Coordenada 1:")
    print(f"Media: {df_resultados['distancia_coord1'].mean():.2f}")
    print(f"Mediana: {df_resultados['distancia_coord1'].median():.2f}")
    print(f"Desviación estándar: {df_resultados['distancia_coord1'].std():.2f}")
    
    print("\nPara Coordenada 2:")
    print(f"Media: {df_resultados['distancia_coord2'].mean():.2f}")
    print(f"Mediana: {df_resultados['distancia_coord2'].median():.2f}")
    print(f"Desviación estándar: {df_resultados['distancia_coord2'].std():.2f}")
    
    return df_resultados

if __name__ == "__main__":
    resultados = comparar_coordenadas()
    if resultados is not None:
        print(f"\nLos resultados se han guardado en 'resultados_comparacion.csv'")
