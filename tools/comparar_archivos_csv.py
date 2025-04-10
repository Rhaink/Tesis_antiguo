import os
import pandas as pd

def cargar_archivo_csv(ruta_archivo):
    """
    Carga un archivo CSV sin encabezados con el formato esperado.
    Devuelve un DataFrame con columnas: indice, categoria, id_imagen
    """
    try:
        df = pd.read_csv(ruta_archivo, header=None)
        return df
    except Exception as e:
        print(f"Error al cargar el archivo {ruta_archivo}: {e}")
        return None

def encontrar_coincidencias(archivos_csv):
    """
    Compara múltiples archivos CSV y encuentra coincidencias por categoría e id_imagen.
    Devuelve un diccionario con las coincidencias encontradas.
    """
    if len(archivos_csv) < 2:
        print("Se necesitan al menos dos archivos para realizar la comparación.")
        return {}
    
    # Cargar todos los archivos CSV
    dataframes = []
    nombres_archivos = []
    
    for archivo in archivos_csv:
        df = cargar_archivo_csv(archivo)
        if df is not None:
            dataframes.append(df)
            nombres_archivos.append(os.path.basename(archivo))
    
    if len(dataframes) < 2:
        print("No se pudieron cargar suficientes archivos para comparar.")
        return {}
    
    # Crear un diccionario para almacenar coincidencias
    coincidencias = {}
    
    # Comparar cada par de DataFrames
    for i in range(len(dataframes)):
        for j in range(i + 1, len(dataframes)):
            df1 = dataframes[i]
            df2 = dataframes[j]
            nombre1 = nombres_archivos[i]
            nombre2 = nombres_archivos[j]
            
            # Encontrar coincidencias (tanto en categoría como en id_imagen)
            # Las columnas son 1 (categoría) y 2 (id_imagen)
            comparacion = pd.merge(
                df1, df2, 
                on=[1, 2],
                suffixes=('_1', '_2')
            )
            
            if not comparacion.empty:
                clave = f"{nombre1} vs {nombre2}"
                coincidencias[clave] = comparacion[[1, 2]].values.tolist()
                print(f"Encontradas {len(comparacion)} coincidencias entre {nombre1} y {nombre2}")
    
    return coincidencias

def generar_reporte(coincidencias, ruta_salida=None):
    """
    Genera un reporte detallado de las coincidencias encontradas.
    Si se proporciona una ruta de salida, guarda el reporte en un archivo CSV.
    """
    if not coincidencias:
        print("No se encontraron coincidencias entre los archivos.")
        return
    
    # Crear un DataFrame para el reporte
    filas = []
    for comparacion, coincidencias_lista in coincidencias.items():
        for cat, id_img in coincidencias_lista:
            filas.append({
                'Comparacion': comparacion,
                'Categoria': cat,
                'ID_Imagen': id_img
            })
    
    reporte = pd.DataFrame(filas)
    
    # Mostrar resumen
    print("\nResumen de coincidencias:")
    for comparacion, coincidencias_lista in coincidencias.items():
        print(f"- {comparacion}: {len(coincidencias_lista)} coincidencias")
    
    # Guardar en archivo si se especifica ruta
    if ruta_salida:
        reporte.to_csv(ruta_salida, index=False)
        print(f"\nReporte guardado en: {ruta_salida}")
    
    return reporte

# Ejemplo de uso directo similar al código original
def main():
    # Definir rutas base
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Definir los archivos a comparar
    indices_dir = os.path.join(base_dir, "indices")
    
    # Buscar todos los archivos CSV en la carpeta de índices
    archivos_csv = []
    for archivo in os.listdir(indices_dir):
        if archivo.endswith('.csv'):
            archivos_csv.append(os.path.join(indices_dir, archivo))
    
    if len(archivos_csv) < 2:
        print("Se necesitan al menos dos archivos CSV en la carpeta de índices para hacer la comparación.")
        return
    
    print(f"Encontrados {len(archivos_csv)} archivos CSV: {', '.join(os.path.basename(f) for f in archivos_csv)}")
    
    # Encontrar coincidencias
    coincidencias = encontrar_coincidencias(archivos_csv)
    
    # Generar y guardar reporte
    reporte_salida = os.path.join(base_dir, "indices/coincidencias.csv")
    generar_reporte(coincidencias, reporte_salida)

if __name__ == "__main__":
    main()