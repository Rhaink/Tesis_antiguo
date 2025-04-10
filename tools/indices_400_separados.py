import pandas as pd
import os

# --- Configuración ---
base_dir = "/home/donrobot/projects/Tesis" # Ajusta si es necesario
archivo_indices_orig1 = os.path.join(base_dir, "indices/indices.csv")
archivo_indices_orig2 = os.path.join(base_dir, "indices/indices_1.csv")
archivo_maestro = os.path.join(base_dir, "indices/indices_maestro.csv") # Asume que este es tu archivo expandido
archivo_salida_nuevos = os.path.join(base_dir, "indices/indices_nuevas_400.csv") # Archivo que contendrá solo los nuevos
# --- Fin Configuración ---

print("Cargando archivos de índices...")
try:
    # Cargar los índices originales (solo necesitamos categoría e ID para comparar)
    df_orig1 = pd.read_csv(archivo_indices_orig1, header=None, usecols=[1, 2], names=['categoria', 'id_imagen'])
    df_orig2 = pd.read_csv(archivo_indices_orig2, header=None, usecols=[1, 2], names=['categoria', 'id_imagen'])
    
    # Cargar el archivo maestro completo
    df_maestro = pd.read_csv(archivo_maestro, header=None, names=['indice_maestro', 'categoria', 'id_imagen'])
    
    print(f"Cargados {len(df_orig1)} índices de {os.path.basename(archivo_indices_orig1)}")
    print(f"Cargados {len(df_orig2)} índices de {os.path.basename(archivo_indices_orig2)}")
    print(f"Cargados {len(df_maestro)} índices de {os.path.basename(archivo_maestro)}")

    # Combinar los índices originales
    df_originales_combinados = pd.concat([df_orig1, df_orig2], ignore_index=True)
    
    # Crear una clave única (categoria, id) para facilitar la comparación
    df_originales_combinados['clave'] = df_originales_combinados.apply(lambda row: (row['categoria'], row['id_imagen']), axis=1)
    df_maestro['clave'] = df_maestro.apply(lambda row: (row['categoria'], row['id_imagen']), axis=1)

    # Identificar las claves únicas de los originales
    claves_originales = set(df_originales_combinados['clave'])

    # Filtrar el dataframe maestro para quedarse solo con las filas cuya clave NO está en las originales
    df_nuevos = df_maestro[~df_maestro['clave'].isin(claves_originales)].copy()

    print(f"Se identificaron {len(df_nuevos)} índices nuevos para etiquetar.")

    if len(df_nuevos) == 0:
        print("No se encontraron índices nuevos. ¿Ya están todos etiquetados o los archivos son incorrectos?")
    elif len(df_nuevos) != 400:
         print(f"Advertencia: Se encontraron {len(df_nuevos)} índices nuevos, pero se esperaban 400. Revisa los archivos de entrada.")

    if not df_nuevos.empty:
        # Re-indexar la primera columna de 0 a N-1 para el nuevo archivo
        df_nuevos['indice_nuevo'] = range(len(df_nuevos))
        
        # Seleccionar y ordenar las columnas para el archivo de salida
        df_salida = df_nuevos[['indice_nuevo', 'categoria', 'id_imagen']]
        
        # Guardar el nuevo archivo de índices
        df_salida.to_csv(archivo_salida_nuevos, header=False, index=False)
        print(f"Archivo con los índices nuevos guardado en: {archivo_salida_nuevos}")

except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo {e.filename}")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")