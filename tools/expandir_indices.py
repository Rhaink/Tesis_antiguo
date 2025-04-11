#!/usr/bin/env python3
"""
Programa para expandir un dataset de índices de imágenes,
cargando índices existentes y añadiendo nuevos índices únicos
hasta alcanzar un tamaño objetivo, considerando el balance de clases.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

class ExpansorIndices:
    """
    Clase para cargar índices existentes y expandirlos con nuevos índices únicos.
    """

    def __init__(self, archivos_existentes: List[str], semilla: Optional[int] = None):
        """
        Inicializa el expansor de índices.

        Args:
            archivos_existentes: Lista de rutas a los archivos CSV con índices existentes.
            semilla: Semilla opcional para la generación aleatoria de nuevos índices.
        """
        # Configurar semilla para reproducibilidad
        if semilla is not None:
            np.random.seed(semilla)

        # Obtener directorio base (asume que el script está en una carpeta y los datos en otra)
        # Ajusta esta lógica si tu estructura de directorios es diferente
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent # Sube un nivel desde donde está el script
        self.dataset_dir = self.base_dir / "COVID-19_Radiography_Dataset" # Asume que esta carpeta existe al mismo nivel que la carpeta del script

        # Directorios de categorías (igual que en tu script original)
        self.categorias = {
            1: self.dataset_dir / "COVID/images",
            2: self.dataset_dir / "Normal/images",
            3: self.dataset_dir / "Viral Pneumonia/images"
        }
        self.prefijos = {
            1: "COVID-",
            2: "Normal-",
            3: "Viral Pneumonia-"
        }

        self.archivos_existentes = [self.base_dir / f for f in archivos_existentes]
        self.df_final = pd.DataFrame(columns=['indice', 'categoria', 'id_imagen']) # DataFrame para el resultado final

        # Cargar y validar los índices existentes
        self._cargar_y_validar_existentes()

    def _cargar_y_validar_existentes(self):
        """
        Carga los índices de los archivos CSV existentes, los combina y elimina duplicados.
        """
        lista_df = []
        print("Cargando índices existentes...")
        for ruta_archivo in self.archivos_existentes:
            if not ruta_archivo.exists():
                print(f"Advertencia: El archivo {ruta_archivo} no existe. Se omitirá.")
                continue
            try:
                # Carga asumiendo sin encabezado y columnas: indice, categoria, id_imagen
                df_temp = pd.read_csv(ruta_archivo, header=None, names=['indice_orig', 'categoria', 'id_imagen'])
                # Validar tipos de datos
                df_temp['categoria'] = df_temp['categoria'].astype(int)
                df_temp['id_imagen'] = df_temp['id_imagen'].astype(int)
                lista_df.append(df_temp[['categoria', 'id_imagen']]) # Solo necesitamos estas para evitar duplicados
                print(f"  - Cargado {ruta_archivo.name} ({len(df_temp)} filas)")
            except Exception as e:
                print(f"Error al cargar o procesar el archivo {ruta_archivo}: {e}")

        if not lista_df:
            print("No se cargaron índices existentes.")
            self.df_final = pd.DataFrame(columns=['indice', 'categoria', 'id_imagen'])
            return

        # Combinar todos los dataframes cargados
        df_combinado = pd.concat(lista_df, ignore_index=True)

        # Eliminar duplicados basados en 'categoria' y 'id_imagen'
        n_antes = len(df_combinado)
        df_combinado.drop_duplicates(subset=['categoria', 'id_imagen'], keep='first', inplace=True)
        n_despues = len(df_combinado)

        if n_antes > n_despues:
            print(f"Se eliminaron {n_antes - n_despues} duplicados de los archivos existentes.")

        print(f"Total de índices únicos existentes cargados: {n_despues}")

        # Reasignar índices secuenciales (temporalmente, se hará al final)
        df_combinado.reset_index(drop=True, inplace=True)
        df_combinado.rename(columns={'index': 'indice'}, inplace=True) # Renombrar si es necesario

        # Guardamos este como el inicio de nuestro df_final
        self.df_final = df_combinado[['categoria', 'id_imagen']].copy()


    def contar_imagenes_disponibles(self) -> Dict[int, int]:
        """
        Cuenta la cantidad total de imágenes disponibles en cada categoría en el disco.

        Returns:
            Dict[int, int]: Diccionario con la cantidad de imágenes por categoría.
        """
        conteo = {}
        print("\nContando imágenes disponibles en disco...")
        for categoria, directorio in self.categorias.items():
            if directorio.exists() and directorio.is_dir():
                count = len(list(directorio.glob("*.png")))
                conteo[categoria] = count
                print(f"  - Categoría {categoria}: {count} imágenes encontradas.")
            else:
                print(f"  - Advertencia: Directorio no encontrado para categoría {categoria}: {directorio}")
                conteo[categoria] = 0
        return conteo

    def obtener_ids_disponibles_categoria(self, categoria: int) -> Set[int]:
        """
        Obtiene todos los IDs de imagen numéricos disponibles para una categoría desde el disco.

        Args:
            categoria: Número de la categoría (1, 2, 3).

        Returns:
            Set[int]: Conjunto de IDs de imagen disponibles.
        """
        directorio = self.categorias.get(categoria)
        if not directorio or not directorio.exists():
            return set()

        ids = set()
        prefijo = self.prefijos.get(categoria, "")
        for archivo in directorio.glob("*.png"):
            nombre = archivo.stem
            if nombre.startswith(prefijo):
                try:
                    id_img = int(nombre[len(prefijo):])
                    ids.add(id_img)
                except ValueError:
                    continue # Ignorar archivos que no sigan el patrón esperado
        return ids

    def expandir_a_objetivo(self, target_total: int, target_por_categoria: Dict[int, int]) -> bool:
        """
        Expande el conjunto de índices actual hasta alcanzar el tamaño total objetivo,
        añadiendo nuevos índices aleatorios y únicos según las cantidades deseadas por categoría.

        Args:
            target_total: Número total de índices deseado en el archivo final.
            target_por_categoria: Diccionario con el número final deseado de índices por categoría.

        Returns:
            bool: True si la expansión fue exitosa, False en caso contrario.
        """
        n_actual = len(self.df_final)
        print(f"\nÍndices únicos actuales: {n_actual}")
        print(f"Tamaño objetivo total: {target_total}")

        # Verificar que el target total coincida con la suma de targets por categoría
        if sum(target_por_categoria.values()) != target_total:
            print(f"Error: La suma de los objetivos por categoría ({sum(target_por_categoria.values())}) "
                  f"no coincide con el objetivo total ({target_total}).")
            return False

        # Verificar si ya tenemos suficientes o demasiados índices
        if n_actual >= target_total:
            print("El número de índices existentes ya cumple o supera el objetivo.")
            # Podríamos truncar si hay demasiados, pero por ahora solo informamos.
            if n_actual > target_total:
                 print(f"Advertencia: Hay {n_actual} índices, que es más que el objetivo {target_total}. Se mantendrán los existentes.")
            # Asegurarse de que los índices estén bien numerados y salir
            self.df_final.reset_index(drop=True, inplace=True)
            self.df_final.insert(0, 'indice', self.df_final.index)
            return True

        print("\nCalculando cuántos índices nuevos se necesitan por categoría...")

        nuevos_indices_a_generar: List[Tuple[int, int]] = [] # Lista de (categoria, id_imagen)

        # Calcular cuántos índices ya tenemos por categoría
        conteo_actual_por_categoria = self.df_final['categoria'].value_counts().to_dict()

        # Obtener conjunto de IDs ya usados para chequeo rápido
        ids_usados_por_categoria: Dict[int, Set[int]] = {cat: set() for cat in self.categorias.keys()}
        for index, row in self.df_final.iterrows():
            ids_usados_por_categoria.setdefault(row['categoria'], set()).add(row['id_imagen'])

        # Determinar cuántos nuevos índices necesitamos por categoría y si es posible
        for categoria in sorted(target_por_categoria.keys()):
            target_cat = target_por_categoria[categoria]
            actual_cat = conteo_actual_por_categoria.get(categoria, 0)
            necesarios_cat = target_cat - actual_cat

            print(f"Categoría {categoria}:")
            print(f"  - Objetivo final: {target_cat}")
            print(f"  - Existentes: {actual_cat}")

            if necesarios_cat < 0:
                print(f"  - Advertencia: Ya existen más índices ({actual_cat}) que el objetivo ({target_cat}). No se añadirán ni quitarán.")
                necesarios_cat = 0 # No necesitamos añadir más
                # Podríamos implementar lógica para quitar si fuera necesario, pero es más complejo.
                continue
            elif necesarios_cat == 0:
                print("  - Ya se cumple el objetivo para esta categoría.")
                continue

            print(f"  - Necesarios nuevos: {necesarios_cat}")

            # Obtener todos los IDs disponibles en disco para esta categoría
            ids_disponibles_disco = self.obtener_ids_disponibles_categoria(categoria)
            if not ids_disponibles_disco:
                print(f"  - Error: No se encontraron imágenes en disco para la categoría {categoria}.")
                return False

            # Obtener los IDs que ya están en nuestro set actual
            ids_ya_usados = ids_usados_por_categoria.get(categoria, set())

            # Calcular los IDs que podemos seleccionar (disponibles en disco y no usados)
            ids_seleccionables = list(ids_disponibles_disco - ids_ya_usados)

            print(f"  - IDs disponibles en disco: {len(ids_disponibles_disco)}")
            print(f"  - IDs ya usados en el dataset: {len(ids_ya_usados)}")
            print(f"  - IDs disponibles para seleccionar: {len(ids_seleccionables)}")

            if len(ids_seleccionables) < necesarios_cat:
                print(f"  - Error: No hay suficientes imágenes únicas disponibles ({len(ids_seleccionables)}) "
                      f"para añadir las {necesarios_cat} requeridas para la categoría {categoria}.")
                return False

            # Seleccionar aleatoriamente los IDs necesarios sin reemplazo
            ids_nuevos_seleccionados = np.random.choice(ids_seleccionables, size=necesarios_cat, replace=False)
            print(f"  - Seleccionados {len(ids_nuevos_seleccionados)} IDs nuevos para categoría {categoria}.")

            # Añadir los nuevos índices a la lista
            for id_img in ids_nuevos_seleccionados:
                nuevos_indices_a_generar.append((categoria, id_img))

        # Crear DataFrame con los nuevos índices
        if nuevos_indices_a_generar:
            df_nuevos = pd.DataFrame(nuevos_indices_a_generar, columns=['categoria', 'id_imagen'])
            print(f"\nGenerados {len(df_nuevos)} nuevos índices en total.")

            # Combinar los índices existentes con los nuevos
            self.df_final = pd.concat([self.df_final[['categoria', 'id_imagen']], df_nuevos], ignore_index=True)
        else:
             print("\nNo se generaron nuevos índices (quizás el objetivo ya se cumplía).")


        # Verificar tamaño final
        n_final = len(self.df_final)
        if n_final != target_total:
             print(f"Advertencia: El tamaño final ({n_final}) no coincide exactamente con el objetivo ({target_total}). "
                   "Esto puede ocurrir si los objetivos por categoría no eran alcanzables o si ya se excedía el límite.")
             # Podría ser necesario ajustar aquí si se quiere forzar el tamaño exacto, por ejemplo, truncando.

        # Asegurar el orden final y reasignar el índice secuencial
        self.df_final.sort_values(by=['categoria', 'id_imagen'], inplace=True) # Opcional: ordenar por categoría e ID
        self.df_final.reset_index(drop=True, inplace=True)
        self.df_final.insert(0, 'indice', self.df_final.index) # Añadir columna 'indice' final

        print(f"\nDataset final contiene {len(self.df_final)} índices únicos.")
        print("Distribución final por categoría:")
        print(self.df_final['categoria'].value_counts().sort_index())

        return True

    def guardar_indices(self, nombre_archivo_salida: str) -> bool:
        """
        Guarda el DataFrame final de índices en un archivo CSV.

        Args:
            nombre_archivo_salida: Nombre del archivo CSV de salida.

        Returns:
            bool: True si se guardó correctamente, False en caso contrario.
        """
        if self.df_final.empty:
            print("Error: No hay índices para guardar.")
            return False

        ruta_salida = self.base_dir / nombre_archivo_salida
        try:
            self.df_final.to_csv(ruta_salida, header=False, index=False)
            print(f"\nÍndices expandidos guardados correctamente en: {ruta_salida}")
            return True
        except Exception as e:
            print(f"Error al guardar el archivo de índices expandidos: {e}")
            return False

# --- Función Principal ---
def main():
    """Función principal para ejecutar el proceso de expansión."""

    # --- Configuración ---
    archivos_indices_existentes = ["indices/indices.csv", "indices/indices_1.csv", "indices/indices_maestro.csv", "indices/indices_nuevas_400.csv"] # Nombres relativos a base_dir
    nombre_archivo_salida = "indices_expandido_1300.csv" # Nombre del nuevo archivo
    target_total_indices = 1300 # Tamaño final deseado
    semilla_aleatoria = 42 # Para reproducibilidad (None para aleatorio real)

    # --- Inicializar Expansor ---
    expansor = ExpansorIndices(archivos_indices_existentes, semilla=semilla_aleatoria)

    # --- Verificar Imágenes Disponibles en Disco ---
    conteo_disponible = expansor.contar_imagenes_disponibles()

    # --- Definir Objetivo por Categoría ---
    # Puedes ajustar esto según tus necesidades. Debe sumar `target_total_indices`.
    # Ejemplo: Intentar mantener un balance o alcanzar uno específico.
    # Aquí un ejemplo para 800 índices, tratando de balancear un poco más:
    target_por_categoria = {
        1: 325, # COVID
        2: 650, # Normal
        3: 325  # Viral Pneumonia
    }
    # Comprobación importante:
    if sum(target_por_categoria.values()) != target_total_indices:
        print(f"Error de configuración: La suma de target_por_categoria ({sum(target_por_categoria.values())}) "
              f"no es igual a target_total_indices ({target_total_indices}). Ajusta los valores.")
        return

    print("\nObjetivo de distribución final por categoría:")
    for cat, count in target_por_categoria.items():
        print(f"  - Categoría {cat}: {count} índices")

    # --- Ejecutar Expansión ---
    exito_expansion = expansor.expandir_a_objetivo(target_total_indices, target_por_categoria)

    # --- Guardar Resultados ---
    if exito_expansion:
        expansor.guardar_indices(nombre_archivo_salida)
    else:
        print("\nLa expansión no se completó debido a errores previos.")

if __name__ == "__main__":
    main()