#!/usr/bin/env python3
"""
Programa para generar archivos indices.csv con selección aleatoria y manual de imágenes.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class GeneradorIndices:
    """Clase para generar y gestionar índices de imágenes."""
    
    def __init__(self):
        """Inicializa el generador de índices."""
        # Obtener directorio base
        self.base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.dataset_dir = self.base_dir / "COVID-19_Radiography_Dataset"
        
        # Directorios de categorías
        self.categorias = {
            1: self.dataset_dir / "COVID/images",
            2: self.dataset_dir / "Normal/images",
            3: self.dataset_dir / "Viral Pneumonia/images"
        }
        
        # Lista para almacenar índices
        self.indices: List[Tuple[int, int, int]] = []
        
    def contar_imagenes(self) -> Dict[int, int]:
        """
        Cuenta la cantidad de imágenes disponibles en cada categoría.
        
        Returns:
            Dict[int, int]: Diccionario con la cantidad de imágenes por categoría
        """
        conteo = {}
        for categoria, directorio in self.categorias.items():
            if directorio.exists():
                conteo[categoria] = len(list(directorio.glob("*.png")))
            else:
                conteo[categoria] = 0
        return conteo
    
    def obtener_ids_disponibles(self, categoria: int) -> List[int]:
        """
        Obtiene los IDs disponibles para una categoría.
        
        Args:
            categoria: Número de categoría (1, 2, 3)
            
        Returns:
            List[int]: Lista de IDs disponibles
        """
        prefijos = {
            1: "COVID-",
            2: "Normal-",
            3: "Viral Pneumonia-"
        }
        
        directorio = self.categorias[categoria]
        if not directorio.exists():
            return []
        
        ids = []
        prefijo = prefijos[categoria]
        for archivo in directorio.glob("*.png"):
            # Extraer ID del nombre del archivo
            nombre = archivo.stem
            if nombre.startswith(prefijo):
                try:
                    id_img = int(nombre[len(prefijo):])
                    ids.append(id_img)
                except ValueError:
                    continue
        return sorted(ids)
    
    def generar_indices_aleatorios(self, cantidades: Dict[int, int], semilla: int = None) -> bool:
        """
        Genera índices aleatorios para cada categoría.
        
        Args:
            cantidades: Diccionario con la cantidad deseada de imágenes por categoría
            semilla: Semilla para reproducibilidad (opcional)
            
        Returns:
            bool: True si se generaron los índices correctamente
        """
        if semilla is not None:
            np.random.seed(semilla)
            
        self.indices = []
        indice_actual = 0
        
        for categoria in sorted(cantidades.keys()):
            cantidad = cantidades[categoria]
            ids_disponibles = self.obtener_ids_disponibles(categoria)
            
            if cantidad > len(ids_disponibles):
                print(f"Error: Se solicitaron {cantidad} imágenes para categoría {categoria}, "
                      f"pero solo hay {len(ids_disponibles)} disponibles")
                return False
            
            # Seleccionar IDs aleatorios
            ids_seleccionados = np.random.choice(ids_disponibles, size=cantidad, replace=False)
            
            # Agregar a la lista de índices
            for id_img in ids_seleccionados:
                self.indices.append((indice_actual, categoria, id_img))
                indice_actual += 1
                
        return True
    
    def agregar_indice_manual(self, indice: int, categoria: int, id_img: int) -> bool:
        """
        Agrega un índice manualmente.
        
        Args:
            categoria: Número de categoría (1, 2, 3)
            id_img: ID de la imagen
            
        Returns:
            bool: True si se agregó correctamente
        """
        # Verificar que la categoría es válida
        if categoria not in self.categorias:
            print(f"Error: Categoría {categoria} no válida")
            return False
        
        # Verificar que el ID existe
        ids_disponibles = self.obtener_ids_disponibles(categoria)
        if id_img not in ids_disponibles:
            print(f"Error: ID {id_img} no existe en la categoría {categoria}")
            return False
        
        # Verificar que no esté duplicado
        for _, cat, id_existente in self.indices:
            if categoria == cat and id_img == id_existente:
                print(f"Error: El índice ya existe en la lista")
                return False
        
        # Agregar nuevo índice con el índice especificado
        self.indices.append((indice, categoria, id_img))
        return True
    
    def agregar_lista_indices(self, lista_indices: str) -> bool:
        """
        Agrega una lista de índices en formato CSV.
        
        Args:
            lista_indices: String con formato 'indice,categoria,id' por línea
            
        Returns:
            bool: True si se agregaron correctamente todos los índices
        """
        lineas = lista_indices.strip().split('\n')
        for linea in lineas:
            try:
                indice, categoria, id_img = map(int, linea.strip().split(','))
                if not self.agregar_indice_manual(indice, categoria, id_img):
                    return False
            except ValueError:
                print(f"Error: Formato inválido en línea: {linea}")
                return False
        return True

    def guardar_indices(self, nombre_archivo: str = "indices.csv") -> bool:
        """
        Guarda los índices en un archivo CSV.
        
        Args:
            nombre_archivo: Nombre del archivo de salida
            
        Returns:
            bool: True si se guardó correctamente
        """
        if not self.indices:
            print("Error: No hay índices para guardar")
            return False
        
        ruta_archivo = self.base_dir / nombre_archivo
        
        try:
            # Convertir a DataFrame y guardar
            df = pd.DataFrame(self.indices)
            df.to_csv(ruta_archivo, header=False, index=False)
            print(f"Índices guardados en: {ruta_archivo}")
            return True
        except Exception as e:
            print(f"Error al guardar archivo: {e}")
            return False

def main():
    """Función principal del programa."""
    generador = GeneradorIndices()
    
    while True:
        print("\n=== Generador de Índices ===")
        print("1. Ver cantidad de imágenes disponibles")
        print("2. Generar índices aleatorios")
        print("3. Agregar índice manual")
        print("4. Agregar lista de índices")
        print("5. Guardar índices")
        print("6. Salir")
        
        opcion = input("\nSeleccione una opción: ")
        
        if opcion == "1":
            conteo = generador.contar_imagenes()
            print("\nImágenes disponibles por categoría:")
            for categoria, cantidad in conteo.items():
                print(f"Categoría {categoria}: {cantidad} imágenes")
                ids = generador.obtener_ids_disponibles(categoria)
                print(f"  Rango de IDs: {min(ids)} - {max(ids)}")
        
        elif opcion == "2":
            cantidades = {}
            for categoria in [1, 2, 3]:
                while True:
                    try:
                        cant = int(input(f"Cantidad de imágenes para categoría {categoria}: "))
                        if cant >= 0:
                            cantidades[categoria] = cant
                            break
                        print("Por favor ingrese un número positivo")
                    except ValueError:
                        print("Por favor ingrese un número válido")
            
            semilla = input("Ingrese semilla (Enter para aleatorio): ")
            semilla = int(semilla) if semilla.strip() else None
            
            if generador.generar_indices_aleatorios(cantidades, semilla):
                print(f"Se generaron {len(generador.indices)} índices")
        
        elif opcion == "3":
            try:
                indice = int(input("Índice: "))
                categoria = int(input("Categoría (1, 2, 3): "))
                id_img = int(input("ID de imagen: "))
                if generador.agregar_indice_manual(indice, categoria, id_img):
                    print("Índice agregado correctamente")
            except ValueError:
                print("Por favor ingrese números válidos")
        
        elif opcion == "4":
            print("Ingrese la lista de índices (formato: indice,categoria,id)")
            print("Presione Ctrl+D (Linux/Mac) o Ctrl+Z (Windows) + Enter cuando termine")
            try:
                lista_indices = ""
                while True:
                    try:
                        linea = input()
                        lista_indices += linea + "\n"
                    except EOFError:
                        break
                
                if generador.agregar_lista_indices(lista_indices):
                    print("Lista de índices agregada correctamente")
                
            except KeyboardInterrupt:
                print("\nEntrada cancelada")
        
        elif opcion == "5":
            nombre = input("Nombre del archivo (Enter para 'indices.csv'): ")
            nombre = nombre.strip() if nombre.strip() else "indices.csv"
            generador.guardar_indices(nombre)
        
        elif opcion == "6":
            break
        
        else:
            print("Opción no válida")

if __name__ == "__main__":
    main()
