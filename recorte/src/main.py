"""
Script principal para el proceso de recorte de imágenes pulmonares.
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
from coordinates.coordinate_manager import CoordinateManager
from image_processing.image_processor import ImageProcessor


def process_images(
    coord_manager: CoordinateManager,
    image_processor: ImageProcessor,
    coordinates_file: str,
    indices_file: str,
) -> Dict:
    """
    Procesa las imágenes y extrae las regiones de interés.
    Lee coordenadas float del archivo, las convierte a int (redondeo + clip)
    para usarlas como puntos etiquetados en la extracción.
    Todas las coordenadas se manejan internamente en sistema 0-based (0-63).

    Args:
        coord_manager: Gestor de coordenadas (ya debe tener cargadas las search_coordinates)
        image_processor: Procesador de imágenes
        coordinates_file: Ruta al archivo CSV de coordenadas ALINEADAS (con floats)
        indices_file: Ruta al archivo CSV de índices para buscar imágenes originales

    Returns:
        Dict: Resultados del procesamiento con conteo de éxitos y fallos por coordenada.
    """
    # Inicializar diccionario de resultados
    # Las claves deben coincidir con las generadas en read_coordinates (Coord1, Coord2, ...)
    # y las usadas en read_search_coordinates (donde se generan coord_data)
    results = {
        coord_name: {"processed": 0, "failed": 0, "errors": []}
        for coord_name in coord_manager.coord_data.keys()
    }  # Usar las claves de coord_data que tiene la config

    total_images = 0  # Inicializar fuera del try

    try:
        print("\nIniciando procesamiento de imágenes...")
        print(f"Leyendo coordenadas (float) desde: {coordinates_file}")
        # read_coordinates ahora lee floats y los guarda en coord_manager.coordinates
        coord_manager.read_coordinates(coordinates_file)

        total_images = len(coord_manager.coordinates)
        print(f"\nTotal de registros de coordenadas leídos: {total_images}")
        if total_images == 0:
            print("Advertencia: No se leyeron coordenadas. Verifique el archivo.")
            return results

        # --- Bucle Principal: Iterar por cada tipo de coordenada (landmark) ---
        # Usamos coord_manager.coord_data que fue poblado por read_search_coordinates
        # y contiene los límites precalculados (sup, inf, left, right, width, height)
        for coord_name, config in coord_manager.coord_data.items():
            print(f"\n--- Procesando Landmark: {coord_name} ---")

            # Validar si la config tiene los datos esperados (calculados desde search_coords)
            if not all(
                k in config for k in ["sup", "inf", "left", "right", "width", "height"]
            ):
                print(
                    f"Advertencia: Configuración incompleta para {coord_name}. Saltando este landmark."
                )
                results[coord_name]["failed"] = (
                    total_images  # Marcar todas como fallidas para esta coord
                )
                results[coord_name]["errors"].append(
                    "Configuración de límites/dimensiones incompleta."
                )
                continue

            # Calcular centro e intersección (usando los límites de la zona de búsqueda)
            try:
                center_x, center_y = coord_manager.calculate_center(
                    config["sup"], config["inf"], config["left"], config["right"]
                )
                intersection_x, intersection_y = coord_manager.calculate_intersection(
                    config["sup"], config["inf"], config["left"], config["right"]
                )
                print(f"Zona de Búsqueda - Centro: ({center_x}, {center_y})")
                print(
                    f"Zona de Búsqueda - Intersección (esquina sup-izq): ({intersection_x}, {intersection_y})"
                )
                print(
                    f"Zona de Búsqueda - Dimensiones (bbox): {config['width']}x{config['height']}"
                )
                print(
                    f"Usando datos precalculados del template para {coord_name}..."
                )  # Añadido para claridad

            except Exception as e:
                print(
                    f"Error calculando centro/intersección para {coord_name}: {e}. Saltando landmark."
                )
                results[coord_name]["failed"] = total_images
                results[coord_name]["errors"].append(
                    f"Error en cálculo de centro/intersección: {e}"
                )
                continue

            processed_count_coord = 0  # Contador para esta coordenada específica

            # --- Bucle Interno: Iterar por cada imagen ---
            # Usamos coord_manager.coordinates que tiene los puntos (float) para cada imagen
            for index, coords_dict in coord_manager.coordinates.items():
                try:
                    # Obtener coordenadas específicas (float) para esta imagen y este landmark
                    if coord_name not in coords_dict:
                        print(
                            f"Advertencia: No se encontraron coordenadas para {coord_name} en imagen {index}. Saltando."
                        )
                        results[coord_name]["failed"] += 1
                        results[coord_name]["errors"].append(
                            f"Faltan datos de {coord_name} para img {index}"
                        )
                        continue

                    float_x, float_y = coords_dict[coord_name]

                    # --- Conversión Float -> Int para Labeled Point ---
                    # Redondear al entero más cercano y asegurar rango [0, 63]
                    labeled_x_int = np.clip(int(round(float_x)), 0, 63)
                    labeled_y_int = np.clip(int(round(float_y)), 0, 63)

                    # Obtener ruta de la imagen original (NO alineada)
                    image_path = image_processor.get_image_path(index, indices_file)

                    # Cargar y redimensionar imagen original a 64x64
                    # NOTA: Estamos cargando la imagen ORIGINAL y redimensionando.
                    # Si la alineación ALTERÓ la imagen, ¿deberíamos cargar la imagen ALINEADA?
                    # Por ahora, seguimos el código original que carga la original.
                    image = image_processor.load_and_resize_image(
                        image_path, size=(64, 64)
                    )  # Asegurar tamaño

                    # Crear la matriz de la región de búsqueda (usando datos del JSON)
                    search_region_matrix = np.zeros(
                        (64, 64), dtype=np.uint8
                    )  # Usar uint8 es suficiente
                    search_coords = coord_manager.get_search_coordinates(
                        coord_name.lower()
                    )  # Busca por 'coordX'
                    if not search_coords:
                        print(
                            f"Advertencia: No se encontraron coordenadas de búsqueda para {coord_name.lower()} en el JSON. Saltando imagen {index}."
                        )
                        results[coord_name]["failed"] += 1
                        results[coord_name]["errors"].append(
                            f"Faltan search_coords JSON para {coord_name} (img {index})"
                        )
                        continue

                    for y_sr, x_sr in search_coords:
                        # Doble verificación de rango para las coordenadas de búsqueda
                        if 0 <= y_sr <= 63 and 0 <= x_sr <= 63:
                            search_region_matrix[y_sr, x_sr] = 1
                        else:
                            # Esto no debería pasar si extraccion_region_busqueda.py funcionó bien
                            print(
                                f"Advertencia: Coordenada de búsqueda fuera de rango [0, 63] para {coord_name}: ({x_sr}, {y_sr})"
                            )

                    # --- Extracción de la Región ---
                    if (
                        index % 50 == 0 or processed_count_coord < 5
                    ):  # Imprimir más detalles al inicio o cada 50
                        print(f"\nProcesando Imagen {index} para {coord_name}:")
                        print(f"  Path: {Path(image_path).name}")
                        print(
                            f"  Punto etiquetado (float original): ({float_x:.2f}, {float_y:.2f})"
                        )
                        print(
                            f"  Punto etiquetado (int usado): ({labeled_x_int}, {labeled_y_int})"
                        )

                    cropped_image = None  # Reiniciar por si falla la extracción
                    try:
                        # Llamar a extract_region, pasando los puntos ENTEROS
                        # La lógica de qué template usar y cómo cortar está dentro de image_processor/template_processor
                        # NOTA: El parámetro template_size=config["width"] parece redundante si
                        # extract_region carga todo desde template_processor.load_template_data.
                        # Lo mantenemos por ahora, pero revisar su uso real en extract_region.
                        coord_num_int = int(coord_name.replace("Coord", ""))

                        cropped_image = image_processor.extract_region(
                            image=image,
                            search_region=search_region_matrix,  # Pasar la matriz creada
                            labeled_point=(
                                labeled_x_int,
                                labeled_y_int,
                            ),  # Pasar tupla de ints
                            coord_num=coord_num_int,  # Pasar el número de coordenada
                            # template_size=config.get("width") # Pasar ancho de la BBox (revisar si se usa)
                        )

                        if cropped_image is None:
                            # Si extract_region retorna None explícitamente por algún error interno manejado
                            raise ValueError(
                                "extract_region devolvió None (error interno en extracción)."
                            )
                        # else: # Si no es None, la extracción (aparentemente) funcionó
                        #    if index % 50 == 0 or processed_count_coord < 5: print("  Región extraída OK.")

                    except Exception as e:
                        print(
                            f"Error en extracción para imagen {index}, coord {coord_name}: {str(e)}"
                        )
                        results[coord_name]["failed"] += 1
                        results[coord_name]["errors"].append(
                            f"Error en extracción img {index}: {str(e)}"
                        )
                        continue  # Saltar al siguiente índice/imagen

                    # --- Guardar Imagen Recortada ---
                    if cropped_image is not None:
                        try:
                            success = image_processor.save_cropped_image(
                                cropped_image=cropped_image,
                                coord_name=coord_name,  # e.g., "Coord1"
                                index=index,
                            )
                            if success:
                                results[coord_name]["processed"] += 1
                                processed_count_coord += 1
                            else:
                                # El error ya se debería haber impreso dentro de save_cropped_image
                                results[coord_name]["failed"] += 1
                                results[coord_name]["errors"].append(
                                    f"Error al guardar imagen {index} para {coord_name}"
                                )

                        except Exception as e:
                            print(
                                f"Error inesperado al intentar guardar imagen {index} para {coord_name}: {str(e)}"
                            )
                            results[coord_name]["failed"] += 1
                            results[coord_name]["errors"].append(
                                f"Error al guardar img {index}: {str(e)}"
                            )
                            continue  # Saltar al siguiente índice/imagen
                    # else: # Si cropped_image es None, ya se contó como fallo en el bloque de extracción

                    # Actualizar progreso para esta coordenada
                    if processed_count_coord > 0 and processed_count_coord % 50 == 0:
                        print(
                            f"  ... {coord_name}: {processed_count_coord} procesadas exitosamente."
                        )

                # --- Fin del try para una imagen individual ---
                except FileNotFoundError as e:
                    print(
                        f"\nError: No se encontró archivo para imagen {index}: {str(e)}"
                    )
                    results[coord_name]["failed"] += 1
                    results[coord_name]["errors"].append(
                        f"Archivo no encontrado para img {index}: {str(e)}"
                    )
                    # Continuar con la siguiente imagen si es posible
                except ValueError as e:  # Errores de conversión, etc.
                    print(
                        f"\nError de valor procesando imagen {index} para {coord_name}: {str(e)}"
                    )
                    results[coord_name]["failed"] += 1
                    results[coord_name]["errors"].append(
                        f"Error de valor img {index}: {str(e)}"
                    )
                    # Continuar con la siguiente imagen
                except Exception as e:
                    # Capturar cualquier otro error inesperado para una imagen
                    print(
                        f"\nError inesperado procesando imagen {index} para {coord_name}: {str(e)}"
                    )
                    results[coord_name]["failed"] += 1
                    results[coord_name]["errors"].append(
                        f"Error general img {index}: {str(e)}"
                    )
                    # Continuar con la siguiente imagen

            # --- Fin del bucle de imágenes para una coordenada ---
            print(f"\nResultados parciales para {coord_name}:")
            print(f"  Procesadas exitosamente: {results[coord_name]['processed']}")
            print(f"  Fallidas: {results[coord_name]['failed']}")
            if results[coord_name]["errors"]:
                print(f"  Primeros errores registrados para {coord_name}:")
                for i, error in enumerate(
                    results[coord_name]["errors"][:3]
                ):  # Mostrar solo los primeros 3
                    print(f"    - {error}")
                if len(results[coord_name]["errors"]) > 3:
                    print(f"    ... ({len(results[coord_name]['errors']) - 3} más)")

        # --- Fin del bucle de coordenadas ---

    except FileNotFoundError as e:
        # Error si el archivo de coordenadas o índices principal no existe
        print(f"\nError Crítico: No se encontró un archivo esencial: {str(e)}")
        # results ya está inicializado, se devolverá vacío o parcialmente lleno
    except ValueError as e:
        # Errores durante la lectura inicial de coordenadas o JSON
        print(f"\nError Crítico de Valor: {str(e)}")
    except Exception as e:
        # Capturar cualquier otro error general en la configuración o bucle principal
        print(f"\nError General Inesperado durante la ejecución: {str(e)}")
        # Considerar añadir más detalles si es posible, e.g., traceback

    print("\n--- Fin del Procesamiento ---")
    return results


def main():
    """Función principal del script."""
    try:
        # Configurar rutas
        PROJECT_ROOT = Path(__file__).parent.parent
        DATASET_PATH = PROJECT_ROOT.parent / "COVID-19_Radiography_Dataset"
        COORDINATES_FILE = (
            PROJECT_ROOT.parent / "coordenadas/coordenadas_entrenamiento_1.csv"
        )
        INDICES_FILE = PROJECT_ROOT.parent / "indices/indices_entrenamiento_1.csv"
        VISUALIZATION_DIR = (
            PROJECT_ROOT.parent / "resultados/recorte/dataset_entrenamiento_1"
        )

        # Validar rutas
        if not DATASET_PATH.exists():
            raise FileNotFoundError(
                f"No se encontró el directorio del dataset: {DATASET_PATH}"
            )
        if not COORDINATES_FILE.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de coordenadas: {COORDINATES_FILE}"
            )
        if not INDICES_FILE.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de índices: {INDICES_FILE}"
            )

        print(f"Usando dataset en: {DATASET_PATH}")
        print(f"Archivo de coordenadas: {COORDINATES_FILE}")
        print(f"Archivo de índices: {INDICES_FILE}")

        # Inicializar componentes
        coord_manager = CoordinateManager()
        image_processor = ImageProcessor(
            base_path=str(DATASET_PATH), visualization_dir=str(VISUALIZATION_DIR)
        )

        # Cargar coordenadas de búsqueda
        coord_manager.read_search_coordinates(
            str(
                PROJECT_ROOT.parent
                / "resultados/region_busqueda/dataset_entrenamiento_1/json/all_search_coordinates.json"
            )
        )

        # Procesar imágenes
        results = process_images(
            coord_manager=coord_manager,
            image_processor=image_processor,
            coordinates_file=str(COORDINATES_FILE),
            indices_file=str(INDICES_FILE),
        )

        # Mostrar resumen final
        print("\nResumen final:")
        total_processed = sum(result["processed"] for result in results.values())
        total_failed = sum(result["failed"] for result in results.values())
        print(f"Total de imágenes procesadas exitosamente: {total_processed}")
        print(f"Total de imágenes fallidas: {total_failed}")
        print(f"Total de operaciones: {total_processed + total_failed}")

        # Mostrar coordenadas con errores
        coords_with_errors = [
            coord for coord, result in results.items() if result["failed"] > 0
        ]
        if coords_with_errors:
            print("\nCoordenadas con errores:")
            for coord in coords_with_errors:
                print(f"  - {coord}: {results[coord]['failed']} errores")

    except Exception as e:
        print(f"\nError durante la ejecución: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
