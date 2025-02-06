"""
Módulo para procesamiento paralelo de imágenes pulmonares.

Este módulo proporciona clases y funciones para:
- Procesamiento paralelo de imágenes usando múltiples CPUs
- División de trabajo en batches
- Monitoreo de progreso y rendimiento
"""

from multiprocessing import Pool, cpu_count, Queue, Lock, Manager
from typing import List, Dict, Any, Tuple
import time
import logging
import pandas as pd
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from ..image_processing.image_processor import ImageProcessor

@dataclass
class BatchResult:
    """Clase para almacenar resultados de procesamiento por batch."""
    batch_id: int
    processed: int
    failed: int
    errors: List[str]
    processing_time: float

class ParallelProcessor:
    """
    Clase para procesamiento paralelo de imágenes pulmonares.
    
    Esta clase maneja:
    - División de trabajo en batches
    - Procesamiento paralelo usando Pool
    - Monitoreo de progreso
    - Recolección de resultados
    """
    
    def __init__(self, 
                 image_processor: ImageProcessor,
                 num_cpus: int = None,
                 batch_size: int = 10):
        """
        Inicializa el procesador paralelo.
        
        Args:
            image_processor: Instancia de ImageProcessor
            num_cpus: Número de CPUs a usar (None = usar todos)
            batch_size: Tamaño de cada batch de imágenes
        """
        self.image_processor = image_processor
        self.num_cpus = num_cpus or cpu_count()
        self.batch_size = batch_size
        self.manager = Manager()
        self.progress_queue = self.manager.Queue()
        self.results_queue = self.manager.Queue()
        self.error_queue = self.manager.Queue()
        self.lock = Lock()
        
        # Configurar logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def create_batches(self, 
                      total_images: int,
                      coord_file: str,
                      indices_file: str) -> List[Dict]:
        """
        Divide el trabajo en batches.
        
        Args:
            total_images: Total de imágenes a procesar
            coord_file: Ruta al archivo de coordenadas
            indices_file: Ruta al archivo de índices
            
        Returns:
            Lista de diccionarios con información de cada batch
        """
        batches = []
        for i in range(0, total_images, self.batch_size):
            end_idx = min(i + self.batch_size, total_images)
            batches.append({
                'batch_id': len(batches),
                'start_idx': i,
                'end_idx': end_idx,
                'coord_file': coord_file,
                'indices_file': indices_file
            })
        return batches

    def process_single_image(self,
                           image_idx: int,
                           coord_file: str,
                           indices_file: str) -> Dict[str, Any]:
        """
        Procesa una sola imagen.
        
        Args:
            image_idx: Índice de la imagen
            coord_file: Ruta al archivo de coordenadas
            indices_file: Ruta al archivo de índices
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        try:
            # Cargar coordenadas para la imagen
            coords = self._load_coordinates(coord_file, image_idx)
            
            # Cargar imagen una sola vez
            image_path = self.image_processor.get_image_path(
                image_idx, indices_file
            )
            image = self.image_processor.load_and_resize_image(image_path)
            
            # Crear región de búsqueda una sola vez
            search_region = np.zeros((64, 64), dtype=np.uint8)
            search_region[20:40, 20:40] = 1
            
            results = {}
            # Procesar puntos 1 y 2 reutilizando la imagen cargada
            for coord_num in [1, 2]:
                coord_name = f"Coord{coord_num}"
                try:
                    # Obtener punto etiquetado
                    labeled_point = coords[coord_name]
                    
                    # Extraer región reutilizando la imagen y región de búsqueda
                    cropped = self.image_processor.extract_region(
                        image=image,
                        search_region=search_region,
                        labeled_point=labeled_point,
                        coord_num=coord_num
                    )
                    
                    # Guardar imagen recortada
                    success = self.image_processor.save_cropped_image(
                        cropped, coord_name, image_idx
                    )
                    
                    results[coord_name] = {
                        'success': success,
                        'error': None if success else "Error al guardar imagen"
                    }
                    
                except Exception as e:
                    results[coord_name] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Liberar memoria explícitamente
            del image
            del search_region
            
            return {
                'image_idx': image_idx,
                'results': results
            }
            
        except Exception as e:
            return {
                'image_idx': image_idx,
                'results': {
                    'Coord1': {'success': False, 'error': str(e)},
                    'Coord2': {'success': False, 'error': str(e)}
                }
            }

    def _load_coordinates(self, coord_file: str, image_idx: int) -> Dict:
        """
        Carga las coordenadas para una imagen específica.
        
        Args:
            coord_file: Ruta al archivo de coordenadas
            image_idx: Índice de la imagen
            
        Returns:
            Diccionario con coordenadas por punto
            
        Raises:
            ValueError: Si no se encuentran coordenadas para el índice
        """
        try:
            # Implementar carga de coordenadas thread-safe
            with self.lock:
                # Leer archivo CSV
                data = pd.read_csv(coord_file, header=None)
                
                # Buscar fila correspondiente al índice
                row = data[data[0] == image_idx]
                if row.empty:
                    raise ValueError(f"No se encontraron coordenadas para el índice {image_idx}")
                
                row = row.iloc[0]
                
                # Las coordenadas están en las columnas:
                # Coord1_x, Coord1_y, Coord2_x, Coord2_y
                return {
                    'Coord1': (int(row[1]), int(row[2])),  # x, y para Coord1
                    'Coord2': (int(row[3]), int(row[4]))   # x, y para Coord2
                }
                
        except Exception as e:
            self.logger.error(f"Error cargando coordenadas para índice {image_idx}: {str(e)}")
            raise

    def process_batch(self, batch: Dict) -> BatchResult:
        """
        Procesa un batch de imágenes.
        
        Args:
            batch: Diccionario con información del batch
            
        Returns:
            BatchResult con resultados del procesamiento
        """
        start_time = time.time()
        processed = 0
        failed = 0
        errors = []
        
        try:
            for idx in range(batch['start_idx'], batch['end_idx']):
                result = self.process_single_image(
                    idx,
                    batch['coord_file'],
                    batch['indices_file']
                )
                
                # Contar éxitos y fallos
                for coord_result in result['results'].values():
                    if coord_result['success']:
                        processed += 1
                    else:
                        failed += 1
                        if coord_result['error']:
                            errors.append(
                                f"Error en imagen {idx}: {coord_result['error']}"
                            )
                
                # Actualizar progreso
                self.progress_queue.put({
                    'batch_id': batch['batch_id'],
                    'current': idx - batch['start_idx'] + 1,
                    'total': batch['end_idx'] - batch['start_idx']
                })
                
        except Exception as e:
            errors.append(f"Error en batch {batch['batch_id']}: {str(e)}")
        
        processing_time = time.time() - start_time
        
        return BatchResult(
            batch_id=batch['batch_id'],
            processed=processed,
            failed=failed,
            errors=errors,
            processing_time=processing_time
        )

    def process_all(self, 
                   total_images: int,
                   coord_file: str,
                   indices_file: str) -> Tuple[List[BatchResult], List[str]]:
        """
        Procesa todas las imágenes en paralelo.
        
        Args:
            total_images: Total de imágenes a procesar
            coord_file: Ruta al archivo de coordenadas
            indices_file: Ruta al archivo de índices
            
        Returns:
            Tupla con (lista de resultados por batch, lista de errores)
        """
        # Crear batches
        batches = self.create_batches(total_images, coord_file, indices_file)
        
        # Iniciar pool de procesos
        with Pool(self.num_cpus) as pool:
            # Procesar batches
            results = pool.map(self.process_batch, batches)
            
        # Recolectar errores
        all_errors = []
        for result in results:
            all_errors.extend(result.errors)
            
        return results, all_errors

    def get_progress(self) -> Dict[int, float]:
        """
        Obtiene el progreso actual de todos los batches.
        
        Returns:
            Diccionario con progreso por batch_id
        """
        progress = {}
        while not self.progress_queue.empty():
            update = self.progress_queue.get()
            batch_id = update['batch_id']
            progress[batch_id] = update['current'] / update['total']
        return progress
