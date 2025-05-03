# Tesis/alineamiento/src/alignment.py

import numpy as np
import logging
from scipy.linalg import orthogonal_procrustes # Importar la función de SciPy

# Establecer en INFO ahora que la depuración anterior no funcionó como se esperaba
# Podemos volver a DEBUG si es necesario
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s') 


class ProcrustesAligner:
    """
    Realiza GPA.
    *** USA ESTRUCTURA DE BUCLE "v2" (Alternativa - Numéricamente Estable) ***
    *** USA scipy.linalg.orthogonal_procrustes para el cálculo de rotación ***
    """
    
    def _center_shape(self, shape):
        """Centra una forma restando su centroide."""
        centroid = np.mean(shape, axis=0)
        return shape - centroid

    def _normalize_scale(self, shape):
        """Normaliza la escala de una forma (asume que está centrada)."""
        norm = np.linalg.norm(shape)
        if norm < 1e-10: 
            logging.warning("Forma con norma cercana a cero encontrada durante la normalización.")
            return shape 
        return shape / norm

    # _is_orthogonal ya no es necesaria si confiamos en SciPy

    def _find_optimal_rotation(self, shape_a, shape_b):
        """
        Encuentra la matriz de rotación óptima R para alinear shape_a con shape_b
        (minimiza || shape_a @ R - shape_b ||) usando SciPy.
        Asume que ambas formas están centradas y escaladas.
        """
        # orthogonal_procrustes(A, B) encuentra R que minimiza ||AR - B||
        try:
            R, scale = orthogonal_procrustes(shape_a, shape_b)
            # scale debería ser cercano a 1 si las shapes están normalizadas
            if abs(scale - 1.0) > 1e-3: # Un pequeño chequeo
                 logging.warning(f"orthogonal_procrustes devolvió una escala inesperada: {scale:.4f}")
            
            # No necesitamos corregir reflexión, SciPy debería devolver rotación propia
            # det_R = np.linalg.det(R) 
            # logging.debug(f"Rotation from SciPy, det={det_R:.4f}")

            return R
        except Exception as e:
            logging.error(f"Error en orthogonal_procrustes: {e}. Devolviendo matriz identidad.")
            return np.identity(shape_a.shape[1]) 

    def gpa(self, shapes, max_iters=100, tolerance=1e-6): # Valores por defecto razonables
        """
        Realiza GPA usando la estructura de bucle "v2" (estable) 
        y la rotación de SciPy.
        """
        if shapes is None or shapes.ndim != 3 or shapes.shape[0] < 2:
            logging.error("Input 'shapes' inválido para GPA.")
            return None, None
            
        N, k, d = shapes.shape
        logging.info(f"Iniciando GPA (v2+SciPy) para {N} formas, {k} landmarks, {d}D (max_iters={max_iters}, tolerance={tolerance})...")

        centered_shapes = np.array([self._center_shape(s) for s in shapes])
        normalized_shapes = np.array([self._normalize_scale(s) for s in centered_shapes]) 
        if np.isnan(normalized_shapes).any():
            logging.error("NaNs detectados después de la normalización inicial. Abortando.")
            return None, None
            
        current_mean_shape = np.mean(normalized_shapes, axis=0) 
        current_mean_shape = self._normalize_scale(current_mean_shape) 
        
        # Este array guardará las formas alineadas en CADA iteración
        # (necesario para calcular la siguiente media)
        aligned_shapes_in_iter = np.zeros_like(normalized_shapes) 

        for iteration in range(max_iters):
            mean_shape_old = current_mean_shape.copy()

            # --- BUCLE ALTERNATIVO ("v2") ---
            # Alinear todas las formas *originales normalizadas* a la media actual (mean_shape_old)
            for i in range(N):
                R = self._find_optimal_rotation(normalized_shapes[i], mean_shape_old)
                # Aplicar R a la forma original normalizada
                aligned_shapes_in_iter[i] = normalized_shapes[i] @ R 
            # --- FIN BUCLE ALTERNATIVO ---

            if np.isnan(aligned_shapes_in_iter).any() or np.isinf(aligned_shapes_in_iter).any():
                logging.error(f"NaN/Inf detectado en aligned_shapes_in_iter en iteración {iteration + 1}. Abortando.")
                return None, mean_shape_old 

            # Recalcular la forma media a partir de las formas alineadas en ESTA iteración
            current_mean_shape = np.mean(aligned_shapes_in_iter, axis=0)
            current_mean_shape = self._normalize_scale(current_mean_shape) 
            mean_norm_after_norm = np.linalg.norm(current_mean_shape) 

            if np.isnan(current_mean_shape).any() or np.isinf(current_mean_shape).any():
                 logging.error(f"NaN/Inf detectado en current_mean_shape en iteración {iteration + 1}. Abortando.")
                 # Devolver las formas alineadas de esta iteración y la media anterior
                 return aligned_shapes_in_iter, mean_shape_old 

            mean_diff_norm = np.linalg.norm(current_mean_shape - mean_shape_old)
            
            logging.info(f"Iter {iteration + 1}: Mean norm={mean_norm_after_norm:.6f}, Diff norm={mean_diff_norm:.8f}")

            if mean_diff_norm < tolerance:
                logging.info(f"GPA convergió en {iteration + 1} iteraciones (tolerancia={tolerance}).")
                break
        else: 
            logging.warning(f"GPA no convergió después de {max_iters} iteraciones. Norma diferencia final: {mean_diff_norm:.8f}")

        # Devolvemos las formas alineadas de la *última* iteración y la media final
        return aligned_shapes_in_iter, current_mean_shape