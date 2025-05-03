# Tesis/alineamiento/src/ssm_builder.py

import numpy as np
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def vectorize_shapes(aligned_shapes):
    """
    Convierte un array de formas alineadas (N, k, d) en una matriz de datos (N, k*d).
    Cada fila representa una forma desenrollada en un vector.

    Args:
        aligned_shapes (np.ndarray): Array de formas alineadas, shape (N, k, d).

    Returns:
        np.ndarray: Matriz de datos vectorizada, shape (N, k*d), o None si el input es inválido.
    """
    if aligned_shapes is None or aligned_shapes.ndim != 3:
        logging.error("Input 'aligned_shapes' inválido para vectorizar. Se requiere (N, k, d).")
        return None
        
    N, k, d = aligned_shapes.shape
    data_matrix = aligned_shapes.reshape(N, k * d)
    logging.info(f"Formas vectorizadas de ({N}, {k}, {d}) a ({N}, {k*d}).")
    return data_matrix

def devectorize_shape(shape_vector, k, d):
    """
    Convierte un vector de forma (k*d,) de nuevo a una matriz de forma (k, d).

    Args:
        shape_vector (np.ndarray): Vector de forma, shape (k*d,).
        k (int): Número de landmarks.
        d (int): Número de dimensiones (normalmente 2).

    Returns:
        np.ndarray: Matriz de forma, shape (k, d), o None si el input es inválido.
    """
    expected_len = k * d
    if shape_vector is None or shape_vector.ndim != 1 or shape_vector.shape[0] != expected_len:
        logging.error(f"Input 'shape_vector' inválido. Se requiere ({expected_len},) pero tiene shape {shape_vector.shape if shape_vector is not None else 'None'}.")
        return None
        
    try:
        shape_matrix = shape_vector.reshape(k, d)
        return shape_matrix
    except ValueError as e:
        logging.error(f"Error al reestructurar el vector a ({k}, {d}): {e}")
        return None


def build_pca_model(data_matrix, n_components=None):
    """
    Construye un modelo PCA a partir de la matriz de datos vectorizada.

    Args:
        data_matrix (np.ndarray): Matriz de datos (N_samples, N_features=k*d).
        n_components (int, float, str or None): Número de componentes a mantener.
            - None: Mantiene min(n_samples, n_features).
            - int: Mantiene ese número de componentes.
            - float (0<n<1): Mantiene componentes que explican esa fracción de varianza.

    Returns:
        tuple: (pca_model, mean_vector, components, explained_variance, std_devs)
               - pca_model: Objeto PCA de sklearn entrenado.
               - mean_vector (np.ndarray): Vector de forma media (k*d,).
               - components (np.ndarray): Componentes principales (eigenvectores), shape (n_components, k*d).
               - explained_variance (np.ndarray): Varianza explicada por componente (eigenvalores), shape (n_components,).
               - std_devs (np.ndarray): Desviación estándar por componente (sqrt(eigenvalores)), shape (n_components,).
            O (None, None, None, None, None) si hay error.
    """
    if data_matrix is None or data_matrix.ndim != 2:
        logging.error("Input 'data_matrix' inválido para PCA. Se requiere (N, k*d).")
        return None, None, None, None, None

    logging.info(f"Construyendo modelo PCA con n_components={n_components} sobre matriz de shape {data_matrix.shape}...")
    
    try:
        pca = PCA(n_components=n_components)
        pca.fit(data_matrix)

        mean_vector = pca.mean_
        components = pca.components_
        explained_variance = pca.explained_variance_
        std_devs = np.sqrt(explained_variance)

        logging.info(f"Modelo PCA construido. Número de componentes retenidos: {pca.n_components_}")
        logging.info(f"Varianza total explicada: {np.sum(pca.explained_variance_ratio_):.4f}")

        return pca, mean_vector, components, explained_variance, std_devs

    except Exception as e:
        logging.error(f"Error durante el ajuste de PCA: {e}")
        return None, None, None, None, None