# Tesis/alineamiento/src/ssm_fitter.py (vP2-Zscore: Normales Estables, Escala Fija, Z-Score Norm)
import numpy as np
import cv2
from scipy.linalg import orthogonal_procrustes, inv, pinv
from scipy import ndimage
import logging
logger = logging.getLogger(__name__)
import os

# Asumiendo que ssm_builder está en el mismo directorio o accesible
try:
    from .ssm_builder import devectorize_shape
except ImportError:
    # Fallback si se ejecuta como script principal o desde otro lugar
    from ssm_builder import devectorize_shape

# --- Funciones Auxiliares (calculate_normal, sample_profile, normalize_profile) ---
#     (Mantener las mismas versiones que en profile_builder.py)
def calculate_normal(p_prev, p_next):
    """Calcula el vector normal unitario a un segmento."""
    segment_vec = p_next - p_prev
    segment_len = np.linalg.norm(segment_vec)
    if segment_len < 1e-6: return np.array([0.0, -1.0])
    normal = np.array([-segment_vec[1], segment_vec[0]]) / segment_len
    return normal

def sample_profile(image, point, normal, length, spacing=1.0):
    """Muestrea un perfil de intensidad a lo largo de una normal."""
    half_length = (length - 1) / 2
    distances = np.linspace(-half_length * spacing, half_length * spacing, length)
    norm_mag = np.linalg.norm(normal)
    safe_normal = normal / norm_mag if norm_mag > 1e-6 else np.array([0.0, -1.0])
    sample_coords_row = point[1] + distances * safe_normal[1]
    sample_coords_col = point[0] + distances * safe_normal[0]
    try:
        profile = ndimage.map_coordinates(image,
                                          [sample_coords_row, sample_coords_col],
                                          order=1, mode='nearest', cval=0.0)
    except Exception as e:
        profile = np.zeros(length)
    return profile

def normalize_profile(profile, method='sum'):
    """Normaliza un perfil usando el método especificado."""
    if method == 'sum':
        norm_factor = np.sum(np.abs(profile)) + 1e-6
        return profile / norm_factor
    elif method == 'zscore':
        mean = np.mean(profile)
        std = np.std(profile) + 1e-6
        return (profile - mean) / std
    else:
        # logger.warning(f"Método de normalización '{method}' no reconocido. Devolviendo perfil original.")
        return profile
# --------------------------------------------

class SSMFitter:
    """
    Implementa el ajuste iterativo de SSM usando búsqueda local basada en
    Modelos de Perfil ASM y Distancia de Mahalanobis.
    *** VERSIÓN P2-Zscore: Normales Estables + Escala Fija + Normalización Z-SCORE ***
    """
    def __init__(self, mean_vector_path, components_path, std_devs_path,
                 profile_models_path,
                 num_landmarks=15, num_dims=2):

        self.k = num_landmarks
        self.d = num_dims
        logger.info("Cargando modelo SSM...")
        try:
            if not os.path.exists(mean_vector_path): raise FileNotFoundError(f"No encontrado: {mean_vector_path}")
            self.mean_vector = np.load(mean_vector_path)

            if not os.path.exists(components_path): raise FileNotFoundError(f"No encontrado: {components_path}")
            self.P = np.load(components_path) # Shape (num_modes, k*d)

            if not os.path.exists(std_devs_path): raise FileNotFoundError(f"No encontrado: {std_devs_path}")
            self.std_devs = np.load(std_devs_path) # Shape (num_modes,)

            self.num_modes = self.P.shape[0]

            # Validaciones de dimensiones
            if self.mean_vector.shape != (self.k * self.d,):
                raise ValueError(f"Mean vector shape {self.mean_vector.shape} no coincide con ({self.k * self.d},)")
            if self.P.shape != (self.num_modes, self.k * self.d):
                 raise ValueError(f"Components matrix P shape {self.P.shape} no coincide con ({self.num_modes}, {self.k * self.d})")
            if self.std_devs.shape != (self.num_modes,):
                raise ValueError(f"Std Devs vector shape {self.std_devs.shape} no coincide con ({self.num_modes},)")

            # Calcular y guardar forma media (k, d) desde el vector medio
            self.mean_shape = self._vector_to_shape(self.mean_vector)
            if self.mean_shape is None:
                 raise ValueError("No se pudo generar mean_shape desde mean_vector.")

            logger.info(f"Modelo SSM cargado: {self.num_modes} modos, {self.k} landmarks.")
            logger.info(f"Forma media (mean_shape) tiene shape: {self.mean_shape.shape}")

        except FileNotFoundError as e:
             logger.error(f"Error crítico: Archivo SSM no encontrado: {e}", exc_info=True); raise
        except ValueError as e:
             logger.error(f"Error crítico: Dimensiones inconsistentes en archivos SSM: {e}", exc_info=True); raise
        except Exception as e:
             logger.error(f"Error crítico inesperado al cargar modelo SSM: {e}", exc_info=True); raise

        logger.info("Cargando modelos de perfil ASM...")
        try:
            if not os.path.exists(profile_models_path): raise FileNotFoundError(f"No encontrado: {profile_models_path}")

            profile_data = np.load(profile_models_path, allow_pickle=True)
            # Asegurarse que la clave 'models' existe
            if 'models' not in profile_data:
                 raise KeyError("La clave 'models' no se encontró en el archivo de perfiles.")
            self.profile_models = profile_data['models'] # Esto debería ser un array de diccionarios

            if not isinstance(self.profile_models, np.ndarray) or self.profile_models.ndim != 1:
                 raise TypeError("Se esperaba un array NumPy 1D de diccionarios en 'models'.")
            if len(self.profile_models) != self.k:
                raise ValueError(f"Número de modelos de perfil ({len(self.profile_models)}) no coincide con k ({self.k}).")

            # Verificar estructura del primer modelo y obtener longitud de perfil
            if self.k > 0:
                 if not isinstance(self.profile_models[0], dict) or 'mean' not in self.profile_models[0]:
                      raise ValueError("El primer modelo de perfil no es un diccionario o no contiene la clave 'mean'.")
                 self.profile_length = len(self.profile_models[0]['mean'])
                 if not isinstance(self.profile_length, int) or self.profile_length <= 0:
                      raise ValueError("Longitud de perfil inválida.")
            else:
                 self.profile_length = 0 # O manejar caso k=0 si es posible

            # --- DEFINIR MÉTODO DE NORMALIZACIÓN PARA BÚSQUEDA LOCAL ---
            self.profile_norm_method = 'zscore' # <-- *** CAMBIO REALIZADO AQUÍ ***
            # --------------------------------------------------------
            logger.info(f"Modelos de perfil cargados para {len(self.profile_models)} landmarks.")
            logger.info(f"Longitud de perfil detectada: {self.profile_length}.")
            logger.info(f"Método de normalización de perfiles para búsqueda local: {self.profile_norm_method.upper()}")

        except FileNotFoundError as e:
             logger.error(f"Error crítico: Archivo de modelos de perfil no encontrado: {e}", exc_info=True); raise
        except (KeyError, ValueError, TypeError) as e:
             logger.error(f"Error crítico: Formato incorrecto en archivo de modelos de perfil: {e}", exc_info=True); raise
        except Exception as e:
             logger.error(f"Error crítico inesperado al cargar modelos de perfil ASM: {e}", exc_info=True); raise


    # --- Métodos _vector_to_shape, _shape_to_vector ---
    def _vector_to_shape(self, vec):
         """Convierte un vector (k*d,) a una forma (k, d)."""
         if vec is None or vec.shape != (self.k * self.d,):
              # logger.warning(f"_vector_to_shape: Input inválido shape {vec.shape if vec is not None else 'None'}")
              return None
         try:
              return vec.reshape((self.k, self.d))
         except ValueError as e:
              # logger.error(f"Error en reshape de _vector_to_shape: {e}")
              return None

    def _shape_to_vector(self, shape):
         """Convierte una forma (k, d) a un vector (k*d,)."""
         if shape is None or shape.shape != (self.k, self.d):
              # logger.warning(f"_shape_to_vector: Input inválido shape {shape.shape if shape is not None else 'None'}")
              return None
         return shape.flatten()

    # --- Métodos de Modelo Estadístico (_generate_shape_instance, _project_shape_to_model, _clamp_shape_params) ---
    def _generate_shape_instance(self, b):
         """Genera una instancia de forma a partir de los parámetros del modelo b."""
         if b is None or b.shape != (self.num_modes,):
              # logger.warning(f"_generate_shape_instance: Parámetros b inválidos shape {b.shape if b is not None else 'None'}. Usando mean_shape.")
              return self.mean_shape.copy() # Devolver copia para evitar modificaciones accidentales
         try:
              # shape_vector = mean_vector + P^T * b
              shape_vector = self.mean_vector + self.P.T @ b
         except ValueError as e:
              logger.error(f"Error en matmul de _generate_shape_instance (P.T @ b): {e}. Usando mean_shape.")
              return self.mean_shape.copy()
         return self._vector_to_shape(shape_vector)

    def _project_shape_to_model(self, shape_in_model_space):
         """Proyecta una forma (en espacio Procrustes) al espacio de parámetros b."""
         if shape_in_model_space is None or shape_in_model_space.shape != (self.k, self.d):
              logger.warning(f"_project_shape_to_model: Input inválido shape {shape_in_model_space.shape if shape_in_model_space is not None else 'None'}. Devolviendo b=0.")
              return np.zeros(self.num_modes)

         shape_vector = self._shape_to_vector(shape_in_model_space)
         if shape_vector is None:
              logger.warning("_project_shape_to_model: No se pudo vectorizar la forma. Devolviendo b=0.")
              return np.zeros(self.num_modes)

         try:
              # b = P * (shape_vector - mean_vector)
              b = self.P @ (shape_vector - self.mean_vector)
         except ValueError as e:
              logger.error(f"Error en matmul de _project_shape_to_model (P @ diff): {e}. Devolviendo b=0.")
              return np.zeros(self.num_modes)
         return b

    def _clamp_shape_params(self, b, n_std_devs=3.0):
         """Limita los parámetros b a +/- n desviaciones estándar."""
         if b is None or b.shape != (self.num_modes,):
              logger.warning(f"_clamp_shape_params: b inválido shape {b.shape if b is not None else 'None'}. Devolviendo b=0.")
              return np.zeros(self.num_modes)
         if self.std_devs is None or self.std_devs.shape != (self.num_modes,):
              logger.warning("_clamp_shape_params: std_devs no disponible. No se aplicará clamping.")
              return b # Devolver sin cambios si no hay std_devs

         # Calcular límites mínimo y máximo para cada modo
         min_b = -n_std_devs * self.std_devs
         max_b = n_std_devs * self.std_devs
         # Aplicar clip
         return np.clip(b, min_b, max_b)

    # --- Métodos de Transformación de Similitud (_apply_similarity_transform, _get_similarity_transform_params) ---
    def _apply_similarity_transform(self, shape, s, theta_rad, tx, ty):
         """Aplica una transformación de similitud (s, theta, tx, ty) a una forma."""
         if shape is None or shape.shape != (self.k, self.d):
              logger.warning(f"_apply_similarity_transform: Input inválido shape {shape.shape if shape is not None else 'None'}")
              return None

         # Crear matriz de rotación 2D
         R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                       [np.sin(theta_rad),  np.cos(theta_rad)]])

         # Aplicar transformación: shape_img = s * R * shape_model + t
         # OJO: En álgebra lineal, la rotación se aplica como shape @ R.T si shape es (N, 2)
         transformed_shape = s * (shape @ R.T)
         # Añadir traslación
         transformed_shape[:, 0] += tx
         transformed_shape[:, 1] += ty
         return transformed_shape

    def _get_similarity_transform_params(self, shape_from, shape_to):
         """
         Estima los parámetros (s, theta, tx, ty) de la transformación de similitud
         que mejor alinea shape_from con shape_to usando Procrustes.
         shape_from: Forma origen (ej. modelo en espacio Procrustes)
         shape_to: Forma destino (ej. puntos objetivo en la imagen)
         Retorna: s_scale, theta_rad, tx, ty
         """
         if shape_from is None or shape_to is None or \
            shape_from.shape != (self.k, self.d) or shape_to.shape != (self.k, self.d) or \
            self.k < 2: # Necesitamos al menos 2 puntos
              logger.warning(f"_get_similarity_transform_params: Inputs inválidos. From:{shape_from.shape if shape_from is not None else 'None'}, To:{shape_to.shape if shape_to is not None else 'None'}. Devolviendo identidad.")
              return 1.0, 0.0, 0.0, 0.0 # Transformación identidad por defecto

         # 1. Centrar ambas formas
         centroid_from = np.mean(shape_from, axis=0)
         centered_from = shape_from - centroid_from
         centroid_to = np.mean(shape_to, axis=0)
         centered_to = shape_to - centroid_to

         # 2. Calcular Rotación óptima R usando SciPy Procrustes
         #    orthogonal_procrustes(A, B) encuentra R que minimiza ||AR - B||_F
         #    Aquí A = centered_from, B = centered_to
         try:
              R, scale_procrustes = orthogonal_procrustes(centered_from, centered_to)
              # scale_procrustes es sum(svdvals), no la escala directa que buscamos
         except Exception as e:
              logger.error(f"Error en orthogonal_procrustes: {e}. Usando matriz identidad.")
              R = np.identity(self.d) # Matriz identidad si falla

         # 3. Calcular Escala óptima s
         #    s = ||centered_to||_F / ||centered_from||_F
         norm_from = np.linalg.norm(centered_from, 'fro')
         norm_to = np.linalg.norm(centered_to, 'fro')
         s_scale = norm_to / (norm_from + 1e-9) # Evitar división por cero si shape_from es un punto

         # 4. Calcular Ángulo de rotación theta desde R
         #    arctan2(R[1, 0], R[0, 0]) es robusto
         theta_rad = np.arctan2(R[1, 0], R[0, 0])

         # 5. Calcular Traslación t
         #    t = centroid_to - s * R * centroid_from (en álgebra matricial)
         #    O, equivalentemente: t = centroid_to - s * (centroid_from @ R.T)
         translation = centroid_to - s_scale * (centroid_from @ R.T)
         tx, ty = translation[0], translation[1]

         return s_scale, theta_rad, tx, ty

    # --------------------------------------------------------------------------

    def _search_local_features(self, image, estimated_points, search_pixels=10,
                               gaussian_ksize=3):
        """
        Busca mejores posiciones para cada landmark usando ASM Mahalanobis.
        *** USA NORMALES ESTABLES (calculadas desde self.mean_shape). ***
        *** USA EL MÉTODO DE NORMALIZACIÓN self.profile_norm_method (AHORA ZSCORE). ***
        """
        if image is None or image.ndim != 2:
             logger.error("Input 'image' inválido para búsqueda ASM (None o no escala de grises).")
             return estimated_points # Devolver puntos originales si la imagen es mala
        if estimated_points is None or estimated_points.shape != (self.k, self.d):
            logger.error(f"Input 'estimated_points' inválido para búsqueda ASM. Shape: {estimated_points.shape if estimated_points is not None else 'None'}")
            # Podríamos intentar devolver algo, pero es mejor indicar fallo
            return None # Indicar fallo

        logger.debug(f"Iniciando búsqueda local ASM (Normales Estables P2, Norm={self.profile_norm_method.upper()})...")
        H, W = image.shape[:2]
        target_points = np.zeros_like(estimated_points) # Array para guardar los mejores puntos encontrados

        # Pre-calcular gradiente (una sola vez por iteración del fitter)
        processed_image = image.astype(np.float64) # Usar float para cálculos de gradiente
        if gaussian_ksize > 0 and gaussian_ksize % 2 == 1:
            processed_image = cv2.GaussianBlur(processed_image, (gaussian_ksize, gaussian_ksize), 0)
        grad_x = cv2.Sobel(processed_image, cv2.CV_64F, 1, 0, ksize=5) # Usar k=5 como en el builder
        grad_y = cv2.Sobel(processed_image, cv2.CV_64F, 0, 1, ksize=5)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Iterar sobre cada landmark estimado
        for i in range(self.k):
            p_current = estimated_points[i] # Punto estimado actual para landmark i

            # === CÁLCULO DE NORMAL ESTABLE (desde la forma media Procrustes) ===
            # Usamos los vecinos en la *forma media* para definir la normal
            # Esto hace que la dirección de búsqueda sea consistente y no dependa
            # de la estimación actual (que puede ser ruidosa).
            idx_prev = (i - 1 + self.k) % self.k
            idx_next = (i + 1) % self.k
            p_prev_mean = self.mean_shape[idx_prev]
            p_next_mean = self.mean_shape[idx_next]
            normal = calculate_normal(p_prev_mean, p_next_mean)
            # ===================================================================

            min_maha_dist = float('inf') # Inicializar distancia mínima a infinito
            best_distance_offset = 0.0 # Mejor desplazamiento encontrado a lo largo de la normal

            # Buscar a lo largo de la normal en la imagen de magnitud de gradiente
            for d_offset in range(-search_pixels, search_pixels + 1):
                # Calcular punto candidato
                p_candidate = p_current + d_offset * normal

                # Verificar si el punto candidato está dentro de los límites de la imagen
                # Usamos una pequeña tolerancia para evitar errores de borde justos
                if not (0 <= p_candidate[0] < W and 0 <= p_candidate[1] < H):
                    continue # Saltar si está fuera de la imagen

                # Muestrear el perfil de gradiente observado en el punto candidato
                #observed_profile = sample_profile(grad_mag, p_candidate, normal, self.profile_length)
                observed_profile = sample_profile(processed_image, p_candidate, normal, self.profile_length)

                # Normalizar el perfil observado usando el método configurado (ZSCORE ahora)
                norm_observed_profile = normalize_profile(observed_profile, method=self.profile_norm_method)

                # Calcular la distancia de Mahalanobis al modelo de este landmark
                model_i = self.profile_models[i]
                diff = norm_observed_profile - model_i['mean']
                try:
                    # maha_dist = diff^T * inv_cov * diff
                    maha_dist = diff.T @ model_i['inv_cov'] @ diff
                    # Asegurarse que no sea NaN/inf
                    if not np.isfinite(maha_dist): maha_dist = float('inf')
                except Exception as e:
                    # logger.debug(f"      Excepción calculando Mahalanobis para Lmk {i}, d={d_offset}: {e}")
                    maha_dist = float('inf') # Penalizar si hay error

                # Actualizar si encontramos una distancia menor
                if maha_dist < min_maha_dist:
                    min_maha_dist = maha_dist
                    best_distance_offset = d_offset

            # El mejor punto encontrado es el punto actual más el mejor desplazamiento
            target_point = p_current + best_distance_offset * normal

            # Asegurar (clip) que el punto final esté estrictamente dentro de los límites de la imagen
            target_point[0] = np.clip(target_point[0], 0, W - 1) # Coordenada X (columna)
            target_point[1] = np.clip(target_point[1], 0, H - 1) # Coordenada Y (fila)

            target_points[i] = target_point # Guardar el mejor punto encontrado
            logger.debug(f"      Lmk {i}: Normal Estable. Best match d={best_distance_offset}, min_Maha^2={min_maha_dist:.4f}")

        return target_points

    def fit(self, image, initial_s=1.0, initial_theta_rad=0.0, initial_tx=None, initial_ty=None, initial_b=None,
            max_iters=100, tolerance=0.05, search_pixels_per_iter=10,
            damping_factor_theta=0.5, damping_factor_t=0.5,
            damping_factor_s=0.0, # <-- *** MANTENER ESCALA FIJA (P2) ***
            gaussian_ksize_per_iter=3,
            clamp_n_std_devs=3.0):
        """
        Ajusta el modelo SSM a una imagen dada usando búsqueda ASM Mahalanobis.
        *** VERSIÓN P2-Zscore: Normales Estables + ESCALA FIJA + Normalización Z-SCORE ***
        """
        if image is None or image.ndim != 2:
             logger.error("Fitter.fit: Imagen de entrada inválida.")
             return None, None, None

        H, W = image.shape[:2]

        # --- Inicialización de Parámetros de Pose y Forma ---
        s = initial_s # Escala inicial (Fija durante el ajuste si damping_factor_s=0)
        theta_rad = initial_theta_rad # Rotación inicial
        # Parámetros de forma iniciales (si no se proveen, empezar con b=0 -> forma media)
        b = initial_b if initial_b is not None else np.zeros(self.num_modes)
        b = self._clamp_shape_params(b, n_std_devs=clamp_n_std_devs) # Asegurar que b inicial sea válido

        # Inicialización de Traslación (si no se provee, centrar la forma inicial en la imagen)
        if initial_tx is None or initial_ty is None:
            # Generar la forma modelo inicial (basada en b inicial)
            shape_init_model = self._generate_shape_instance(b)
            if shape_init_model is None:
                 logger.error("Fallo al generar forma inicial desde b. Abortando fit.")
                 return None, None, None

            # Calcular centroide de la forma modelo inicial
            centroid_model = np.mean(shape_init_model, axis=0)
            # Calcular la traslación necesaria para centrarla en la imagen,
            # considerando la escala y rotación iniciales.
            c, si = np.cos(theta_rad), np.sin(theta_rad)
            tx = W / 2.0 - (s * (centroid_model[0] * c - centroid_model[1] * si))
            ty = H / 2.0 - (s * (centroid_model[0] * si + centroid_model[1] * c))
            logger.info(f"Init: Centrando forma inicial. Pose: s={s:.3f}(Fija), th={np.degrees(theta_rad):.1f}, t=({tx:.1f},{ty:.1f})")
        else:
            # Usar traslación inicial provista
            tx = initial_tx
            ty = initial_ty
            logger.info(f"Init: Usando pose provista. s={s:.3f}(Fija), th={np.degrees(theta_rad):.1f}, t=({tx:.1f},{ty:.1f})")

        logger.info(f"Inicio Fit [ASM Mahalanobis P2-Zscore NormalesEstables, Escala FIJA]: s={s:.3f}, th={np.degrees(theta_rad):.1f}, t=({tx:.1f},{ty:.1f}), |b|={np.linalg.norm(b):.3f}")
        logger.info(f"Params b iniciales: {np.round(b, 3)}")

        # Variables para convergencia
        shape_image_estimated_old = None
        converged = False

        # --- Bucle Iterativo Principal ---
        for iteration in range(max_iters):
            logger.debug(f"--- Fit Iteration {iteration + 1}/{max_iters} [ASM P2-Zscore Normales Estables, Escala Fija]---")

            # 1. Generar instancia de forma actual desde parámetros 'b'
            shape_model = self._generate_shape_instance(b)
            if shape_model is None:
                 logger.error(f"Iter {iteration+1}: Fallo al generar shape_model desde b. Abortando.");
                 # Devolver estado anterior si existe
                 final_shape_img = shape_image_estimated_old
                 final_pose = {'s': s, 'theta_rad': theta_rad, 'tx': tx, 'ty': ty}
                 return final_shape_img, b, final_pose

            # 2. Aplicar transformación de similitud actual (s, th, tx, ty) para obtener puntos en la imagen
            shape_image_estimated = self._apply_similarity_transform(shape_model, s, theta_rad, tx, ty)
            if shape_image_estimated is None:
                 logger.error(f"Iter {iteration+1}: Fallo al aplicar transform de similitud. Abortando.");
                 final_shape_img = shape_image_estimated_old
                 final_pose = {'s': s, 'theta_rad': theta_rad, 'tx': tx, 'ty': ty}
                 return final_shape_img, b, final_pose

            # Guardar la forma de la iteración anterior para el chequeo de convergencia
            if iteration == 0:
                shape_image_estimated_old = shape_image_estimated.copy()

            # 3. Buscar nuevas posiciones objetivo (target) en la imagen para cada landmark
            #    Usando la búsqueda local ASM Mahalanobis (con normales estables y norm ZSCORE)
            shape_image_target = self._search_local_features(
                image, shape_image_estimated,
                search_pixels=search_pixels_per_iter,
                gaussian_ksize=gaussian_ksize_per_iter
            )
            # Verificar si la búsqueda local falló (devolvió None)
            if shape_image_target is None:
                logger.error(f"Iter {iteration+1}: Fallo la búsqueda local ASM (_search_local_features devolvió None). Abortando.");
                final_shape_img = shape_image_estimated # Usar la última estimación válida
                final_pose = {'s': s, 'theta_rad': theta_rad, 'tx': tx, 'ty': ty}
                return final_shape_img, b, final_pose
            if shape_image_target.shape != shape_image_estimated.shape:
                 logger.error(f"Iter {iteration+1}: Búsqueda local devolvió shape incorrecto {shape_image_target.shape}. Abortando.");
                 final_shape_img = shape_image_estimated
                 final_pose = {'s': s, 'theta_rad': theta_rad, 'tx': tx, 'ty': ty}
                 return final_shape_img, b, final_pose

            # 4. Calcular la transformación de similitud implícita por los puntos objetivo
            #    Alineando la forma modelo (shape_model) a los puntos objetivo (shape_image_target)
            s_target, theta_target, tx_target, ty_target = self._get_similarity_transform_params(
                shape_model, shape_image_target
            )
            logger.debug(f"      Iter {iteration+1} Raw Pose Target (calculada): s={s_target:.4f}, th={np.degrees(theta_target):.2f}, t=({tx_target:.2f},{ty_target:.2f})")

            # 5. Actualizar los parámetros de pose (s, theta, tx, ty) aplicando amortiguamiento (damping)
            #    OJO: s NO se actualiza porque damping_factor_s = 0.0
            if damping_factor_s != 0.0: # Solo por si acaso se cambia el default
                delta_s = s_target - s
                s = s + damping_factor_s * delta_s
                logger.warning(f"      Iter {iteration+1} ESCALA ACTUALIZADA (damping={damping_factor_s}) a s={s:.4f}")


            # Calcular cambio en ángulo (manejando el ajuste de -pi a pi)
            delta_theta = theta_target - theta_rad
            delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta)) # Ajustar a [-pi, pi]
            theta_rad = theta_rad + damping_factor_theta * delta_theta
            theta_rad = np.arctan2(np.sin(theta_rad), np.cos(theta_rad)) # Normalizar ángulo final

            # Actualizar traslación
            tx = tx + damping_factor_t * (tx_target - tx)
            ty = ty + damping_factor_t * (ty_target - ty)

            logger.debug(f"      Iter {iteration+1} Pose Updated (DAMPED): s={s:.4f}(FIJA), th={np.degrees(theta_rad):.2f}, t=({tx:.2f},{ty:.2f})")

            # 6. Actualizar los parámetros de forma 'b'
            #    Proyectando los puntos objetivo (target) de vuelta al espacio del modelo.
            #    Primero, aplicar la inversa de la transformación de pose ACTUALIZADA (s, theta, tx, ty)
            #    a los puntos objetivo (shape_image_target) para llevarlos al espacio del modelo.

            # Calcular inversa de la rotación y escala
            theta_rad_inv = -theta_rad
            R_inv = np.array([[np.cos(theta_rad_inv), -np.sin(theta_rad_inv)],
                              [np.sin(theta_rad_inv),  np.cos(theta_rad_inv)]])
            inv_s = (1.0 / s) if abs(s) > 1e-6 else 1.0 # Usar escala actual (fija en 1.0 usualmente)

            # Aplicar transformación inversa: model = inv_s * R_inv * (target - t)
            translation_vec = np.array([tx, ty])
            shape_target_centered = shape_image_target - translation_vec
            # Aplicar rotación inversa y luego escala inversa
            shape_target_model_space = inv_s * (shape_target_centered @ R_inv.T)

            # Proyectar esta forma al espacio de parámetros 'b'
            b_new = self._project_shape_to_model(shape_target_model_space)
            # Limitar los nuevos parámetros b
            b = self._clamp_shape_params(b_new, n_std_devs=clamp_n_std_devs)
            logger.debug(f"      Iter {iteration+1} Updated b norm: {np.linalg.norm(b):.4f} (Params: {np.round(b, 3)})")

            # 7. Comprobar convergencia
            if shape_image_estimated_old is not None:
                 # Calcular el cambio medio en la posición de los puntos entre iteraciones
                 delta_points_norm = np.mean(np.linalg.norm(shape_image_estimated - shape_image_estimated_old, axis=1))
                 logger.info(f"Iter {iteration+1}: Avg point change={delta_points_norm:.4f}")
                 # Si el cambio es menor que la tolerancia, considerar convergido
                 if iteration > 0 and delta_points_norm < tolerance:
                     logger.info(f"CONVERGIDO en {iteration+1} iteraciones (delta < {tolerance}).")
                     converged = True
                     break # Salir del bucle for
            else:
                 logger.info(f"Iter {iteration+1}: (Delta omitido en primera iteración)")

            # Actualizar la forma "vieja" para la siguiente iteración
            shape_image_estimated_old = shape_image_estimated.copy()

        # --- Fin del Bucle Iterativo ---

        if not converged:
            logger.warning(f"NO CONVERGIÓ en {max_iters} iteraciones (delta >= {tolerance}).")

        # Generar la forma final en la imagen usando los últimos parámetros b y pose
        final_shape_model = self._generate_shape_instance(b)
        if final_shape_model is None:
             logger.error("Fallo al generar forma final desde b finales.")
             return None, None, None # Fallo

        final_shape_img = self._apply_similarity_transform(final_shape_model, s, theta_rad, tx, ty)
        if final_shape_img is None or np.isnan(final_shape_img).any():
             logger.error("Fallo al aplicar transform final o resultado contiene NaN/Inf.")
             return None, None, None # Fallo

        # Crear diccionario con la pose final
        final_pose = {'s': s, 'theta_rad': theta_rad, 'tx': tx, 'ty': ty}

        logger.info(f"Fin Fit. Pose final: s={s:.3f}(FIJA), th={np.degrees(theta_rad):.1f}deg, t=({tx:.1f},{ty:.1f})")
        logger.info(f"Params b finales: {np.round(b, 3)}")

        return final_shape_img, b, final_pose