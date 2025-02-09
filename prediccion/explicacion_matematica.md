# Análisis Matemático y Geométrico del Sistema de Predicción de Landmarks

## Índice
1. [Fundamentos Matemáticos y Geométricos](#1-fundamentos-matemáticos-y-geométricos)
2. [Procesamiento de Imágenes](#2-procesamiento-de-imágenes)
3. [Sistema de Coordenadas y Búsqueda](#3-sistema-de-coordenadas-y-búsqueda)
4. [Visualización Matemática](#4-visualización-matemática)
5. [Proceso de Predicción](#5-proceso-de-predicción)
6. [Optimización y Mejoras Sugeridas](#6-optimización-y-mejoras-sugeridas)

## 1. Fundamentos Matemáticos y Geométricos

### 1.1 Análisis de Componentes Principales (PCA)
*Localización en el código: `src/pca_analyzer.py`*

El PCA es una técnica fundamental en este proyecto que permite reducir la dimensionalidad de los datos mientras mantiene la información más relevante. Su implementación incluye varios pasos matemáticos clave:

#### 1.1.1 Transformación de Datos
```python
X = np.array([img.astype(float) for img in images])
X = X.reshape(len(images), -1)  # Matriz de datos NxD
```

La matriz X representa cada imagen como un vector fila, donde:
- N = número de imágenes
- D = dimensiones del template (ancho × alto)

#### 1.1.2 Proceso Matemático
1. **Centrado de Datos**:
   ```
   X_centered = X - μ
   donde μ = media de cada característica
   ```

2. **Matriz de Covarianza**:
   ```
   Σ = (X_centered.T @ X_centered) / (n_samples - 1)
   ```

3. **Descomposición en Valores Propios**:
   ```
   Σv = λv
   donde:
   λ = valores propios
   v = vectores propios (eigenfaces)
   ```

4. **Selección de Componentes**:
   ```python
   cumulative_variance_ratio = np.cumsum(explained_variance_ratio_)
   n_components = np.argmax(cumulative_variance_ratio >= threshold) + 1
   ```

### 1.2 Geometría de Templates
*Localización en el código: `src/template_processor.py`*

#### 1.2.1 Transformaciones Geométricas

1. **Traslación de Templates**:
   ```python
   template_start_x = search_x - intersection_x
   template_start_y = search_y - intersection_y
   ```
   Esta operación mueve el template manteniendo la relación geométrica entre el punto de intersección y los bordes del template.

2. **Validación Geométrica**:
   ```python
   if (template_start_x < 0 or 
       template_start_y < 0 or 
       template_start_x + template_width > image.shape[1] or 
       template_start_y + template_height > image.shape[0]):
       raise ValueError("Template fuera de límites")
   ```

## 2. Procesamiento de Imágenes
*Localización en el código: `src/image_processor.py` y `src/contrast_enhancer.py`*

### 2.1 Mejora de Contraste SAHS (Statistically Adaptive Histogram Stretching)

#### Formulación Matemática:
```python
gray_mean = np.mean(gray_image)
std_above = np.sqrt(np.mean((above_mean - gray_mean) ** 2))
std_below = np.sqrt(np.mean((below_or_equal_mean - gray_mean) ** 2))

max_value = gray_mean + 2.5 * std_above
min_value = gray_mean - 2.0 * std_below
```

El algoritmo SAHS ajusta el contraste de forma asimétrica considerando la distribución estadística de los niveles de gris por encima y por debajo de la media.

### 2.2 Normalización de Imágenes
```python
normalized_image = image.astype(float) / 255.0
```

## 3. Sistema de Coordenadas y Búsqueda
*Localización en el código: `src/coordinate_manager.py`*

### 3.1 Espacio de Búsqueda
El sistema define un espacio de búsqueda discreto mediante:

```python
for x in range(region_bounds['left'], region_bounds['right'] + 1):
    for y in range(region_bounds['sup'], region_bounds['inf'] + 1):
        search_coords.append((y, x))
```

### 3.2 Estructuras Geométricas

1. **Template**:
   ```python
   template_bounds = {
       'min_x': int,  # Límite izquierdo
       'max_x': int,  # Límite derecho
       'min_y': int,  # Límite superior
       'max_y': int,  # Límite inferior
       'width': int,  # Ancho
       'height': int  # Alto
   }
   ```

2. **Región de Búsqueda**:
   ```python
   region_bounds = {
       'sup': int,    # Límite superior
       'inf': int,    # Límite inferior
       'left': int,   # Límite izquierdo
       'right': int,  # Límite derecho
       'width': int,  # Ancho
       'height': int  # Alto
   }
   ```

## 4. Visualización Matemática
*Localización en el código: `src/visualizer.py`*

### 4.1 Análisis de Errores
```python
plt.hist(errors, bins=50, alpha=0.75)
```
Visualiza la distribución estadística de los errores de reconstrucción.

### 4.2 Representación del Espacio de Búsqueda
```python
coords = np.array(search_coordinates)
plt.scatter(coords[:, 0], coords[:, 1])
```

## 5. Proceso de Predicción
*Localización en el código: `src/predict_points.py`*

### 5.1 Algoritmo de Optimización

1. **Función Objetivo**:
   ```python
   error = np.linalg.norm(original.flatten() - reconstructed.flatten())
   ```
   Minimiza la norma L2 entre la imagen original y su reconstrucción PCA.

2. **Búsqueda Exhaustiva**:
   ```python
   min_error = float('inf')
   for coord in search_coordinates:
       error = model.analyze_search_region(...)
       if error < min_error:
           min_error = error
           min_error_coords = coord
   ```

## 6. Optimización y Mejoras Sugeridas

### 6.1 Optimización Matemática
1. **Búsqueda Gradiente**:
   - Implementar descenso por gradiente para optimización local
   - Usar métodos de búsqueda global (e.g., simulated annealing)

2. **Submuestreo Inteligente**:
   - Implementar estrategias de muestreo adaptativo
   - Utilizar técnicas de aprendizaje activo

### 6.2 Mejoras Geométricas
1. **Transformaciones Adicionales**:
   - Rotación del template
   - Escalado adaptativo
   - Deformaciones no rígidas

2. **Optimización de Regiones**:
   - Análisis de covarianza para ajustar regiones de búsqueda
   - Implementación de templates deformables

### 6.3 Validación Estadística
1. **Métricas de Calidad**:
   - Intervalos de confianza para predicciones
   - Tests de hipótesis para validación
   - Análisis de robustez

2. **Mejoras de Precisión**:
   - Cross-validation para selección de parámetros
   - Ensemble de modelos PCA
   - Regularización adaptativa
