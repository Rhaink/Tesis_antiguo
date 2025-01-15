# PulmoAlign - Parte 4: Sistema de Búsqueda y Optimización

## 1. Fundamentos de Búsqueda

### 1.1 Espacio de Búsqueda

#### a) Definición Matemática
El espacio de búsqueda S para cada punto anatómico se define como:
```math
S = {(x,y) ∈ ℤ² | x_min ≤ x ≤ x_max, y_min ≤ y ≤ y_max}
```

#### b) Discretización
Para cada coordenada anatómica:
```python
# Ejemplo para Coord1
search_region = {
    "x_range": range(1, 17),  # 17 puntos
    "y_range": range(23, 39)  # 17 puntos
}
```

### 1.2 Función Objetivo

#### a) Error de Reconstrucción
```math
E(x,y) = ||ROI(x,y) - \hat{ROI}(x,y)||₂
```

#### b) Normalización del Error
```math
E_{norm}(x,y) = \frac{E(x,y)}{max_{(x',y')∈S} E(x',y')}
```

## 2. Algoritmo de Búsqueda

### 2.1 Búsqueda Exhaustiva

#### a) Algoritmo Base
```python
def search_optimal_point(image, search_coordinates):
    min_error = float('inf')
    min_error_coords = None
    errors = []
    
    for coord in search_coordinates:
        error = calculate_reconstruction_error(image, coord)
        errors.append(error)
        
        if error < min_error:
            min_error = error
            min_error_coords = coord
            
    return min_error, min_error_coords, errors
```

#### b) Complejidad Computacional
```math
O(|S| × (w × h))
```
donde:
- |S|: tamaño del espacio de búsqueda
- w × h: dimensiones de la ROI

### 2.2 Optimización de Búsqueda

#### a) Paralelización
```python
def parallel_search(image, search_coordinates, n_processes):
    with Pool(n_processes) as pool:
        results = pool.map(partial(process_coordinate, image), 
                         search_coordinates)
    return min(results, key=lambda x: x[0])
```

#### b) Búsqueda por Lotes
```python
def batch_search(image, search_coordinates, batch_size):
    for i in range(0, len(search_coordinates), batch_size):
        batch = search_coordinates[i:i+batch_size]
        process_batch(image, batch)
```

## 3. Análisis de Rendimiento

### 3.1 Métricas de Rendimiento

#### a) Error Medio Global
```math
\bar{E} = \frac{1}{|S|} \sum_{(x,y)∈S} E(x,y)
```

#### b) Desviación Estándar del Error
```math
σ_E = \sqrt{\frac{1}{|S|} \sum_{(x,y)∈S} (E(x,y) - \bar{E})²}
```

### 3.2 Análisis de Convergencia

#### a) Tasa de Convergencia
```math
r_k = \frac{||E_{k+1} - E^*||}{||E_k - E^*||}
```

#### b) Criterio de Parada
```math
||E_k - E_{k-1}|| < ε
```

## 4. Optimización de Parámetros

### 4.1 Parámetros del Sistema

#### a) Dimensiones de ROI
```python
roi_config = {
    "Coord1": {"width": 47, "height": 47},
    "Coord2": {"width": 51, "height": 39}
}
```

#### b) Umbrales de Error
```python
thresholds = {
    "reconstruction_error": 0.1,
    "variance_explained": 0.95,
    "convergence": 1e-6
}
```

### 4.2 Ajuste Adaptativo

#### a) Ajuste de Región de Búsqueda
```math
S_{k+1} = {(x,y) | ||x-x^*_k||₂ ≤ r_k, (x,y) ∈ S_k}
```

#### b) Actualización de Parámetros
```math
θ_{k+1} = θ_k - α∇E(θ_k)
```

## 5. Validación y Control de Calidad

### 5.1 Validación de Resultados

#### a) Validación Geométrica
```python
def validate_coordinates(coords, anatomical_constraints):
    distances = calculate_pairwise_distances(coords)
    return check_anatomical_constraints(distances)
```

#### b) Análisis Estadístico
```python
def analyze_error_distribution(errors):
    z_scores = (errors - np.mean(errors)) / np.std(errors)
    return np.abs(z_scores) < 3  # Identificar valores atípicos
```

### 5.2 Control de Calidad

#### a) Métricas de Calidad
```math
Q_{score} = \frac{1}{1 + E_{norm}} × \frac{1}{1 + σ_E}
```

#### b) Validación Cruzada
```python
def cross_validate(model, data, k_folds):
    scores = []
    for train, test in k_fold_split(data, k_folds):
        score = validate_model(model, train, test)
        scores.append(score)
    return np.mean(scores), np.std(scores)
```

Este documento representa la cuarta parte de la documentación técnica del sistema PulmoAlign, enfocándose en los aspectos matemáticos y algorítmicos del sistema de búsqueda y optimización.
