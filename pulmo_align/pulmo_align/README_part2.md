# PulmoAlign - Parte 2: Análisis de Componentes Principales (PCA)

## 1. Fundamentos Matemáticos del PCA

### 1.1 Espacio de Características
Cada imagen ROI de dimensión w×h se representa como un vector en un espacio de alta dimensión:
```math
x ∈ ℝ^{w×h}
```

### 1.2 Matriz de Datos
Las imágenes de entrenamiento se organizan en una matriz X:
```math
X = [x₁, x₂, ..., xₙ]ᵀ ∈ ℝ^{n×d}
```
donde:
- n: número de imágenes
- d: dimensionalidad (w×h)

### 1.3 Centrado de Datos
Se calcula y resta la media (mean face):
```math
μ = \frac{1}{n} \sum_{i=1}^n xᵢ
X_{centered} = X - 1μᵀ
```

### 1.4 Matriz de Covarianza
```math
C = \frac{1}{n} X_{centered}ᵀ X_{centered} ∈ ℝ^{d×d}
```

## 2. Descomposición PCA

### 2.1 Problema de Eigenvalores
Resolver la ecuación:
```math
Cv = λv
```
donde:
- λ: eigenvalores
- v: eigenvectores (eigenfaces)

### 2.2 Selección de Componentes
El número de componentes k se determina por la varianza explicada:
```math
k = argmin_{m} \{\sum_{i=1}^m λᵢ / \sum_{i=1}^n λᵢ ≥ threshold\}
```

### 2.3 Base del Subespacio
Los k eigenvectores principales forman la matriz de proyección:
```math
V_k = [v₁, v₂, ..., vₖ] ∈ ℝ^{d×k}
```

## 3. Proyección y Reconstrucción

### 3.1 Proyección en el Subespacio PCA
Para una imagen x:
```math
ω = V_k^T(x - μ) ∈ ℝ^k
```

### 3.2 Reconstrucción
```math
\hat{x} = μ + V_kω ∈ ℝ^d
```

### 3.3 Error de Reconstrucción
```math
E = ||x - \hat{x}||₂
```

## 4. Análisis de Regiones de Búsqueda

### 4.1 Proceso de Búsqueda
Para cada coordenada candidata (x, y):

1. Extracción de ROI:
```math
ROI(x,y) = I[y-h/2:y+h/2, x-w/2:x+w/2]
```

2. Normalización:
```math
ROI_{norm} = \frac{ROI}{255}
```

3. Cálculo de error:
```math
E(x,y) = ||ROI_{norm} - \hat{ROI}_{norm}||₂
```

### 4.2 Optimización
```math
(x*, y*) = argmin_{(x,y)∈S} E(x,y)
```
donde S es el conjunto de coordenadas de búsqueda.

## 5. Interpretación Geométrica

### 5.1 Espacio de Características
- Dimensión original: ℝ^{w×h}
- Dimensión reducida: ℝ^k
- Relación: k << w×h

### 5.2 Transformaciones Geométricas

#### a) Proyección PCA
```math
T: ℝ^{w×h} → ℝ^k
T(x) = V_k^T(x - μ)
```

#### b) Reconstrucción
```math
R: ℝ^k → ℝ^{w×h}
R(ω) = μ + V_kω
```

### 5.3 Propiedades Geométricas

#### a) Ortogonalidad
Los eigenvectores forman una base ortonormal:
```math
v_i^T v_j = \begin{cases} 1 & \text{si } i=j \\ 0 & \text{si } i≠j \end{cases}
```

#### b) Minimización de Error
El subespacio PCA minimiza el error de reconstrucción:
```math
V_k = argmin_V \sum_{i=1}^n ||x_i - VV^T(x_i - μ) - μ||₂²
```

## 6. Análisis Estadístico

### 6.1 Varianza Explicada
Para k componentes:
```math
Var_{explained}(k) = \frac{\sum_{i=1}^k λᵢ}{\sum_{i=1}^n λᵢ}
```

### 6.2 Error Medio
```math
E_{mean} = \frac{1}{n} \sum_{i=1}^n ||x_i - \hat{x}_i||₂
```

### 6.3 Desviación Estándar del Error
```math
σ_E = \sqrt{\frac{1}{n} \sum_{i=1}^n (||x_i - \hat{x}_i||₂ - E_{mean})²}
```

## 7. Implementación Práctica

### 7.1 Optimizaciones Computacionales

#### a) Precálculo de Proyecciones
```python
V_k_precomputed = pca.components_[:k].T
mean_precomputed = pca.mean_
```

#### b) Vectorización
```python
errors = np.linalg.norm(X - X_reconstructed, axis=1)
```

### 7.2 Gestión de Memoria
- Procesamiento por lotes
- Liberación de recursos
- Optimización de cálculos matriciales

Este documento representa la segunda parte de la documentación técnica del sistema PulmoAlign, enfocándose en los fundamentos matemáticos y geométricos del análisis PCA y su aplicación en el procesamiento de imágenes médicas.
