# PulmoAlign - Parte 1: Fundamentos y Procesamiento de Imágenes

## 1. Introducción al Sistema

PulmoAlign es un sistema avanzado para el análisis automático de radiografías pulmonares que implementa técnicas sofisticadas de procesamiento de imágenes y análisis estadístico. El sistema está diseñado para detectar y alinear automáticamente 15 puntos anatómicos clave en radiografías torácicas.

### 1.1 Objetivos del Sistema
- Detección automática de puntos anatómicos clave
- Alineación precisa de estructuras pulmonares
- Análisis estadístico de variabilidad anatómica
- Validación geométrica de resultados

## 2. Sistema de Coordenadas Anatómicas

### 2.1 Estructura de Puntos Anatómicos
El sistema utiliza 15 puntos anatómicos clave, cada uno definido por un conjunto específico de parámetros:

```python
{
    "Coord1": {"sup": 0, "inf": 47, "left": 22, "right": 25, "width": 47, "height": 47},
    "Coord2": {"sup": 38, "inf": 1, "left": 26, "right": 25, "width": 51, "height": 39},
    ...
    "Coord15": {"sup": 38, "inf": 0, "left": 44, "right": 0, "width": 44, "height": 38}
}
```

### 2.2 Cálculos Geométricos Fundamentales

#### a) Cálculo del Centro
Para cada punto anatómico, el centro se calcula mediante:

```math
center_x = \frac{left + right}{2}
center_y = \frac{sup + inf}{2}
```

#### b) Punto de Intersección
El punto de intersección se define como:

```math
intersection_x = left
intersection_y = sup
```

#### c) Ajuste de Desplazamiento
El ajuste de posición se realiza mediante:

```math
dx = new_x - intersection_x
dy = new_y - intersection_y
adjusted_x = center_x + dx
adjusted_y = center_y + dy
```

## 3. Procesamiento de Imágenes

### 3.1 Algoritmo SAHS (Statistical Asymmetric Histogram Stretching)

El algoritmo SAHS implementa un método adaptativo para mejorar el contraste en radiografías:

#### a) Análisis Estadístico del Histograma
1. Cálculo de la media global:
```math
μ = \frac{1}{N} \sum_{i=1}^{N} I(i)
```

2. Separación de grupos:
```python
G₁ = {I(i) | I(i) > μ}  # Grupo superior
G₂ = {I(i) | I(i) ≤ μ}  # Grupo inferior
```

3. Cálculo de desviaciones estándar asimétricas:
```math
σ₁ = \sqrt{\frac{1}{|G₁|} \sum_{i∈G₁} (I(i) - μ)²}  # Grupo superior
σ₂ = \sqrt{\frac{1}{|G₂|} \sum_{i∈G₂} (I(i) - μ)²}  # Grupo inferior
```

#### b) Determinación de Límites Adaptativos
```math
max\_value = μ + 2.5σ₁
min\_value = μ - 2.0σ₂
```

#### c) Normalización Final
```math
I_{enhanced} = 255 \cdot \frac{I - min\_value}{max\_value - min\_value}
```

### 3.2 Extracción de Regiones de Interés (ROI)

#### a) Cálculo de Coordenadas ROI
```python
start_x = center_x - width//2
start_y = center_y - height//2
end_x = start_x + width
end_y = start_y + height
```

#### b) Validación de Límites
```python
valid_start_x = max(0, start_x)
valid_start_y = max(0, start_y)
valid_end_x = min(image_width, end_x)
valid_end_y = min(image_height, end_y)
```

#### c) Extracción y Normalización
```python
roi = image[valid_start_y:valid_end_y, valid_start_x:valid_end_x]
roi_normalized = roi.astype(float) / 255.0
```

## 4. Interpretación Geométrica

### 4.1 Espacio de Coordenadas
El sistema opera en un espacio de coordenadas 2D discreto donde:
- El origen (0,0) está en la esquina superior izquierda
- El eje X aumenta hacia la derecha
- El eje Y aumenta hacia abajo
- Las coordenadas son enteros no negativos

### 4.2 Transformaciones Geométricas

#### a) Traslación de ROI
La traslación de una ROI se define mediante el vector de desplazamiento:
```math
\vec{d} = (dx, dy) = (new_x - intersection_x, new_y - intersection_y)
```

#### b) Región de Búsqueda
Para cada punto anatómico, la región de búsqueda forma un subespacio discreto:
```math
S = \{(x,y) ∈ ℤ² | x_{min} ≤ x ≤ x_{max}, y_{min} ≤ y ≤ y_{max}\}
```

## 5. Validación y Control de Calidad

### 5.1 Verificaciones Geométricas
- Validación de límites de imagen
- Verificación de dimensiones de ROI
- Control de coordenadas válidas

### 5.2 Control de Calidad de Imagen
- Verificación de contraste
- Validación de normalización
- Control de valores atípicos

## 6. Consideraciones de Implementación

### 6.1 Eficiencia Computacional
- Uso de operaciones vectorizadas
- Optimización de cálculos estadísticos
- Gestión eficiente de memoria

### 6.2 Robustez
- Manejo de casos límite
- Validación de parámetros
- Control de errores

Este documento representa la primera parte de la documentación técnica del sistema PulmoAlign, enfocándose en los fundamentos matemáticos y geométricos del procesamiento de imágenes y la gestión de coordenadas anatómicas.
