# Extracción de Regiones de Búsqueda

## Descripción

Este script analiza las coordenadas de puntos anatómicos en imágenes médicas para generar **regiones de búsqueda optimizadas**. Su propósito es identificar las áreas más probables donde pueden encontrarse puntos anatómicos específicos, reduciendo el espacio de búsqueda en algoritmos de detección.

## Funcionalidad Principal

### `generate_search_zone(csv_file, coord_num)`

Genera una región de búsqueda rectangular para un punto anatómico específico basándose en la distribución histórica de coordenadas.

**Parámetros:**
- `csv_file`: Archivo CSV con coordenadas de entrenamiento
- `coord_num`: Número del punto anatómico (1-15)

**Proceso:**
1. **Extracción de datos**: Lee coordenadas desde CSV
2. **Normalización**: Limita coordenadas al rango [0, 63]
3. **Creación de heatmap**: Mapea frecuencia de aparición
4. **Cálculo de bounding box**: Encuentra región mínima que contiene todos los puntos
5. **Generación de zona**: Crea región rectangular de búsqueda
6. **Visualización**: Genera gráfico de la zona
7. **Exportación**: Guarda coordenadas y visualización

## Matemáticas Utilizadas

### 1. Mapeo de Coordenadas

Para el punto anatómico `n`, las coordenadas se extraen usando:
```
x_column = (n - 1) × 2 + 1
y_column = x_column + 1
```

### 2. Normalización por Clipping

Se aplica la función de clipping para mantener coordenadas en el rango válido:
```
coord_normalized = clip(coord_original, 0, 63)
```

Donde:
```
clip(x, min, max) = {
    min    si x < min
    x      si min ≤ x ≤ max
    max    si x > max
}
```

### 3. Creación de Heatmap

Se crea una matriz H de 64×64 donde cada elemento H[y,x] representa la frecuencia:
```
H[y,x] = Σ δ(xi - x, yi - y)
```

Donde δ es la función delta de Kronecker:
```
δ(a,b) = {
    1  si a = b
    0  si a ≠ b
}
```

### 4. Cálculo del Bounding Box

Se encuentra el rectángulo mínimo que contiene todos los puntos no-cero:
```
min_x = min{x : ∃y tal que H[y,x] > 0}
max_x = max{x : ∃y tal que H[y,x] > 0}
min_y = min{y : ∃x tal que H[y,x] > 0}
max_y = max{y : ∃x tal que H[y,x] > 0}
```

### 5. Región de Búsqueda

La zona de búsqueda S es una matriz binaria:
```
S[y,x] = {
    1  si min_y ≤ y ≤ max_y AND min_x ≤ x ≤ max_x
    0  en otro caso
}
```

## Estructura de Datos

### Entrada (CSV)
```
imagen_id, x1, y1, x2, y2, ..., x15, y15
```

### Salida (JSON)
```json
{
  "coord1": [(y1, x1), (y2, x2), ...],
  "coord2": [(y1, x1), (y2, x2), ...],
  ...
  "coord15": [(y1, x1), (y2, x2), ...]
}
```

## Ejecución

```bash
cd /path/to/Tesis
python tools/extraccion_region_busqueda.py
```

## Requisitos

- Python 3.6+
- numpy
- pandas
- matplotlib
- seaborn

## Archivos Generados

### Visualizaciones
- `resultados/region_busqueda/dataset_entrenamiento_1/imagenes/coord{N}_search_zone.png`
- Heatmaps de 64×64 píxeles con cuadrícula cada 8 píxeles
- Resolución: 600 DPI

### Datos
- `resultados/region_busqueda/dataset_entrenamiento_1/json/all_search_coordinates.json`
- Coordenadas (y,x) de todos los píxeles en cada zona de búsqueda

## Aplicaciones

1. **Optimización de búsqueda**: Reduce el espacio de búsqueda en algoritmos de detección
2. **Análisis de variabilidad**: Evalúa la consistencia de puntos anatómicos
3. **Validación de datos**: Identifica outliers en coordenadas
4. **Preprocesamiento**: Genera masks para algoritmos de template matching

## Limitaciones

- Asume distribución rectangular de puntos
- No considera correlaciones entre puntos anatómicos
- Sensible a outliers en los datos de entrenamiento
- Región fija independiente del tipo de imagen

## Mejoras Futuras

- Implementar regiones elípticas basadas en distribución gaussiana
- Considerar correlaciones espaciales entre puntos
- Aplicar filtros de outliers antes del cálculo
- Adaptar regiones según características de la imagen