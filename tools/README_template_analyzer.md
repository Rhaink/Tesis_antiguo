# Template Analyzer

## Descripción

Este script implementa un **sistema de análisis geométrico** para generar templates de recorte optimizados basados en regiones de búsqueda de puntos anatómicos. Su propósito es calcular las distancias y puntos de intersección necesarios para extraer regiones de interés de manera eficiente en algoritmos de template matching.

## Arquitectura

### Clase Principal: `TemplateAnalyzer`

Maneja todo el pipeline de análisis desde la carga de coordenadas hasta la generación de resultados.

**Atributos principales:**
- `search_coordinates`: Coordenadas de búsqueda cargadas desde JSON
- `template_data`: Resultados del análisis para cada coordenada
- `output_dir`: Directorio de salida para resultados y visualizaciones

## Funcionalidades Principales

### 1. `load_search_coordinates()`
Carga las coordenadas de búsqueda desde el archivo JSON generado por `extraccion_region_busqueda.py`.

### 2. `calculate_region_bounds(coord_points)`
Calcula los límites rectangulares de una región de búsqueda.

**Proceso:**
```python
left = min(x_coords)
right = max(x_coords)  
sup = min(y_coords)
inf = max(y_coords)
width = right - left + 1
height = inf - sup + 1
```

### 3. `calculate_template_distances(search_region, template_size=64)`
**Función crítica** que calcula las distancias a, b, c, d desde la región de búsqueda al template original de 64×64.

### 4. `create_cutting_template(a, b, c, d)`
Genera el template de recorte basado en las distancias calculadas.

### 5. `find_intersection_point(a, d)`
Determina el punto de intersección en coordenadas globales.

## Matemáticas Utilizadas

### 1. Cálculo de Límites de Región

Para una conjunto de puntos {(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)}:

```
left = min{xᵢ : i = 1,...,n}
right = max{xᵢ : i = 1,...,n}
sup = min{yᵢ : i = 1,...,n}
inf = max{yᵢ : i = 1,...,n}
```

Las dimensiones incluyen los extremos:
```
width = right - left + 1
height = inf - sup + 1
```

### 2. Cálculo de Distancias del Template

Dada una región de búsqueda R en un espacio 64×64, las distancias se calculan como:

```
a = min_y    (distancia desde borde superior)
b = 63 - max_x    (distancia desde borde derecho)  
c = 63 - max_y    (distancia desde borde inferior)
d = min_x    (distancia desde borde izquierdo)
```

**Restricciones de validez:**
```
a + c < 64    (distancias verticales)
b + d < 64    (distancias horizontales)
```

### 3. Generación del Template de Recorte

El template T es una matriz binaria donde:

```
T[y,x] = {
    1  si 0 ≤ y < (c + a) AND 0 ≤ x < (b + d)
    0  en otro caso
}
```

Las dimensiones del template son:
```
height_template = c + a
width_template = b + d
```

### 4. Punto de Intersección

El punto de intersección P en coordenadas globales se define como:
```
P = (d, a)
```

Donde:
- **d**: desplazamiento horizontal desde el origen
- **a**: desplazamiento vertical desde el origen

Este punto representa la esquina superior-izquierda del template dentro del espacio de 64×64.

### 5. Transformación de Coordenadas

**Coordenadas locales del template → Coordenadas globales:**
```
x_global = x_local + d
y_global = y_local + a
```

**Coordenadas globales → Coordenadas locales del template:**
```
x_local = x_global - d
y_local = y_global - a
```

## Estructura de Datos

### Entrada (JSON)
```json
{
  "coord1": [[y1, x1], [y2, x2], ...],
  "coord2": [[y1, x1], [y2, x2], ...],
  ...
}
```

### Salida (JSON)
```json
{
  "coord1": {
    "region_bounds": {
      "sup": int, "inf": int, "left": int, "right": int,
      "width": int, "height": int
    },
    "distances": {
      "a": int, "b": int, "c": int, "d": int
    },
    "template_bounds": {
      "min_x": int, "max_x": int, "min_y": int, "max_y": int,
      "width": int, "height": int
    },
    "intersection_point": {
      "x": int, "y": int
    }
  }
}
```

## Algoritmo Principal

```python
def analyze_templates():
    for coord_name, coord_points in search_coordinates.items():
        # 1. Calcular límites de región
        bounds = calculate_region_bounds(coord_points)
        
        # 2. Crear matriz binaria de región
        search_region = create_search_region(coord_points)
        
        # 3. Calcular distancias a, b, c, d
        a, b, c, d = calculate_template_distances(search_region)
        
        # 4. Generar template de recorte
        template = create_cutting_template(a, b, c, d)
        
        # 5. Encontrar punto de intersección
        intersection_point = find_intersection_point(a, d)
        
        # 6. Guardar resultados y generar visualización
        save_results()
        visualize_template()
```

## Ejecución

```bash
cd /path/to/Tesis
python tools/template_analyzer.py
```

### Requisitos Previos
1. Archivo `all_search_coordinates.json` generado por `extraccion_region_busqueda.py`
2. Estructura de directorios creada

## Archivos Generados

### Visualizaciones
- `template_analysis_coord{N}.png`: Visualización lado a lado de región de búsqueda y template
- Resolución: 300 DPI
- Incluye marcadores para origen y punto de intersección

### Datos
- `template_analysis_results.json`: Resultados completos del análisis
- Contiene límites, distancias, dimensiones y puntos de intersección para cada coordenada

## Aplicaciones

1. **Template Matching Optimizado**: Reduce el espacio de búsqueda mediante templates precisos
2. **Extracción de ROI**: Define regiones de interés para análisis posteriores  
3. **Alineación de Imágenes**: Proporciona puntos de referencia para transformaciones
4. **Validación Geométrica**: Verifica consistencia espacial de puntos anatómicos

## Ventajas del Enfoque

1. **Eficiencia Computacional**: Reduce template de 64×64 a regiones más pequeñas
2. **Precisión Geométrica**: Calcula posiciones exactas basadas en datos históricos
3. **Flexibilidad**: Adaptable a diferentes tipos de puntos anatómicos
4. **Trazabilidad**: Mantiene relación entre coordenadas locales y globales

## Limitaciones

1. **Dependencia de datos de entrenamiento**: Calidad del análisis depende de la completitud de los datos
2. **Asunción de distribución rectangular**: No considera formas complejas de distribución
3. **Tamaño fijo**: Diseñado específicamente para templates 64×64
4. **Sin validación estadística**: No considera variabilidad o outliers

## Integración con el Pipeline

1. **Entrada**: Coordenadas de `extraccion_region_busqueda.py`
2. **Procesamiento**: Análisis geométrico y generación de templates
3. **Salida**: Templates optimizados para algoritmos de detección
4. **Siguiente paso**: Uso en módulos de predicción y template matching

## Mejoras Futuras

1. **Templates adaptativos**: Ajustar tamaño según variabilidad de datos
2. **Análisis estadístico**: Incorporar confianza y distribuciones probabilísticas
3. **Optimización multi-escala**: Soporte para diferentes resoluciones
4. **Validación cruzada**: Métricas de rendimiento de los templates generados