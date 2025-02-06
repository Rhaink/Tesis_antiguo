# Template Analyzer

Este programa analiza y calcula los templates de recorte y puntos de intersección para el sistema PulmoAlign.

## Funcionalidad

El programa:
1. Lee las coordenadas de búsqueda desde un archivo JSON
2. Calcula las distancias (a,b,c,d) para cada coordenada
3. Genera los templates de recorte
4. Determina los puntos de intersección
5. Genera visualizaciones del proceso
6. Guarda todos los resultados en un archivo JSON

## Estructura de Datos

El programa genera un archivo JSON con la siguiente estructura para cada coordenada:

```json
{
  "coord1": {
    "region_bounds": {
      "sup": int,      // Límite superior
      "inf": int,      // Límite inferior
      "left": int,     // Límite izquierdo
      "right": int,    // Límite derecho
      "width": int,    // Ancho de la región
      "height": int    // Alto de la región
    },
    "distances": {
      "a": int,        // Distancia al borde superior
      "b": int,        // Distancia al borde derecho
      "c": int,        // Distancia al borde inferior
      "d": int         // Distancia al borde izquierdo
    },
    "template_bounds": {
      "min_x": int,    // Límite izquierdo del template
      "max_x": int,    // Límite derecho del template
      "min_y": int,    // Límite superior del template
      "max_y": int     // Límite inferior del template
    },
    "intersection_point": {
      "x": int,        // Coordenada x del punto de intersección
      "y": int         // Coordenada y del punto de intersección
    }
  }
}
```

## Visualizaciones

Para cada coordenada, el programa genera una imagen que muestra:
- La región de búsqueda original
- El template de recorte resultante con el punto de intersección marcado

## Uso

1. Asegúrate de tener el archivo de coordenadas de búsqueda (`all_search_coordinates.json`) en el directorio Tesis.

2. Ejecuta el programa:
```bash
python template_analyzer.py
```

3. Los resultados se guardarán en:
- `template_analysis/template_analysis_results.json`: Datos calculados
- `template_analysis/template_analysis_coordX.png`: Visualizaciones

## Dependencias

- numpy
- matplotlib
- seaborn
- pathlib

## Notas

- El programa espera que las coordenadas estén en un sistema 64x64
- Los puntos de intersección se calculan en el sistema de coordenadas global
- Las visualizaciones muestran tanto la región de búsqueda como el template resultante
