# Proyecto de Recorte de Imágenes Pulmonares

Este proyecto es una versión simplificada del sistema PulmoAlign, enfocada específicamente en el proceso de recorte de imágenes pulmonares.

## Estructura del Proyecto

```
recorte/
├── src/
│   ├── coordinates/        # Manejo de coordenadas
│   ├── image_processing/   # Procesamiento de imágenes
│   └── main.py            # Script principal
├── requirements.txt        # Dependencias del proyecto
└── README.md              # Esta documentación
```

## Requisitos

- Python 3.6 o superior
- Dependencias listadas en requirements.txt:
  - numpy
  - opencv-python
  - pandas
  - matplotlib
  - scikit-learn
  - Pillow

## Instalación

1. Clonar el repositorio
2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Archivos Necesarios

El proyecto requiere los siguientes archivos en el directorio padre:

1. `COVID-19_Radiography_Dataset/` - Directorio con las imágenes del dataset
2. `coordenadas.csv` - Archivo de coordenadas
3. `indices.csv` - Archivo de índices
4. `all_search_coordinates.json` - Archivo de coordenadas de búsqueda
5. `tools/template_analysis/template_analysis_results.json` - Datos pre-calculados de templates

## Uso

Para ejecutar el proceso de recorte:

```bash
cd src
python main.py
```

El script procesará todas las imágenes y generará los siguientes resultados:

1. Imágenes recortadas en:
   - `processed_images/cropped_images_Coord1/`
   - `processed_images/cropped_images_Coord2/`

2. Visualizaciones del proceso en:
   - `visualization_results/`

## Proceso

1. Carga de coordenadas y configuración
2. Procesamiento de cada imagen:
   - Mejora de contraste usando SAHS
   - Extracción de regiones usando templates
   - Alineación y recorte preciso
3. Generación de resultados y visualizaciones

## Notas

- Las dimensiones de las imágenes se mantienen en 64x64 píxeles
- Se procesan solo las coordenadas 1 y 2
- Se mantiene la precisión del algoritmo original
- Los resultados son compatibles con el sistema de entrenamiento
