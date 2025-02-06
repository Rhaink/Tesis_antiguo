# Proyecto de Predicción PCA

Este proyecto implementa el análisis PCA y predicción de puntos anatómicos en imágenes de rayos X pulmonares, basado en el proyecto original pulmo_align pero sin interfaz gráfica.

## Estructura del Proyecto

```
prediccion/
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── pca_analyzer.py      # Clase principal para análisis PCA
│   ├── coordinate_manager.py # Manejo de coordenadas
│   ├── image_processor.py   # Procesamiento de imágenes
│   ├── visualizer.py        # Visualización de resultados
│   └── utils/
│       ├── __init__.py
│       ├── image_utils.py   # Utilidades para imágenes
│       └── file_utils.py    # Utilidades para archivos
└── scripts/
    ├── train_models.py      # Script para entrenar modelos PCA
    └── predict_points.py    # Script para predicción de puntos
```

## Pasos de Implementación

1. **Configuración Inicial**
   - Crear estructura de directorios
   - Configurar requirements.txt con dependencias necesarias:
     * numpy
     * opencv-python
     * scikit-learn
     * matplotlib
     * pillow

2. **PCA Analyzer (src/pca_analyzer.py)**
   - Implementar la clase PCAAnalyzer del proyecto original
   - Mantener funcionalidades clave:
     * Entrenamiento del modelo
     * Cálculo de eigenfaces
     * Análisis de regiones de búsqueda
     * Cálculo de errores

3. **Coordinate Manager (src/coordinate_manager.py)**
   - Simplificar la gestión de coordenadas
   - Funciones para:
     * Cargar coordenadas de búsqueda
     * Obtener coordenadas específicas
     * Validar datos de coordenadas

4. **Image Processor (src/image_processor.py)**
   - Funciones para:
     * Cargar imágenes
     * Preprocesamiento
     * Redimensionamiento
     * Normalización

5. **Visualizer (src/visualizer.py)**
   - Funciones para:
     * Visualizar distribución de errores
     * Mostrar resultados de predicción
     * Generar gráficos de análisis

6. **Scripts de Ejecución**

   a) **train_models.py**:
   ```python
   # Proceso de entrenamiento:
   1. Cargar coordenadas de búsqueda
   2. Para cada punto anatómico:
      - Cargar imágenes de entrenamiento
      - Crear y entrenar modelo PCA
      - Guardar modelo entrenado
   ```

   b) **predict_points.py**:
   ```python
   # Proceso de predicción:
   1. Cargar modelos PCA entrenados
   2. Cargar imagen de prueba
   3. Para cada punto anatómico:
      - Analizar región de búsqueda
      - Encontrar mejor coincidencia
   4. Visualizar resultados
   ```

## Flujo de Trabajo

1. **Preparación**:
   ```bash
   # Crear entorno virtual
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # o
   .\venv\Scripts\activate  # Windows
   
   # Instalar dependencias
   pip install -r requirements.txt
   ```

2. **Entrenamiento**:
   ```bash
   # Entrenar modelos
   python scripts/train_models.py
   ```

3. **Predicción**:
   ```bash
   # Predecir puntos en nueva imagen
   python scripts/predict_points.py --image path/to/image.png
   ```

## Diferencias con el Proyecto Original

1. **Simplificación**:
   - Eliminación de la interfaz gráfica
   - Proceso automatizado por línea de comandos
   - Configuración mediante archivos

2. **Mejoras**:
   - Código más modular y reutilizable
   - Mejor manejo de errores
   - Documentación clara
   - Facilidad de uso en scripts

3. **Rendimiento**:
   - Procesamiento por lotes posible
   - Menos overhead por GUI
   - Mejor gestión de memoria

## Consideraciones Importantes

1. **Datos**:
   - Mantener estructura de datos consistente con proyecto original
   - Usar mismo formato de coordenadas
   - Conservar parámetros de PCA (varianza_threshold=0.95)

2. **Validación**:
   - Implementar validación de resultados
   - Comparar con resultados del proyecto original
   - Documentar diferencias si existen

3. **Optimización**:
   - Posibilidad de procesamiento paralelo
   - Cacheo de resultados intermedios
   - Mejor gestión de recursos

Este nuevo proyecto mantendrá la misma funcionalidad core del proyecto original pero con una implementación más limpia y fácil de usar en entornos automatizados.
