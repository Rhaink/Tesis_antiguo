# Módulo de Entrenamiento PCA

Este módulo se encarga del entrenamiento de modelos PCA para la detección de puntos anatómicos en imágenes de rayos X pulmonares.

## Estructura del Proyecto

```
entrenamiento/
├── requirements.txt
├── scripts/
│   └── train_models.py
└── src/
    ├── __init__.py
    ├── pca_analyzer.py
    ├── coordinate_manager.py
    ├── image_processor.py
    ├── template_processor.py
    ├── visualizer.py
    └── utils/
        ├── __init__.py
        ├── file_utils.py
        └── image_utils.py
```

## Requisitos

1. Crear un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
.\venv\Scripts\activate  # Windows
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Proceso de Entrenamiento

1. **Preparación de Datos**:
   - Las imágenes de entrenamiento deben estar en el directorio especificado
   - El archivo de coordenadas JSON debe estar disponible
   - Verificar estructura de directorios para salida de modelos

2. **Ejecutar Entrenamiento**:
```bash
python scripts/train_models.py --config config.yaml
```

3. **Parámetros de Entrenamiento**:
   - `variance_threshold`: Umbral de varianza explicada (default: 0.95)
   - `template_size`: Tamaño del template (default: 64x64)
   - `enhance_contrast`: Aplicar mejora de contraste (default: True)

4. **Resultados**:
   - Los modelos entrenados se guardan en `../models/`
   - Se generan archivos de log en `logs/`
   - Las visualizaciones se guardan en `visualization_results/`

## Validación de Resultados

1. **Métricas de Entrenamiento**:
   - Varianza explicada por componentes principales
   - Error de reconstrucción promedio
   - Tiempo de entrenamiento por modelo

2. **Visualizaciones**:
   - Eigenfaces generados
   - Distribución de errores
   - Reconstrucciones de ejemplo

## Mantenimiento

1. **Actualización de Modelos**:
   - Frecuencia recomendada: cuando haya nuevos datos significativos
   - Mantener respaldo de modelos anteriores
   - Documentar cambios y mejoras

2. **Control de Calidad**:
   - Ejecutar pruebas unitarias: `pytest tests/`
   - Validar resultados con conjunto de prueba
   - Verificar consistencia con modelos anteriores

## Solución de Problemas

1. **Errores Comunes**:
   - Memoria insuficiente: Reducir batch size
   - GPU no detectada: Verificar CUDA
   - Datos corruptos: Validar imágenes

2. **Optimización**:
   - Ajustar parámetros de PCA
   - Modificar preprocesamiento
   - Balancear conjunto de entrenamiento

## Referencias

- Documentación técnica completa: `docs/`
- Explicación matemática: `explicacion_matematica.md`
- Guías de contribución: `CONTRIBUTING.md`

## Integración

Este módulo genera los modelos que serán utilizados por el módulo de prueba. Asegurarse de:

1. Mantener compatibilidad de versiones
2. Documentar cambios en formato de modelos
3. Actualizar rutas compartidas según sea necesario

## Contacto

Para reportar problemas o sugerir mejoras:
- Crear issue en el repositorio
- Documentar claramente el problema
- Incluir logs relevantes
