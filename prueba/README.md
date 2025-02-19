# Módulo de Predicción PCA

Este módulo se encarga de la predicción de puntos anatómicos en imágenes de rayos X pulmonares utilizando modelos PCA pre-entrenados.

## Estructura del Proyecto

```
prueba/
├── requirements.txt
├── scripts/
│   └── predict_points.py
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

## Uso del Módulo

1. **Preparación**:
   - Asegurarse de que los modelos PCA estén disponibles en `../models/`
   - Verificar archivo de coordenadas JSON
   - Preparar imagen(es) para predicción

2. **Ejecutar Predicción**:
```bash
python scripts/predict_points.py --image path/to/image.png --output results/
```

3. **Parámetros de Predicción**:
   - `--image`: Ruta de la imagen a analizar
   - `--output`: Directorio para resultados
   - `--models`: Directorio de modelos (default: ../models/)
   - `--visualize`: Generar visualizaciones (default: True)

4. **Resultados**:
   - Coordenadas predichas en formato JSON
   - Visualizaciones de resultados
   - Logs de predicción

## Interpretación de Resultados

1. **Formato de Salida**:
   ```json
   {
     "coord1": {"x": 120, "y": 150, "error": 0.023},
     "coord2": {"x": 180, "y": 200, "error": 0.019}
   }
   ```

2. **Visualizaciones**:
   - Puntos predichos sobre la imagen
   - Regiones de búsqueda
   - Heatmaps de confianza

## Validación de Predicciones

1. **Métricas de Calidad**:
   - Error de predicción
   - Tiempo de procesamiento
   - Confianza de la predicción

2. **Verificación Visual**:
   - Revisar posición de puntos
   - Verificar alineación
   - Identificar anomalías

## Solución de Problemas

1. **Errores Comunes**:
   - Modelo no encontrado: Verificar rutas
   - Imagen incompatible: Revisar formato
   - Memoria insuficiente: Procesar por lotes

2. **Optimización**:
   - Ajustar parámetros de búsqueda
   - Optimizar preprocesamiento
   - Configurar cache de resultados

## Mantenimiento

1. **Actualización**:
   - Verificar compatibilidad con nuevos modelos
   - Mantener logs de predicciones
   - Respaldar configuraciones

2. **Monitoreo**:
   - Registrar tiempos de predicción
   - Analizar patrones de error
   - Documentar casos especiales

## Integración

Este módulo utiliza los modelos generados por el módulo de entrenamiento. Consideraciones:

1. Versiones compatibles de modelos
2. Rutas compartidas actualizadas
3. Formato de datos consistente

## Referencias

- Manual técnico: `docs/technical.md`
- Guía de diagnóstico: `docs/troubleshooting.md`
- FAQ: `docs/faq.md`

## Contacto

Para soporte técnico o reportes de problemas:
- Abrir ticket en el sistema de issues
- Incluir detalles de la predicción
- Adjuntar logs relevantes
