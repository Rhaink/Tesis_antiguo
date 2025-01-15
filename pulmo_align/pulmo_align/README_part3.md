# PulmoAlign - Parte 3: Sistema de Visualización y Análisis Visual

## 1. Fundamentos de Visualización

### 1.1 Espacios de Color y Transformaciones

#### a) Conversión BGR a RGB
```python
RGB = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
```

#### b) Conversión a Escala de Grises
```math
Gray = 0.299R + 0.587G + 0.114B
```

### 1.2 Normalización de Visualización
```math
I_{normalized} = \frac{I - I_{min}}{I_{max} - I_{min}} × 255
```

## 2. Visualización Individual de Resultados

### 2.1 Visualización de Puntos Anatómicos

#### a) Representación de Puntos
- Coordenadas óptimas: (x*, y*)
- Región de búsqueda: S = {(x,y)}
- Rectángulo ROI: (width × height)

#### b) Codificación Visual
```python
# Puntos de búsqueda
scatter(search_coords[:, 1], search_coords[:, 0], 
        c='red', s=20, alpha=0.5)

# Punto óptimo
plot(min_x, min_y, 'g*', markersize=25)

# Región ROI
Rectangle((min_x - left, min_y - sup),
          width, height,
          fill=False, edgecolor='green')
```

### 2.2 Visualización de Errores

#### a) Distribución de Errores
```python
plt.hist(errors, bins=50, density=True)
```

#### b) Mapa de Calor de Error
```math
E_{map}(x,y) = ||ROI(x,y) - \hat{ROI}(x,y)||₂
```

### 2.3 Visualización de Caminos de Búsqueda

#### a) Trayectoria de Búsqueda
```python
plt.scatter(coords[:, 1], coords[:, 0], 
           c=errors, cmap='viridis')
```

#### b) Gradiente de Error
```math
∇E(x,y) = (\frac{∂E}{∂x}, \frac{∂E}{∂y})
```

## 3. Visualización Combinada

### 3.1 Integración de Múltiples Coordenadas

#### a) Superposición de Resultados
```python
for coord_name, result in results.items():
    min_x, min_y = result['min_error_coords']
    plt.plot(min_x, min_y, 'g*', 
            label=f'{coord_name} ({min_x}, {min_y})')
```

#### b) Visualización de Relaciones Espaciales
```python
distances = calculate_pairwise_distances(coordinates)
plt.hist(distances, bins='auto', density=True)
```

### 3.2 Análisis Visual Específico

#### a) Visualización Coord1-Coord2
```python
for coord_name in ['Coord1', 'Coord2']:
    if coord_name in results:
        min_x, min_y = results[coord_name]['min_error_coords']
        plt.plot(min_x, min_y, 'g*', markersize=15)
```

## 4. Análisis Estadístico Visual

### 4.1 Métricas de Visualización

#### a) Error Medio por Coordenada
```math
\bar{E}_{coord} = \frac{1}{n} \sum_{i=1}^n E_i
```

#### b) Distribución de Errores
```math
P(E) = \frac{1}{n} \sum_{i=1}^n δ(E - E_i)
```

### 4.2 Visualización de Calidad

#### a) Score de Reconstrucción
```math
Q_{recon} = 1 - \frac{E}{E_{max}}
```

#### b) Score de Estabilidad
```math
S_{score} = \frac{1}{1 + σ_E}
```

## 5. Implementación Técnica

### 5.1 Sistema de Coordenadas de Visualización

#### a) Transformación de Coordenadas
```python
def transform_coordinates(x, y, height):
    return x, height - y  # Inversión del eje Y
```

#### b) Escalado de Visualización
```python
display_width = 900
ratio = display_width / img.width
display_height = int(img.height * ratio)
```

### 5.2 Gestión de Memoria Visual

#### a) Liberación de Recursos
```python
plt.close('all')
gc.collect()
```

#### b) Optimización de Imágenes
```python
img = img.resize((display_width, display_height), 
                 Image.Resampling.LANCZOS)
```

## 6. Interfaz Gráfica

### 6.1 Estructura de la Interfaz

#### a) Panel Principal
```python
main_frame = ttk.PanedWindow(orient='vertical')
```

#### b) Paneles de Visualización
```python
image_frame = ttk.Frame()
control_frame = ttk.Frame()
log_frame = ttk.LabelFrame()
```

### 6.2 Navegación de Resultados

#### a) Control de Navegación
```python
def next_image(self):
    self.current_index = (self.current_index + 1) % len(self.images)
```

#### b) Actualización de Visualización
```python
def update_display(self):
    self.show_current_image()
    self.update_counter()
```

## 7. Consideraciones Prácticas

### 7.1 Optimización de Rendimiento

#### a) Caché de Imágenes
```python
self.photo_references = []  # Mantener referencias
```

#### b) Actualización Eficiente
```python
self.root.update()  # Actualización de GUI
```

### 7.2 Control de Calidad Visual

#### a) Validación de Visualización
- Verificación de dimensiones
- Control de aspectos
- Validación de colores

#### b) Gestión de Errores
- Manejo de excepciones
- Feedback visual
- Logging de errores

Este documento representa la tercera parte de la documentación técnica del sistema PulmoAlign, enfocándose en los aspectos técnicos y matemáticos del sistema de visualización y análisis visual de resultados.
