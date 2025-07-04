# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical image analysis project for COVID-19 detection using chest X-ray images. The system uses PCA/Kernel PCA for anatomical landmark detection and template matching for precise region alignment. The project processes images from the COVID-19 Radiography Dataset to train models for automated detection and classification.

## Architecture

The project is organized into specialized modules:

### Core Modules
- **`entrenamiento/`**: PCA/Kernel PCA model training system
- **`etiquetado/`**: GUI-based image labeling tool
- **`prediccion/`**: Model prediction and testing
- **`recorte/`**: Image cropping and preprocessing
- **`tools/`**: Analysis utilities and template processing

### Data Flow
1. Raw images processed through `etiquetado/` for coordinate labeling
2. `tools/template_analyzer.py` generates search regions and templates
3. `entrenamiento/` trains PCA models using labeled coordinates
4. `prediccion/` uses trained models for anatomical landmark detection
5. `recorte/` performs final image cropping and alignment

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training Models
```bash
# Train PCA models
cd entrenamiento
python scripts/train_models.py

# Train Kernel PCA models
python scripts/train_models.py --use_kernel --kernel rbf --gamma 0.1
```

### Template Analysis
```bash
# Generate template analysis
python tools/template_analyzer.py
```

### Image Labeling
```bash
# Launch GUI labeling tool
cd etiquetado
python etiquetar_imagenes.py
# or
./ejecutar_etiquetador.sh
```

### Testing and Prediction
```bash
# Run coordinate predictions
cd prediccion
python prediccion_coordenadas.py

# Generate animated predictions
python prediccion_coordenadas_animado.py
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .

# Run tests
pytest tests/
```

## Key Components

### Image Processing Pipeline
- **SAHS contrast enhancement**: Implemented in `contrast_enhancer.py`
- **Template matching**: Core algorithm in `template_processor.py`
- **PCA analysis**: Statistical modeling in `pca_analyzer.py`
- **Coordinate management**: Spatial data handling in `coordinate_manager.py`

### Data Structures
- **Coordinates**: Stored as JSON files with [y,x] format (0-based indexing)
- **Models**: Serialized PCA models saved as .pkl files
- **Templates**: 64x64 binary matrices for region matching
- **Images**: Processed as 64x64 pixel regions

### Configuration
- **Base paths**: Hardcoded to `/home/donrobot/Projects/Tesis`
- **Dataset**: COVID-19_Radiography_Dataset with COVID, Normal, and Viral Pneumonia classes
- **Template size**: Fixed at 64x64 pixels
- **Coordinate range**: 0-63 for both x and y axes

## File Structure Conventions

### Results Organization
```
resultados/
├── analisis_regiones/
├── entrenamiento/
├── prediccion/
├── recorte/
└── region_busqueda/
```

### Model Storage
- Models saved in `resultados/entrenamiento/dataset_entrenamiento_1/models/`
- Naming convention: `{coord_name}[_kernel]_model.pkl`
- Visualization results in corresponding `visualization_results/` directories

### Dataset Structure
- Training images in `dataset/dataset_prueba_1/`
- Coordinate files in `coordenadas/` and `indices/`
- Main dataset in `COVID-19_Radiography_Dataset/`

## Important Notes

### Coordinate System
- All coordinates use 0-based indexing (0-63 range)
- Format: [y, x] in JSON files
- Image matrices follow numpy convention: [row, col]

### Template Matching
- Search regions defined by coordinate clusters
- Templates calculated using distances (a,b,c,d) from region bounds
- Intersection points mark optimal alignment positions

### Model Training
- Supports both linear PCA and Kernel PCA
- Default variance threshold: 0.95
- Kernel options: rbf, poly, sigmoid, linear
- Models trained per anatomical landmark (coord1-coord15)

### Dependencies
- Core: numpy, opencv-python, scikit-learn, matplotlib
- GUI: tkinter, Pillow
- Data: pandas, PyYAML
- Development: pytest, black, flake8, mypy

## Troubleshooting

### Common Issues
- **Memory errors**: Reduce batch size in training
- **Path errors**: Verify hardcoded paths match system
- **CUDA issues**: Check GPU availability for kernel PCA
- **Image corruption**: Validate dataset integrity

### Performance Optimization
- Adjust PCA variance threshold for model size
- Tune kernel parameters for accuracy
- Use parallel processing for batch operations
- Monitor memory usage during training