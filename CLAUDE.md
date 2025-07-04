# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a medical image analysis research project for lung X-ray image processing and anatomical landmark detection. The project implements PCA-based models for analyzing pulmonary radiographs using computer vision techniques.

## Project Structure

```
/
├── COVID-19_Radiography_Dataset/    # Source medical image dataset
├── coordenadas/                     # Coordinate data files (CSV format)
├── dataset/                         # Processed and organized image datasets
├── entrenamiento/                   # Training module for PCA models
├── etiquetado/                      # Image annotation/labeling module
├── indices/                         # Image index files
├── prediccion/                      # Prediction and coordinate extraction modules
├── recorte/                         # Image cropping and processing module
├── tools/                           # Utility scripts and analysis tools
└── requirements.txt                 # Main project dependencies
```

## Common Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Module-Specific Commands

#### Image Annotation
```bash
# Run image labeling tool
cd etiquetado
./ejecutar_etiquetador.sh
# or
python3 etiquetar_imagenes.py
```

#### Training Module
```bash
cd entrenamiento
python scripts/train_models.py --config config.yaml
```

#### Image Cropping
```bash
cd recorte/src
python main.py
```

#### Prediction
```bash
cd prediccion
python prediccion_coordenadas.py
```

### Testing
```bash
# Run tests (pytest is included in requirements)
pytest tests/
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## Architecture Overview

### Core Components

1. **Data Pipeline**: 
   - Raw images from COVID-19_Radiography_Dataset
   - Coordinate files linking images to anatomical landmarks
   - Index files for dataset organization

2. **Processing Modules**:
   - **etiquetado**: Interactive annotation tool for creating training data
   - **recorte**: Image preprocessing and region extraction
   - **entrenamiento**: PCA model training and statistical analysis
   - **prediccion**: Coordinate prediction and landmark detection

3. **Data Flow**:
   - Images → Annotation → Coordinate extraction → PCA training → Prediction
   - Each module operates on CSV coordinate files and image datasets

### Key Technologies

- **Computer Vision**: OpenCV for image processing
- **Machine Learning**: scikit-learn for PCA analysis
- **Data Processing**: pandas/numpy for coordinate manipulation
- **Visualization**: matplotlib for plotting and analysis
- **GUI**: tkinter for annotation interface

## File Conventions

- **CSV Files**: Coordinate data with consistent column structure
- **Image Naming**: Follows pattern: `{Category}-{ID}.png` (e.g., COVID-1234.png, Normal-5678.png)
- **Results**: Stored in module-specific directories (models/, visualization_results/, etc.)

## Development Notes

- Python 3.6+ required
- Virtual environment strongly recommended
- All modules use absolute paths for cross-platform compatibility
- Medical image data requires careful handling - no PHI in commits
- Models and large datasets are excluded from git (see .gitignore)

## Module Dependencies

- **entrenamiento** → generates models for **prediccion**
- **etiquetado** → generates coordinate files for **entrenamiento**
- **recorte** → preprocesses images for all modules
- **tools** → provides utilities for all modules

## Common Patterns

- CSV coordinate files have consistent format across modules
- Image processing uses 64x64 pixel templates
- PCA models stored in .joblib format
- Visualization results saved as PNG files
- Progress tracking with tqdm for long-running operations