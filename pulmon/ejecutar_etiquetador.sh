#!/bin/bash

# Script para ejecutar el programa de etiquetado de imágenes médicas
# Configura el PYTHONPATH y ejecuta el programa

# Obtener el directorio actual
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configurar PYTHONPATH al directorio padre
export PYTHONPATH="$(dirname "$DIR")"

# Ejecutar el programa
python3 "$DIR/etiquetar_imagenes.py"
