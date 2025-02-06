import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from pulmo_align.pulmo_align.pulmo_align.coordinates.coordinate_manager import CoordinateManager

def main():
    # Crear instancia del gestor
    manager = CoordinateManager()
    
    # Cargar coordenadas de búsqueda
    # Usar el archivo all_search_coordinates.json del módulo pulmo_align
    json_path = Path(__file__).parent / 'pulmo_align/pulmo_align/all_search_coordinates.json'
    manager.read_search_coordinates(str(json_path))
    
    # Verificar los límites calculados para cada coordenada
    for i in range(1, 16):
        coord_name = f"Coord{i}"
        config = manager.get_coordinate_config(coord_name)
        if config:
            print(f"\n{coord_name}:")
            print(f"sup: {config['sup']}, inf: {config['inf']}")
            print(f"left: {config['left']}, right: {config['right']}")
            print(f"width: {config['width']}, height: {config['height']}")

if __name__ == "__main__":
    main()
