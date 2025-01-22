import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def generate_search_zone(csv_file, coord_num):
    # Load data from CSV file
    df = pd.read_csv(csv_file, header=None)
    x_col = (coord_num - 1) * 2 + 1
    y_col = x_col + 1
    coord_x = df.iloc[:, x_col].values
    coord_y = df.iloc[:, y_col].values
    
    print(f"\nDiagnostics for Coord{coord_num}:")
    print(f"Total number of points: {len(coord_x)}")
    print(f"X Range: {coord_x.min()} - {coord_x.max()}")
    print(f"Y Range: {coord_y.min()} - {coord_y.max()}")
    
    # Create a 2D array for the heatmap
    heatmap = np.zeros((64, 64))
    for x, y in zip(coord_x, coord_y):
        x_adj, y_adj = x - 1, y - 1
        if 0 <= x_adj < 64 and 0 <= y_adj < 64:
            heatmap[y_adj, x_adj] += 1
        else:
            print(f"Warning: Coordinate ({x}, {y}) out of bounds")
    
    # Calculate the center and dimensions of the area of interest
    non_zero = np.nonzero(heatmap)
    if len(non_zero[0]) == 0:
        print("Error: No valid points in the heatmap")
        return
    
    min_y, max_y = non_zero[0].min(), non_zero[0].max()
    min_x, max_x = non_zero[1].min(), non_zero[1].max()
    
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    
    print(f"Center of area of interest: ({center_x}, {center_y})")
    print(f"Dimensions of area of interest:")
    print(f"  Height: {height}")
    print(f"  Width: {width}")
    
    # Create the search zone
    search_zone = np.zeros((64, 64))
    half_width = width // 2
    half_height = height // 2
    search_zone[max(0, center_y-half_height):min(64, center_y+half_height+1), 
                max(0, center_x-half_width):min(64, center_x+half_width+1)] = 1
    
    # Get coordinates of pixels within the search zone
    search_coordinates = np.argwhere(search_zone == 1)
    search_coordinates = [(int(y+1), int(x+1)) for y, x in search_coordinates]  # Adjust to 1-indexed
    
    print(f"Number of pixels in search zone: {len(search_coordinates)}")
    print("First 5 search zone coordinates:")
    for coord in search_coordinates[:5]:
        print(f"  {coord}")
    
    # Visualize the search zone
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(search_zone, cmap='RdYlBu_r', cbar=False, square=True, linewidths=.5, linecolor='black')
    plt.ylim(64, 0)
    
    # Add grid
    for i in range(0, 65, 8):
        plt.axhline(y=i, color='black', linewidth=1.5)
        plt.axvline(x=i, color='black', linewidth=1.5)
    
    plt.title(f'Search Zone for Coord{coord_num}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.xticks(np.arange(0, 65, 8))
    plt.yticks(np.arange(0, 65, 8))
    
    # Add text with dimensions
    plt.text(0, -2, f'X min: {min_x+1}', horizontalalignment='left', verticalalignment='top')
    plt.text(64, -2, f'X max: {max_x+1}', horizontalalignment='right', verticalalignment='top')
    plt.text(32, 66, f'Y min: {min_y+1}', horizontalalignment='center', verticalalignment='bottom')
    plt.text(32, -2, f'Y max: {max_y+1}', horizontalalignment='center', verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f'coord{coord_num}_search_zone.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return search_coordinates

# Execute the analysis for each coordinate
csv_file = 'Tesis/coordenadas_64x64.csv'
all_search_zones = {}

for coord_num in range(1, 16):
    search_coordinates = generate_search_zone(csv_file, coord_num)
    all_search_zones[f'coord{coord_num}'] = search_coordinates

# Save all search coordinates to a single JSON file
with open('all_search_coordinates_200.json', 'w') as f:
    json.dump(all_search_zones, f, indent=2)

print("\nAll search coordinates have been saved to 'all_search_coordinates_200.json'")