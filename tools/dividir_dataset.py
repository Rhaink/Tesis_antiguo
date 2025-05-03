import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Define input and output filenames
indices_maestro_file = 'Tesis/indices/indices_maestro_1.csv'
coordenadas_maestro_file = 'Tesis/coordenadas/coordenadas_aligned_maestro_1.csv'

indices_entrenamiento_file = 'Tesis/indices/indices_aligned_entrenamiento_1.csv'
coordenadas_entrenamiento_file = 'Tesis/coordenadas/coordenadas_aligned_entrenamiento_1.csv'
indices_prueba_file = 'Tesis/indices/indices_aligned_prueba_1.csv'
coordenadas_prueba_file = 'Tesis/coordenadas/coordenadas_aligned_prueba_1.csv'

# Define the split ratio (20% for testing, so test_size = 0.2)
test_size_ratio = 0.20

# Define the random seed for reproducibility
random_seed_value = 1 

# --- Step 1: Check if input files exist ---
if not os.path.exists(indices_maestro_file):
    print(f"Error: Input file '{indices_maestro_file}' not found.")
    exit()
if not os.path.exists(coordenadas_maestro_file):
    print(f"Error: Input file '{coordenadas_maestro_file}' not found.")
    exit()

print(f"Reading master files: {indices_maestro_file} and {coordenadas_maestro_file}")

# --- Step 2: Read the CSV files into pandas DataFrames ---
# T|he files do NOT have a header row
# header=None tells pandas not to treat the first row as headers
try:
    df_indices = pd.read_csv(indices_maestro_file, header=None)
    df_coords = pd.read_csv(coordenadas_maestro_file, header=None)
except Exception as e:
    print(f"Error reading CSV files: {e}")
    exit()

# --- Step 3: Verify data consistency ---
# Ensure both files have the same number of rows
if len(df_indices) != len(df_coords):
    print("Error: The number of rows in indices_maestro.csv and coordenadas_maestro.csv do not match.")
    print(f"Indices rows: {len(df_indices)}, Coordenadas rows: {len(df_coords)}")
    exit()

total_rows = len(df_indices)
print(f"Successfully read {total_rows} rows from both master files.")

# --- Step 4: Perform the random split ---
# train_test_split splits the data into random train and test subsets.
# By passing both dataframes together, it ensures the split is applied
# consistently to the rows in both files, maintaining their correspondence.
# random_state ensures the split is the same every time you run the script with this seed.
try:
    indices_train_df, indices_test_df, coords_train_df, coords_test_df = train_test_split(
        df_indices,
        df_coords,
        test_size=test_size_ratio,
        random_state=random_seed_value
        # If you need to ensure the class distribution (column 1 in indices_maestro_1.csv)
        # is similar in both train and test sets, uncomment the line below:
        # stratify=df_indices[1]
    )
except Exception as e:
    print(f"Error during train/test split: {e}")
    exit()

print(f"Split successful: {len(indices_train_df)} training rows, {len(indices_test_df)} testing rows.")

# --- Step 5: Save the split DataFrames to new CSV files ---
# Use index=False because we don't want pandas to write the DataFrame index as a column
# Use header=False because the original files did not have a header row
try:
    indices_train_df.to_csv(indices_entrenamiento_file, index=False, header=False)
    coords_train_df.to_csv(coordenadas_entrenamiento_file, index=False, header=False)
    indices_test_df.to_csv(indices_prueba_file, index=False, header=False)
    coords_test_df.to_csv(coordenadas_prueba_file, index=False, header=False)

    print("\nSuccessfully created the following files:")
    print(f"- {indices_entrenamiento_file}")
    print(f"- {coordenadas_entrenamiento_file}")
    print(f"- {indices_prueba_file}")
    print(f"- {coordenadas_prueba_file}")

except Exception as e:
    print(f"Error saving output CSV files: {e}")
    exit()