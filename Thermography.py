import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

# --- 1. Fetch Dataset ---
# This part remains the same.
print("Fetching dataset from ucimlrepo...")
infrared_thermography_temperature = fetch_ucirepo(id=925)

# Extract features (X) and target (y) as pandas DataFrames/Series
X = infrared_thermography_temperature.data.features
y = infrared_thermography_temperature.data.targets
print("Dataset fetched successfully.")

# --- 2. Combine and Save Data ---
# Combine features and target into a single table
combined_df = pd.concat([X, y], axis=1)

# Define the full, absolute path for the output file
output_path = "C:/Users/rowan/Documents/themograpy_table.csv" # Change path here

# Make sure the directory exists before trying to save the file
output_dir = os.path.dirname(output_path)
os.makedirs(output_dir, exist_ok=True)

# Save the combined DataFrame to the specified CSV file
# index=False prevents pandas from writing the row numbers into the file
try:
    combined_df.to_csv(output_path, index=False)
    print(f"Data successfully saved to: {output_path}")
except Exception as e:
    print(f"An error occurred while saving the file: {e}")
