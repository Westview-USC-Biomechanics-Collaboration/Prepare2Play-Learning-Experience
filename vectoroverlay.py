import pandas as pd
import json

# Read the .csv file
data = pd.read_csv('data/SM_SbS_02_Raw_Data - SM_SoftballSwing_Trial2_Raw_Data.csv')

# Assuming your CSV has columns: time, x_coord, y_coord, force_x, force_y
x_coords = data['x_coord'].tolist()
y_coords = data['y_coord'].tolist()
force_x = data['force_x'].tolist()
force_y = data['force_y'].tolist()

# Save the data to a JSON file for the HTML to use
output_data = {
    'x_coords': x_coords,
    'y_coords': y_coords,
    'force_x': force_x,
    'force_y': force_y
}

with open('data.json', 'w') as outfile:
    json.dump(output_data, outfile)
