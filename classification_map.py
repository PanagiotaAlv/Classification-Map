import os
import argparse
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Argument parsing
parser = argparse.ArgumentParser(description='Generate classification map from cloud and land NetCDF files.')
parser.add_argument('cloud_file', type=str, help='Path to the cloud NetCDF file')
parser.add_argument('land_file', type=str, help='Path to the land NetCDF file')
parser.add_argument('--all_land', action='store_true', help='Show all land regardless of cloud cover')
parser.add_argument('--x_min', type=int, default=None, help='Minimum x index for subset')
parser.add_argument('--x_max', type=int, default=None, help='Maximum x index for subset')
parser.add_argument('--y_min', type=int, default=None, help='Minimum y index for subset')
parser.add_argument('--y_max', type=int, default=None, help='Maximum y index for subset')
parser.add_argument('--output', type=str, required=True, help='Directory to save output image')
args = parser.parse_args()

# Load datasets
ds_cloud = Dataset(args.cloud_file, 'r')
ds_land = Dataset(args.land_file, 'r')

cloud_data = ds_cloud.variables['cma_extended'][0, :, :]
land_data = ds_land.variables['landuse'][0, :, :]

# Subset if coordinates provided
if all(v is not None for v in [args.x_min, args.x_max, args.y_min, args.y_max]):
    cloud_data = cloud_data[args.y_min:args.y_max+1, args.x_min:args.x_max+1]
    land_data = land_data[args.y_min:args.y_max+1, args.x_min:args.x_max+1]

# Convert to NumPy arrays
cloud_data = np.array(cloud_data)
land_data = np.array(land_data)

# Create masks
no_data_mask = (cloud_data == 255) | (land_data == 255)
cloudfree_mask = (cloud_data == 0)
cloud_mask = ((cloud_data == 1) | (cloud_data == 2))
cfree_land_mask = ((land_data != 16) & (cloudfree_mask))
cfree_water_mask = ((land_data == 16) & cloudfree_mask)
all_land_mask = (land_data != 16)
ice_mask = (cloud_data == 3)

# Create classification map
classification = np.zeros(cloud_data.shape)
classification[cloud_mask] = 3
classification[cfree_land_mask] = 4
classification[cfree_water_mask] = 1
classification[ice_mask] = 2
classification[no_data_mask] = 0

# If --all_land is set, override with all land in green
if args.all_land:
    classification[all_land_mask] = 4

# Plot
colors = ['black', 'blue', 'white', 'gray', 'green']
labels = ['No data', 'Water', 'Ice', 'Clouds', 'Land']
cmap = ListedColormap(colors)

plt.figure(figsize=(10, 8))
plt.imshow(classification, interpolation='none', cmap=cmap, vmin=0, vmax=4)
plt.title('Classification Map')
plt.axis('off')
patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]
plt.legend(handles=patches, loc='lower right')

# Always save the figure
output_path = os.path.join(args.output, 'classification_map.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved classification map to {output_path}")
