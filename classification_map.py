import os
import argparse
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerPatch
from scipy import ndimage

# Argument parsing
parser = argparse.ArgumentParser(description='Generate classification maps from cloud and land NetCDF files.')
parser.add_argument('cloud_file', type=str, help='Path to the cloud NetCDF file')
parser.add_argument('land_file', type=str, help='Path to the land NetCDF file')
parser.add_argument('--all_land', action='store_true', help='Show all land regardless of cloud cover')
parser.add_argument('--x_min', type=int, default=None, help='Minimum x index for subset')
parser.add_argument('--x_max', type=int, default=None, help='Maximum x index for subset')
parser.add_argument('--y_min', type=int, default=None, help='Minimum y index for subset')
parser.add_argument('--y_max', type=int, default=None, help='Maximum y index for subset')
parser.add_argument('--output', type=str, required=True, help='Directory to save output images')
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output, exist_ok=True)

# Load cloud and land datasets
ds_cloud = Dataset(args.cloud_file, 'r')
ds_land = Dataset(args.land_file, 'r')
cloud_data = ds_cloud.variables['cma_extended'][0, :, :]
land_data = ds_land.variables['landuse'][0, :, :]

# Apply subsetting if coordinates are provided
if all(v is not None for v in [args.x_min, args.x_max, args.y_min, args.y_max]):
    cloud_data = cloud_data[args.y_min:args.y_max+1, args.x_min:args.x_max+1]
    land_data = land_data[args.y_min:args.y_max+1, args.x_min:args.x_max+1]

# Convert to numpy arrays for mask operations
cloud_data = np.array(cloud_data)
land_data = np.array(land_data)

# Define masks for different categories
no_data_mask = (cloud_data == 255) | (land_data == 255)
cloudfree_mask = (cloud_data == 0)
cloud_mask = ((cloud_data == 1) | (cloud_data == 2))
cfree_land_mask = ((land_data != 16) & (cloudfree_mask))
cfree_water_mask = ((land_data == 16) & cloudfree_mask)
all_land_mask = (land_data != 16)
ice_mask = (cloud_data == 3)

# Choose land mask depending on --all_land flag
land_mask = all_land_mask if args.all_land else cfree_land_mask
water_mask = cfree_water_mask

# Structuring element for neighborhood operations
struct3 = ndimage.generate_binary_structure(2, 1)

# Define colors and labels for classification map
colors = [
    'black', 'blue', 'white', 'gray', 'green',
    'red', 'orange', 'gold', 'cyan', 'navy', 'purple'
]
labels = [
    'No data', 'Water', 'Ice', 'Clouds', 'Land',
    'InW', 'I2nW', 'I3nW', 'WnI', 'W2nI', 'W3nI'
]
cmap = ListedColormap(colors)

# Custom legend handler for proximity overlay
colors_dict = {'water': 'blue', 'ice': 'white', 'cloud': 'gray', 'land': 'green'}
class HalfPatchHandler(HandlerPatch):
    def __init__(self, left_color, center_color, right_color):
        self.left_color = left_color
        self.center_color = center_color
        self.right_color = right_color
        super().__init__()
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        p1 = Rectangle([xdescent, ydescent], width/3, height,
                       facecolor=self.left_color, hatch='///',
                       edgecolor='gray', transform=trans)
        p2 = Rectangle([xdescent + width/3, ydescent], width/3, height,
                       facecolor=self.center_color, hatch='///',
                       edgecolor='gray', transform=trans)
        p3 = Rectangle([xdescent + 2*width/3, ydescent], width/3, height,
                       facecolor=self.right_color, hatch='///',
                       edgecolor='gray', transform=trans)
        return [p1, p2, p3]

# Step 1: Base classification
classification_base = np.zeros(cloud_data.shape, dtype=np.uint8)
classification_base[no_data_mask] = 0
classification_base[water_mask] = 1
classification_base[ice_mask] = 2
classification_base[cloud_mask] = 3
classification_base[land_mask] = 4

plt.figure(figsize=(10, 8))
plt.imshow(classification_base, interpolation='none', cmap=cmap, vmin=0, vmax=10)
plt.title('Step 1: Base Classification')
plt.axis('off')
patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(5)]
plt.legend(handles=patches, loc='lower right', fontsize='small')
plt.savefig(os.path.join(args.output, 'classification_base.png'), dpi=300, bbox_inches='tight')
plt.close()

# Step 2: Proximity to clouds or land
cloud_dilated = ndimage.binary_dilation(cloud_mask.astype(bool), structure=struct3, iterations=2)
cloud_proximity_mask = cloud_dilated & (~cloud_mask)
land_dilated = ndimage.binary_dilation(cfree_land_mask.astype(bool), structure=struct3, iterations=2)
land_proximity_mask = land_dilated & (~cfree_land_mask)
combined_proximity_mask = (cloud_proximity_mask | land_proximity_mask) & (~no_data_mask)

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(classification_base, interpolation='none', cmap=cmap, vmin=0, vmax=10)
ax.set_title('Step 2: Classification + Proximity')
ax.axis('off')

# Overlay hatch pattern for proximity areas
for (r, c), val in np.ndenumerate(combined_proximity_mask):
    if val:
        pixel_class = classification_base[r, c]
        if pixel_class == 0:
            base_color = colors[0]
        elif pixel_class == 2:
            base_color = colors_dict['ice']
        elif pixel_class == 4:
            base_color = colors_dict['land']
        else:
            base_color = colors_dict['water']
        ax.add_patch(Rectangle((c-0.5, r-0.5), 1, 1,
                               facecolor=base_color, edgecolor='lightgray',
                               hatch='///', linewidth=0))

patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(5)]
prox_handle = Rectangle((0, 0), 1, 1)
patches.append(prox_handle)
ax.legend(handles=patches,
          labels=['No data', 'Water', 'Ice', 'Clouds', 'Land', 'Prox to C/L'],
          handler_map={prox_handle: HalfPatchHandler(colors_dict['ice'], colors_dict['land'], colors_dict['water'])},
          loc='lower right', fontsize='small')
plt.savefig(os.path.join(args.output, 'classification_proximity.png'), dpi=300, bbox_inches='tight')
plt.close()

# Step 3: Neighbors (ice ↔ water)
def plot_neighbors(mode, fname):
    classification = classification_base.copy()
    ice_bool = (classification == 2)
    water_bool = (classification == 1)

    if mode == "ice":
        # Ice expansion into water (wNi, wNNi, wNNNi)
        dilated_ice = ndimage.binary_dilation(ice_bool, structure=struct3)
        wNi   = dilated_ice & water_bool & (~ice_bool)
        wNNi  = ndimage.binary_dilation(wNi, structure=struct3) & water_bool & (~wNi)
        wNNNi = ndimage.binary_dilation(wNNi, structure=struct3) & water_bool & (~wNi) & (~wNNi)
        classification[wNi]   = 8
        classification[wNNi]  = 9
        classification[wNNNi] = 10

    elif mode == "water":
        # Water expansion into ice (iNw, iNNw, iNNNw)
        dilated_water = ndimage.binary_dilation(water_bool, structure=struct3)
        iNw   = dilated_water & ice_bool & (~water_bool)
        iNNw  = ndimage.binary_dilation(iNw, structure=struct3) & ice_bool & (~iNw)
        iNNNw = ndimage.binary_dilation(iNNw, structure=struct3) & ice_bool & (~iNw) & (~iNNw)
        classification[iNw]   = 5
        classification[iNNw]  = 6
        classification[iNNNw] = 7

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(classification, interpolation='none', cmap=cmap, vmin=0, vmax=10)
    ax.set_title(f'Step 3: Neighbors ({mode}) + Proximity')
    ax.axis('off')

    # Overlay hatch pattern for proximity areas
    for (r, c), val in np.ndenumerate(combined_proximity_mask):
        if val:
            pixel_class = classification[r, c]
            if pixel_class == 0:
                base_color = colors[0]
            elif pixel_class == 2:
                base_color = colors_dict['ice']
            elif pixel_class == 4:
                base_color = colors_dict['land']
            else:
                base_color = colors_dict['water']
            ax.add_patch(Rectangle((c-0.5, r-0.5), 1, 1,
                                   facecolor=base_color, edgecolor='lightgray',
                                   hatch='///', linewidth=0))

    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]
    prox_handle = Rectangle((0, 0), 1, 1)
    patches.append(prox_handle)
    ax.legend(handles=patches,
              labels=labels + ['Prox to C/L'],
              handler_map={prox_handle: HalfPatchHandler(colors_dict['ice'], colors_dict['land'], colors_dict['water'])},
              loc='lower right', fontsize='small')
    plt.savefig(os.path.join(args.output, fname), dpi=300, bbox_inches='tight')
    plt.close()

# Generate neighbor maps
plot_neighbors("ice", "classification_neighbors_ice.png")
plot_neighbors("water", "classification_neighbors_water.png")

print(f"Saved results in {args.output}")