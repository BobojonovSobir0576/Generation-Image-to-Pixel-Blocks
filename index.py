from PIL import Image
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import itertools
import json


def rgb_to_hex(red, green, blue):
    return f'#{red:02x}{green:02x}{blue:02x}'


def generate_names_and_ids(n):
    letters = ['A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y']
    numbers = [str(i) for i in range(1, 10)]

    # Generate all possible combinations
    names = []
    ids = []
    length = 1
    current_id = 1

    while len(names) < n:
        length += 1
        combinations = itertools.product(letters, repeat=length - 1)
        for combination in combinations:
            for number in numbers:
                names.append(''.join(combination) + number)
                ids.append(current_id)
                current_id += 1
                if len(names) >= n:
                    break
            if len(names) >= n:
                break
    return names, ids


def get_color_table(image_path, n_clusters=256):
    # Open the image
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure image is in RGB format

    # Get image data
    pixels = np.array(img.getdata())

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(pixels)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Create a DataFrame to store the colors and their frequencies
    df = pd.DataFrame(pixels, columns=['Red', 'Green', 'Blue'])
    df['Cluster'] = labels

    # Aggregate by clusters
    color_table = df.groupby('Cluster').size().reset_index(name='Frequency')
    color_table[['Red', 'Green', 'Blue']] = cluster_centers[color_table['Cluster']].astype(int)
    color_table['hex'] = color_table.apply(lambda row: rgb_to_hex(row['Red'], row['Green'], row['Blue']), axis=1)

    # Generate unique names and IDs
    names, ids = generate_names_and_ids(len(color_table))
    color_table['name'] = names
    color_table['id'] = ids

    # Reorder columns
    color_table = color_table[['id', 'Red', 'Green', 'Blue', 'Frequency', 'hex', 'name']]

    # Map each pixel to its cluster's name, id, hex, and color name
    pixel_mapping = []
    unique_ids = {}  # To store unique IDs for each cluster

    for i, label in enumerate(labels):
        cluster_id = color_table.loc[label, 'id']
        if cluster_id not in unique_ids:
            unique_ids[cluster_id] = len(unique_ids) + 1  # Generate new unique ID
        pixel_info = {
            "color_name": color_table.loc[label, 'name'],
            "id": unique_ids[cluster_id],
            "hex": color_table.loc[label, 'hex']
        }
        pixel_mapping.append(pixel_info)

    # Reshape pixel mapping to match image dimensions
    pixel_mapping_2d = [pixel_mapping[i * img.width:(i + 1) * img.width] for i in range(img.height)]

    return color_table, pixel_mapping_2d


# Path to your image
image_path = 'pixel_KPHTzCU.jpg'

# Generate color table and pixel name mapping
color_table, pixel_name_mapping = get_color_table(image_path, n_clusters=256)

# Save the color table to a JSON file
color_table.to_json('color_table.json', orient='records')

# Save the pixel name mapping to a JSON file
with open('pixel_name_mapping.json', 'w') as f:
    json.dump(pixel_name_mapping, f, indent=2)

print("Color table saved to 'color_table.json'")
print("Pixel name mapping saved to 'pixel_name_mapping.json'")

from PIL import Image


def cut_image(image_path, width, height):
    # Open the image
    img = Image.open(image_path)

    # Get the dimensions of the image
    img_width, img_height = img.size

    # Calculate number of rows and columns
    rows = img_height // height
    cols = img_width // width

    # Cut the image into tiles
    for r in range(rows):
        for c in range(cols):
            box = (c * width, r * height, (c + 1) * width, (r + 1) * height)
            tile = img.crop(box)
            # Save each tile with a unique name or process it as needed
            tile.save(f"tile_{r}_{c}.png")  # Example: Saving each tile as a PNG file

    # Any remainder of the image that doesn't fit into a complete tile can be ignored
    # or handled separately based on your needs.

    print(f"Image cut into {rows} rows and {cols} columns of size {width}x{height}.")


# Example usage:
image_path = "pixel_KPHTzCU.jpg"
cut_image(image_path, 9, 15)

