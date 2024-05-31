import json

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def pixelate_rgb(img, window):
    n, m, _ = img.shape
    img1 = np.zeros_like(img)
    for x in range(0, n, window):
        for y in range(0, m, window):
            x_end = min(x + window, n)
            y_end = min(y + window, m)
            img1[x:x_end, y:y_end] = img[x:x_end, y:y_end].mean(axis=(0, 1), keepdims=True)
    return img1


# Load the image
img = plt.imread('pic-main.jpg')

# Normalize the image if it is in integer format
if img.dtype == np.uint8:
    img = img.astype(float) / 255.0

# Pixelate the image with different window sizes
window_sizes = [5, 10, 20, 30]
pixelated_imgs = [pixelate_rgb(img, w) for w in window_sizes]

# Plot and save the pixelated images
fig, ax = plt.subplots(1, len(window_sizes), figsize=(20, 5))
for i, (pixelated_img, window_size) in enumerate(zip(pixelated_imgs, window_sizes)):
    ax[i].imshow(pixelated_img)
    ax[i].set_title(f'Pixel Size: {window_size}')
    ax[i].set_axis_off()

# Save the first pixelated image as a JPG file
plt.imsave('pixelated_image_1.jpg', pixelated_imgs[0], vmin=0, vmax=1)


def get_colors_hex(img):
    """
    Get the colors in hexadecimal format from the image.

    Parameters:
    - img: Image object.

    Returns:
    - JSON string containing the list of colors with their IDs, names, and hexadecimal values.
    """
    # Convert the image to RGB mode to ensure consistency
    img_rgb = img.convert('RGB')

    # Get the colors and their counts
    colors = img_rgb.getcolors(maxcolors=256*256)  # Increase the maxcolors to handle large images

    # Check if colors is None or empty
    if colors is None or len(colors) == 0:
        return json.dumps({"error": "No distinct colors found in the image."})

    # Convert RGB values to hexadecimal strings
    color_dict = [{"id": i+1, "name": f"Color {i+1}", "hex": '#' + ''.join(f'{c:02x}' for c in color)} for i, (count, color) in enumerate(colors)]

    return json.dumps(color_dict)


img = Image.open('pic-main.jpg')
colors_json = get_colors_hex(img)
print(colors_json)
