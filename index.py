from PIL import Image

def count_old_color_pixels(image_path, old_hex, tolerance=10):
    # Open the image
    img = Image.open(image_path)
    img = img.convert("RGBA")  # Convert image to RGBA mode for transparency support
    width, height = img.size

    # Convert hex string to RGB tuple
    old_rgb = tuple(int(old_hex[i:i+2], 16) for i in (0, 2, 4))

    # Initialize count for old color pixels
    count = 0

    # Iterate through each pixel in the image
    for x in range(width):
        for y in range(height):
            current_color = img.getpixel((x, y))[:3]  # Get RGB values of current pixel

            # Check if the current pixel color is within tolerance of the old color
            if all(abs(current_color[i] - old_rgb[i]) <= tolerance for i in range(3)):
                count += 1

    return count

# Example usage:
image_path = 'pixel_KPHTzCU.jpg'
old_color_hex = 'f3e5bd'  # Example: Old color in hex

num_old_color_pixels = count_old_color_pixels(image_path, old_color_hex)
print(f"Number of pixels with old color '{old_color_hex}': {num_old_color_pixels}")

def modify_color(image_path, old_hex, new_hex, tolerance=10):
    # Open the image
    img = Image.open(image_path)
    img = img.convert("RGBA")  # Convert image to RGBA mode for transparency support
    width, height = img.size

    # Convert hex strings to RGB tuples
    old_rgb = tuple(int(old_hex[i:i+2], 16) for i in (0, 2, 4))
    new_rgb = tuple(int(new_hex[i:i+2], 16) for i in (0, 2, 4))

    # Create a blank image for output
    modified_img = Image.new('RGBA', (width, height))

    # Iterate through each pixel in the image
    for x in range(width):
        for y in range(height):
            current_color = img.getpixel((x, y))[:3]  # Get RGB values of current pixel

            # Check if the current pixel color is within tolerance of the old color
            if all(abs(current_color[i] - old_rgb[i]) <= tolerance for i in range(3)):
                # Replace the old color with the new color
                modified_img.putpixel((x, y), new_rgb)
            else:
                # Preserve the original pixel if it doesn't match the old color
                modified_img.putpixel((x, y), img.getpixel((x, y)))

    # Save modified image
    modified_img.save('modified_image.png')

# Example usage:
image_path = 'pixel_KPHTzCU.jpg'
old_color_hex = '172c20'  # Example: Old color in hex
new_color_hex = '1e00ff'  # Example: New color in hex

modify_color(image_path, old_color_hex, new_color_hex)
