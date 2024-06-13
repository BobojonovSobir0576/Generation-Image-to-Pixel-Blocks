from PIL import Image
import uuid

def pixelate_image(image_path):
    # Open the image
    pixel_size = 16
    img = Image.open(image_path)

    # Calculate dimensions for the pixelation
    width, height = img.size
    new_width = width // pixel_size
    new_height = height // pixel_size

    # Resize the image to small size
    small_img = img.resize((new_width, new_height), resample=Image.BILINEAR)

    # Scale back up using NEAREST to introduce pixelation effect
    pixelated_img = small_img.resize(img.size, resample=Image.NEAREST)

    return pixelated_img


def images(image):
    image_path = image
    pixel_size = 16  # Assuming we want 16x16 big pixels
    pixelated_image = pixelate_image(image_path)
    pixelated_image.save(f"media/pixels/pixel.jpg")  # Save the pixelated image
