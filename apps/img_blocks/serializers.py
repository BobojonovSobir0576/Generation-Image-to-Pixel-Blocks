from rest_framework import serializers
from .models import ImageModel
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json
from sklearn.cluster import KMeans

class ImageModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageModel
        fields = ['id', 'image', 'colors']
        read_only_fields = ['colors']

    def create(self, validated_data):
        image_instance = ImageModel.objects.create(image=validated_data['image'])

        # Process the image to get colors
        image_path = image_instance.image.path
        img = plt.imread(image_path)

        if img.dtype == np.uint8:
            img = img.astype(float) / 255.0

        pixelated_img = self.pixelate_rgb(img, 5)  # Using window size 5 for pixelation
        lego_img = self.apply_lego_effect(pixelated_img, 5)
        plt.imsave(image_path, lego_img, vmin=0, vmax=1)

        img_pil = Image.open(image_path)
        colors = self.get_colors_hex(img_pil)

        # Save colors as a JSON string
        image_instance.colors = list(colors)
        image_instance.save()

        return image_instance

    def pixelate_rgb(self, img, window):
        n, m, _ = img.shape
        img1 = np.zeros_like(img)
        for x in range(0, n, window):
            for y in range(0, m, window):
                x_end = min(x + window, n)
                y_end = min(y + window, m)
                img1[x:x_end, y:y_end] = img[x:x_end, y:y_end].mean(axis=(0, 1), keepdims=True)
        return img1

    def apply_lego_effect(self, img, block_size):
        n, m, _ = img.shape
        img_lego = np.zeros_like(img)
        for x in range(0, n, block_size):
            for y in range(0, m, block_size):
                x_end = min(x + block_size, n)
                y_end = min(y + block_size, m)
                block = img[x:x_end, y:y_end]
                color = block.mean(axis=(0, 1))

                # Handle blocks that are not full size at the edges
                block_shaded = self.add_lego_bulge(color, x_end - x, y_end - y)
                img_lego[x:x_end, y:y_end] = block_shaded

        return img_lego

    def add_lego_bulge(self, color, block_height, block_width):
        color = (color * 255).astype(int)
        color_hex = ''.join(f'{c:02x}' for c in color)

        img_block = Image.new('RGB', (block_width, block_height), f'#{color_hex}')
        draw = ImageDraw.Draw(img_block)

        # Create a circular bulge effect
        center_x, center_y = block_width // 2, block_height // 2
        radius = min(center_x, center_y) - 1  # Adjust radius to fit within block

        # Draw the main circular bulge
        draw.ellipse([(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)], fill=f'#{color_hex}')

        # Create the shadow and highlight effects
        shadow = Image.new('RGBA', (block_width, block_height))
        draw_shadow = ImageDraw.Draw(shadow)
        draw_shadow.ellipse([(center_x - radius + 1, center_y - radius + 1), (center_x + radius - 1, center_y + radius - 1)], fill=(0, 0, 0, 80))
        img_block.paste(shadow, (0, 0), shadow)

        highlight = Image.new('RGBA', (block_width, block_height))
        draw_highlight = ImageDraw.Draw(highlight)
        draw_highlight.ellipse([(center_x - radius // 2, center_y - radius // 2), (center_x + radius // 2, center_y + radius // 2)], fill=(255, 255, 255, 80))
        img_block.paste(highlight, (0, 0), highlight)

        return np.array(img_block) / 255.0

    def get_colors_hex(self, img, n_colors=10):
        img_rgb = img.convert('RGB')
        img_array = np.array(img_rgb)
        img_array = img_array.reshape((-1, 3))

        # Use KMeans to cluster colors
        kmeans = KMeans(n_clusters=n_colors)
        kmeans.fit(img_array)
        unique_colors = kmeans.cluster_centers_.astype(int)

        color_dict = [{"id": i+1, "name": f"Color {i+1}", "hex": '#' + ''.join(f'{c:02x}' for c in color)} for i, color in enumerate(unique_colors)]
        return color_dict


class ImageListSerializer(serializers.ModelSerializer):

    class Meta:
        model = ImageModel
        fields = '__all__'
