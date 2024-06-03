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
        img = Image.open(image_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize image to reduce processing time
        img = img.resize((img.width // 2, img.height // 2))

        img_array = np.array(img).astype(float) / 255.0

        pixelated_img = self.pixelate_rgb(img_array, 5)
        lego_img = self.apply_lego_effect(pixelated_img, 5)

        # Convert back to image and save
        lego_img_pil = Image.fromarray((lego_img * 255).astype(np.uint8))
        lego_img_pil.save(image_path)

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

                # Create a simplified block effect without heavy drawing
                img_lego[x:x_end, y:y_end] = color

        return img_lego

    def get_colors_hex(self, img, n_colors=10):
        img_rgb = img.convert('RGB')
        img_array = np.array(img_rgb)
        img_array = img_array.reshape((-1, 3))

        # Use KMeans to cluster colors
        kmeans = KMeans(n_clusters=n_colors, n_init=10, max_iter=300)
        kmeans.fit(img_array)
        unique_colors = kmeans.cluster_centers_.astype(int)

        color_dict = [{"id": i+1, "name": f"Color {i+1}", "hex": '#' + ''.join(f'{c:02x}' for c in color)} for i, color in enumerate(unique_colors)]
        return color_dict

class ImageListSerializer(serializers.ModelSerializer):

    class Meta:
        model = ImageModel
        fields = '__all__'
