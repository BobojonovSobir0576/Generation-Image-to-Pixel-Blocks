from rest_framework import serializers
from .models import ImageModel
from PIL import Image, ImageDraw, ImageStat
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import io
import time
from pixelator import Pixelator


class ImageModelSerializer(serializers.ModelSerializer):
    image = serializers.ImageField()

    class Meta:
        model = ImageModel
        fields = ['id', 'image', 'colors']
        read_only_fields = ['colors']

    def create(self, validated_data):
        image_instance = ImageModel.objects.create(image=validated_data['image'])

        # Open and resize original image if needed (optional)
        img_original = Image.open(image_instance.image.path)
        max_size = 512
        if max(img_original.width, img_original.height) > max_size:
            img_original.thumbnail((max_size, max_size), Image.LANCZOS)

        # Pixelate the original image
        block_size = 6
        pixelated_img_original = self.pixelate_rgb(img_original, block_size)

        # Save pixelated image temporarily
        pixelated_img_original.save("media/pixels/pixel.jpg")

        # Read the saved pixelated image and save it to image_instance.image
        with open("media/pixels/pixel.jpg", 'rb') as f:
            image_instance.image.save('pixel.jpg', f)

        # Get colors from the pixelated image
        colors_original = self.get_colors_hex(pixelated_img_original)

        # Update ImageModel instance with colors
        image_instance.colors = colors_original
        image_instance.save()

        return image_instance

    def get_colors_hex(self, img, n_colors=10):
        img_rgb = img.convert('RGB')
        img_array = np.array(img_rgb)
        img_flat = np.unique(img_array.reshape(-1, 3), axis=0)

        # Apply color quantization (e.g., K-means clustering)
        kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(img_flat)

        # Get cluster centers (colors)
        cluster_centers = kmeans.cluster_centers_.astype(int)

        # Generate color dictionary with hex values
        color_dict = []
        for i, color in enumerate(cluster_centers):
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
            color_dict.append({
                "id": i + 1,
                "name": f"Color {i + 1}",
                "hex": hex_color
            })

        return color_dict

    def pixelate_rgb(self, img, block_size):
        width, height = img.size
        new_width = width // block_size
        new_height = height // block_size
        small_img = img.resize((new_width, new_height), resample=Image.BILINEAR)
        pixelated_img = small_img.resize(img.size, resample=Image.NEAREST)
        return pixelated_img

class ImageListSerializer(serializers.ModelSerializer):

    class Meta:
        model = ImageModel
        fields = '__all__'

