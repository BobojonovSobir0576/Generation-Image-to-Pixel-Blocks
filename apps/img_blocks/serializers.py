from rest_framework import serializers
from .models import ImageModel
from PIL import Image, ImageDraw, ImageStat
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import io
import time
from pixelator import Pixelator
from io import BytesIO
from django.core.files.base import ContentFile


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
        block_size = 4
        pixelated_img_original = self.pixelate_rgb(img_original, block_size)

        # Save pixelated image temporarily in memory
        pixelated_img_original_io = BytesIO()
        pixelated_img_original.save(pixelated_img_original_io, format='JPEG')
        pixelated_img_original_content = ContentFile(pixelated_img_original_io.getvalue(), 'pixel.jpg')

        # Save the pixelated image to image_instance.image
        image_instance.image.save('pixel.jpg', pixelated_img_original_content)

        # Get colors from the pixelated image
        colors_original = self.get_colors_hex(pixelated_img_original, n_colors=256)

        # Update ImageModel instance with colors
        image_instance.colors = colors_original
        image_instance.main_colors = image_instance.colors
        image_instance.save()

        return image_instance

    def get_colors_hex(self, img, n_colors=256):
        img_rgb = img.convert('RGB')
        img_array = np.array(img_rgb)
        img_flat = img_array.reshape(-1, 3)

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
        # Convert image to numpy array for easier manipulation
        img_array = np.array(img)
        n, m, _ = img_array.shape
        n, m = n - n % block_size, m - m % block_size
        img_array = img_array[:n, :m, :]
        img1 = np.zeros((n, m, 3), dtype=np.uint8)

        for x in range(0, n, block_size):
            for y in range(0, m, block_size):
                img1[x:x + block_size, y:y + block_size] = img_array[x:x + block_size, y:y + block_size].mean(axis=(0, 1))

        # Convert the pixelated array back to an image
        pixelated_img = Image.fromarray(img1)

        return pixelated_img

class ImageListSerializer(serializers.ModelSerializer):
    parent = serializers.SerializerMethodField()

    class Meta:
        model = ImageModel
        fields = ['id', 'image', 'colors', 'parent']

    def get_parent(self, obj):
        if obj.parent is not None:
            # Correctly pass the request context to the nested serializer
            request = self.context.get('request')
            serializer = ImageListSerializer(obj.parent, context={'request': request})
            return serializer.data
        return None
