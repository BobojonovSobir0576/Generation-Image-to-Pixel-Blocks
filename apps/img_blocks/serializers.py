from rest_framework import serializers
from rest_framework.views import APIView

from .models import ImageModel
from PIL import Image, ImageDraw, ImageStat, ImageFont
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import io
import time
from pixelator import Pixelator
from io import BytesIO
from django.core.files.base import ContentFile

from .utils import cut_image_and_save_colors_as_json


class ImageModelSerializer(serializers.ModelSerializer):
    image = serializers.ImageField()

    class Meta:
        model = ImageModel
        fields = ['uuid', 'image', 'colors', 'user_identifier']
        read_only_fields = ['colors']



    def create(self, validated_data):
        image_instance = ImageModel.objects.create(image=validated_data['image'], user_identifier=self.context.get('user_identifier'))

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
        colors_original_without_sorting = self.get_colors_hex_without_sorting(pixelated_img_original, n_colors=256)
        # Update ImageModel instance with colors
        image_instance.colors = colors_original_without_sorting
        image_instance.main_colors = colors_original_without_sorting
        image_instance.save()

        return image_instance

    def get_colors_hex_without_sorting(self, img, n_colors=256):
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
        fields = ['uuid', 'image', 'colors', "main_colors", 'parent', 'user_identifier']

    def get_parent(self, obj):
        if obj.parent is not None:
            # Correctly pass the request context to the nested serializer
            request = self.context.get('request')
            serializer = ImageListSerializer(obj.parent, context={'request': request})
            return serializer.data
        return None


class ColorSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    name = serializers.CharField()
    hex = serializers.CharField()


class UpdateImageModelSerializer(serializers.ModelSerializer):
    colors = ColorSerializer(many=True)

    class Meta:
        model = ImageModel
        fields = ['id', 'uuid', 'image', 'colors', 'user_identifier']

    def update(self, instance, validated_data):
        colors_data = validated_data.pop('colors', None)
        if colors_data:
            instance.colors = colors_data
        instance.save()
        return instance


class GetSchemasImageSerializers(serializers.Serializer):
    schemas = serializers.SerializerMethodField()

    def get_schemas(self, obj):
        image_path = obj.image.path
        return cut_image_and_save_colors_as_json(image_path, 9, 15,)
