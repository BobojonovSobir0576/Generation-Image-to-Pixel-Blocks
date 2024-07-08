from rest_framework import serializers
from rest_framework.views import APIView

from .models import ImageModel, ImageSchemas
from PIL import Image, ImageDraw, ImageStat, ImageFont
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from io import BytesIO
from django.core.files.base import ContentFile
from .utils import cut_image_and_save_colors_as_json


class ImageModelSerializer(serializers.ModelSerializer):
    image = serializers.ImageField()
    width = serializers.IntegerField(write_only=True)
    height = serializers.IntegerField(write_only=True)
    user_identifier = serializers.UUIDField(required=False)

    class Meta:
        model = ImageModel
        fields = ['uuid', 'image', 'colors', 'user_identifier', 'width', 'height']
        read_only_fields = ['colors']

    def create(self, validated_data):
        width = validated_data['width']
        height = validated_data['height']
        # Resize the image
        resized_image_content = self.resize_image(validated_data['image'], width, height)

        image_instance = ImageModel.objects.create(image=resized_image_content,
                                                   user_identifier=self.context.get('user_identifier'))

        # Open and resize original image if needed (optional)
        img_original = Image.open(image_instance.image.path)
        max_size = 512
        if max(img_original.width, img_original.height) > max_size:
            img_original.thumbnail((max_size, max_size), Image.LANCZOS)

        # Pixelate the original image
        block_size = 1
        pixelated_img_original = self.pixelate_rgb(img_original, block_size)

        # Convert image to RGB mode if necessary
        if pixelated_img_original.mode != 'RGB':
            pixelated_img_original = pixelated_img_original.convert('RGB')

        # Save pixelated image temporarily in memory
        pixelated_img_original_io = BytesIO()
        pixelated_img_original.save(pixelated_img_original_io, format='JPEG')
        pixelated_img_original_content = ContentFile(pixelated_img_original_io.getvalue(), 'pixel.jpg')

        # Save the pixelated image to image_instance.image
        image_instance.image.save('pixel.jpg', pixelated_img_original_content)
        image_instance.main_image = image_instance.image

        # Calculate colors and update ImageModel instance
        colors_original_with_count = self.get_colors_hex_with_count(pixelated_img_original, n_colors=256)
        image_instance.colors = colors_original_with_count
        image_instance.main_colors = colors_original_with_count
        image_instance.save()

        return image_instance

    def resize_image(self, image_field, width, height):
        # Open the image from the uploaded image field
        image = Image.open(image_field)

        # Resize the image to the specified width and height
        resized_image = image.resize((width, height), Image.LANCZOS)

        # Save the resized image to a temporary BytesIO object
        image_io = BytesIO()
        resized_image.save(image_io, format='JPEG')

        # Create a ContentFile from the BytesIO object
        resized_image_content = ContentFile(image_io.getvalue(), 'resized.jpg')

        return resized_image_content

    def get_colors_hex_with_count(self, img, n_colors=256):
        img_rgb = img.convert('RGB')
        img_array = np.array(img_rgb)
        img_flat = img_array.reshape(-1, 3)

        # Apply color quantization (e.g., K-means clustering)
        kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(img_flat)

        # Get cluster centers (colors) and counts
        labels = kmeans.predict(img_flat)
        cluster_centers = kmeans.cluster_centers_.astype(int)

        # Count occurrences of each color
        color_counts = {}
        for label in np.unique(labels):
            count = np.count_nonzero(labels == label)
            color_counts[label] = count

        # Generate color dictionary with hex values and counts
        color_dict = []
        for i, (color, count) in enumerate(zip(cluster_centers, color_counts.values())):
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
            color_dict.append({
                "id": i + 1,
                "name": f"Color {i + 1}",
                "hex": hex_color,
                "count": count
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
                block = img_array[x:x + block_size, y:y + block_size]
                mean_color = block.mean(axis=(0, 1)).astype(np.uint8)
                img1[x:x + block_size, y:y + block_size] = mean_color

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


class SchemasListSerializers(serializers.ModelSerializer):

    class Meta:
        model = ImageSchemas
        fields = "__all__"


class ImagePixelChangeSerializer(serializers.ModelSerializer):
    block_size = serializers.IntegerField(write_only=True)
    user_identifier = serializers.UUIDField(required=False)

    class Meta:
        model = ImageModel
        fields = ['uuid', 'image', 'colors', 'user_identifier', 'block_size']

    def update(self, instance, validated_data):
        block_size = validated_data.pop('block_size', 0)
        if instance.color_image:
            img_original = Image.open(instance.color_image.path)
        else:
            img_original = Image.open(instance.main_image.path)

        max_size = 512
        if max(img_original.width, img_original.height) > max_size:
            img_original.thumbnail((max_size, max_size), Image.LANCZOS)

        pixelated_img_original = self.pixelate_rgb(img_original, block_size)

        # Convert image to RGB mode if necessary
        if pixelated_img_original.mode != 'RGB':
            pixelated_img_original = pixelated_img_original.convert('RGB')

        # Save pixelated image temporarily in memory
        pixelated_img_original_io = BytesIO()
        pixelated_img_original.save(pixelated_img_original_io, format='JPEG')
        pixelated_img_original_content = ContentFile(pixelated_img_original_io.getvalue(), 'pixel.jpg')
        # Save the pixelated image to image_instance.image
        instance.image.save('pixel.jpg', pixelated_img_original_content)

        instance.save()
        return instance

    def pixelate_rgb(self, img, block_size):
        # Convert image to numpy array for easier manipulation
        img_array = np.array(img)
        n, m, _ = img_array.shape
        img_array = img_array[:n - n % block_size, :m - m % block_size, :]
        img1 = np.zeros((img_array.shape), dtype=np.uint8)

        for x in range(0, img_array.shape[0], block_size):
            for y in range(0, img_array.shape[1], block_size):
                block = img_array[x:x + block_size, y:y + block_size]
                mean_color = block.mean(axis=(0, 1)).astype(np.uint8)
                img1[x:x + block_size, y:y + block_size] = np.tile(mean_color, (block_size, block_size, 1))

        # Convert the pixelated array back to an image
        pixelated_img = Image.fromarray(img1)

        return pixelated_img
