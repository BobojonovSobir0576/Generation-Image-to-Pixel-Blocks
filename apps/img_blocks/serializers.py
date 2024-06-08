from rest_framework import serializers
from .models import ImageModel
from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import io
import time


class ImageModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageModel
        fields = ['id', 'image', 'colors']
        read_only_fields = ['colors']

    def create(self, validated_data):
        start_time = time.time()

        image_instance = ImageModel.objects.create(image=validated_data['image'])

        image_path = image_instance.image.path
        img = Image.open(image_path).convert('RGB')

        # Resize image to reduce processing time if needed
        max_size = 1024
        if max(img.width, img.height) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)

        img_array = np.array(img) / 255.0

        # Apply mosaic effect
        mosaic_img = self.apply_mosaic_effect(img_array, 10)

        # Convert back to image and save
        mosaic_img_pil = Image.fromarray((mosaic_img * 255).astype(np.uint8))

        buffer = io.BytesIO()
        mosaic_img_pil.save(buffer, format='JPEG')
        buffer.seek(0)
        img_pil = Image.open(buffer)

        colors = self.get_colors_hex(img_pil)

        image_instance.colors = colors
        image_instance.save()

        end_time = time.time()
        print(f"Processing Time: {end_time - start_time} seconds")

        return image_instance

    def apply_mosaic_effect(self, img, block_size):
        # Pad the image to make its dimensions divisible by the block size
        pad_height = (block_size - img.shape[0] % block_size) % block_size
        pad_width = (block_size - img.shape[1] % block_size) % block_size
        padded_img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')

        n, m, _ = padded_img.shape

        # Create an array of the block averages
        reshaped_img = padded_img.reshape(n // block_size, block_size, m // block_size, block_size, 3)
        block_averages = reshaped_img.mean(axis=(1, 3))

        # Use the block averages to fill in the mosaic image
        mosaic_img = np.repeat(np.repeat(block_averages, block_size, axis=0), block_size, axis=1)

        return mosaic_img[:img.shape[0], :img.shape[1]]

    def get_colors_hex(self, img, n_colors=10):
        img_rgb = img.convert('RGB')
        img_array = np.array(img_rgb).reshape((-1, 3))

        kmeans = MiniBatchKMeans(n_clusters=n_colors, n_init=10, max_iter=300)
        kmeans.fit(img_array)
        unique_colors = kmeans.cluster_centers_.astype(int)

        color_dict = [{"id": i + 1, "name": f"Color {i + 1}", "hex": '#' + ''.join(f'{c:02x}' for c in color)} for
                      i, color in enumerate(unique_colors)]
        return color_dict
class ImageListSerializer(serializers.ModelSerializer):

    class Meta:
        model = ImageModel
        fields = '__all__'
