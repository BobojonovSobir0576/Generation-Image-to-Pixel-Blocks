# views.py
from django.shortcuts import get_object_or_404
from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .models import ImageModel
from .serializers import ImageModelSerializer, ImageListSerializer
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import json
import cv2


class ImageUploadAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        request_body=ImageModelSerializer,
        operation_description="Create a new generate block image",
        tags=['Generate image'],
        responses={201: ImageModelSerializer(many=False)}
    )
    def post(self, request, *args, **kwargs):
        serializer = ImageModelSerializer(data=request.data)
        if serializer.is_valid():
            image_instance = serializer.save()
            response_data = {
                'id': image_instance.id,
                'image': request.build_absolute_uri(image_instance.image.url),
                'colors': image_instance.colors
            }
            return Response(response_data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(
        operation_description="Get image",
        tags=['Generate image'],
        responses={200: ImageModelSerializer(many=False)}
    )
    def get(self, request, *args, **kwargs):
        images = ImageModel.objects.all().last()
        serializer = ImageModelSerializer(images)
        return Response(serializer.data, status=status.HTTP_200_OK)


import concurrent.futures
class UpdateImageColors(APIView):
    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter(
                'limit_colors',
                openapi.IN_QUERY,
                description="Limit for the number of colors",
                type=openapi.TYPE_INTEGER
            ),
        ],
        tags=['Generate image'],
    )
    def get(self, request, image_id):
        image_instance = get_object_or_404(ImageModel, id=image_id)

        limit_colors = request.query_params.get('limit_colors', None)
        if limit_colors is None:
            serializer = ImageListSerializer(image_instance, context={'request': request})
            return Response(serializer.data, status=status.HTTP_200_OK)

        try:
            limit_colors = int(limit_colors)
        except ValueError:
            return Response({'error': 'Invalid limit_colors value. Please provide an integer.'},
                            status=status.HTTP_400_BAD_REQUEST)

        if limit_colors > len(image_instance.colors):
            return Response({'error': f'The number of unique colors in the image exceeds the limit of {limit_colors}'},
                            status=status.HTTP_400_BAD_REQUEST)

        image_path = image_instance.image.path
        colors = image_instance.colors[:limit_colors]

        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32)

        # Convert hex colors to RGB tuples
        rgb_colors = np.array([self.hex_to_rgb(color['hex']) for color in colors], dtype=np.float32)

        # Detect faces and apply selective color quantization
        try:
            updated_img_array = self.apply_selective_quantization(img_array, rgb_colors)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Save the updated image
        updated_image = Image.fromarray(updated_img_array.astype(np.uint8))
        updated_image.save(image_path)
        image_instance.colors = list(colors)
        image_instance.save()
        serializer = ImageListSerializer(image_instance, context={'request': request})
        return Response(serializer.data, status=status.HTTP_200_OK)

    @staticmethod
    def hex_to_rgb(hex_color):
        # Convert hexadecimal color string to RGB tuple
        return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))

    def apply_selective_quantization(self, img_array, color_list):
        # Convert image to grayscale for face detection
        gray_img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Load OpenCV's pre-trained Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Create a mask for face regions
        face_mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=bool)
        for (x, y, w, h) in faces:
            face_mask[y:y + h, x:x + w] = True

        # Apply color quantization with dithering to non-face regions
        non_face_region = np.where(~face_mask)
        face_region = np.where(face_mask)

        # Use multi-threading for parallel execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            non_face_quantization = executor.submit(self.update_image_colors_with_dithering, img_array[non_face_region],
                                                    color_list)
            face_quantization = executor.submit(self.update_image_colors, img_array[face_region], color_list)

        updated_img_array = np.copy(img_array)
        updated_img_array[non_face_region] = non_face_quantization.result()
        updated_img_array[face_region] = face_quantization.result()

        return updated_img_array

    def update_image_colors_with_dithering(self, img_array, color_list):
        # Dithering using Floyd-Steinberg algorithm
        img_array = img_array.reshape((-1, 3))
        for i in range(len(img_array)):
            old_pixel = img_array[i]
            new_pixel = self.find_closest_color(old_pixel, color_list)
            img_array[i] = new_pixel
            error = old_pixel - new_pixel
            if i + 1 < len(img_array):
                img_array[i + 1] += error * 7 / 16
            if i + 2 < len(img_array):
                img_array[i + 2] += error * 5 / 16
        return img_array.reshape((-1, 3))

    def update_image_colors(self, img_array, color_list):
        img_array = img_array.reshape((-1, 3))
        distances = np.sqrt(((img_array[:, None] - color_list) ** 2).sum(axis=2))
        closest_color_indices = np.argmin(distances, axis=1)
        updated_img_array = color_list[closest_color_indices]
        return updated_img_array.reshape((-1, 3))

    def find_closest_color(self, pixel_color, color_list):
        distances = np.sqrt(((pixel_color - color_list) ** 2).sum(axis=1))
        closest_color_index = np.argmin(distances)
        return color_list[closest_color_index]
