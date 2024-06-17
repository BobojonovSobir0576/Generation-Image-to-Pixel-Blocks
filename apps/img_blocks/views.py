# views.py
import io
import os

from django.http import HttpResponse
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
import concurrent.futures
from io import BytesIO
from django.core.files.base import ContentFile


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
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(
        operation_description="Get image",
        tags=['Generate image'],
        responses={200: ImageListSerializer(many=False)}
    )
    def get(self, request, *args, **kwargs):
        images = ImageModel.objects.order_by('-id').first()
        if images:
            serializer = ImageListSerializer(images, context={'request': request})
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response({'detail': 'No images found'}, status=status.HTTP_404_NOT_FOUND)


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

        # Fetch the latest image
        latest_image = ImageModel.objects.filter(parent=None).order_by("-id").first()
        if not latest_image:
            return Response({'error': 'No image found to process.'}, status=status.HTTP_404_NOT_FOUND)

        image_path = latest_image.image.path
        colors = image_instance.main_colors[:limit_colors]

        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32)

        # Convert hex colors to RGB tuples
        rgb_colors = np.array([self.hex_to_rgb(color['hex']) for color in colors], dtype=np.float32)

        # Detect faces and apply selective color quantization
        try:
            updated_img_array = self.apply_selective_quantization(img_array, rgb_colors)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Save the updated image as a new image
        updated_image = Image.fromarray(updated_img_array.astype(np.uint8))
        new_image_io = BytesIO()
        updated_image.save(new_image_io, format='JPEG')
        new_image_content = ContentFile(new_image_io.getvalue(), name=f"{os.path.basename(image_path)}")

        # Create a new ImageModel instance with the updated image
        new_image_instance = ImageModel.objects.create(image=new_image_content, colors=list(colors), main_colors=image_instance.main_colors, parent=image_instance)
        new_image_instance.save()

        serializer = ImageListSerializer(new_image_instance, context={'request': request})
        return Response(serializer.data, status=status.HTTP_200_OK)

    @staticmethod
    def hex_to_rgb(hex_color):
        return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))

    def apply_selective_quantization(self, img_array, color_list):
        gray_img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        scale_factor = 0.25
        small_gray_img = cv2.resize(gray_img, None, fx=scale_factor, fy=scale_factor)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(small_gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        face_mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=bool)
        for (x, y, w, h) in faces:
            x, y, w, h = int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)
            face_mask[y:y + h, x:x + w] = True

        non_face_region = np.where(~face_mask)
        face_region = np.where(face_mask)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            non_face_future = executor.submit(self.update_image_colors, img_array[non_face_region], color_list)
            face_future = executor.submit(self.update_image_colors, img_array[face_region], color_list)

            updated_non_face = non_face_future.result()
            updated_face = face_future.result()

        updated_img_array = np.copy(img_array)
        updated_img_array[non_face_region] = updated_non_face
        updated_img_array[face_region] = updated_face

        return updated_img_array

    def update_image_colors(self, img_array, color_list):
        img_array = img_array.reshape((-1, 3))
        distances = np.linalg.norm(img_array[:, None] - color_list, axis=2)
        closest_color_indices = np.argmin(distances, axis=1)
        updated_img_array = color_list[closest_color_indices]
        return updated_img_array.reshape((-1, 3))

    @staticmethod
    def get_biggest_and_smallest_images():
        all_images = ImageModel.objects.all()
        biggest_image = max(all_images, key=lambda img: img.image.width * img.image.height)
        smallest_image = min(all_images, key=lambda img: img.image.width * img.image.height)
        return biggest_image, smallest_image


class BackProcessViews(APIView):
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Back to process",
        tags=['Back Process'],
        responses={200: ImageListSerializer(many=False)}
    )
    def get(self, request, id):
        images = get_object_or_404(ImageModel, id=id)
        serializer = ImageListSerializer(images, context={'request': request})
        return Response(serializer.data, status=status.HTTP_200_OK)
