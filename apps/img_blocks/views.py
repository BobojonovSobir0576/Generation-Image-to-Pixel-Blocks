import os

from django.shortcuts import get_object_or_404
from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .models import ImageModel
from .serializers import ImageModelSerializer, ImageListSerializer, UpdateImageModelSerializer
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import cv2
import concurrent.futures
from io import BytesIO
from django.core.files.base import ContentFile
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import uuid


class ImageUploadAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        request_body=ImageModelSerializer,
        operation_description="Create a new generate block image",
        tags=['Generate image'],
        responses={201: ImageModelSerializer(many=False)}
    )
    @method_decorator(csrf_exempt)
    def post(self, request, *args, **kwargs):
        user_identifier = request.COOKIES.get('user_identifier')
        if not user_identifier:
            user_identifier = str(uuid.uuid4())
            response = Response()
            response.set_cookie('user_identifier', user_identifier)
        else:
            response = None

        data = request.data.copy()
        serializer = ImageModelSerializer(data=data, context={'request': request, 'user_identifier': user_identifier})
        if serializer.is_valid():
            serializer.save()
            if response:
                response.data = serializer.data
                response.status_code = status.HTTP_201_CREATED
                return response
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @swagger_auto_schema(
        operation_description="Get user's images",
        tags=['Retrieve image'],
        responses={200: ImageListSerializer(many=True)}
    )
    def get(self, request, *args, **kwargs):
        user_identifier = request.headers.get('user-identifier')
        if not user_identifier:
            return Response({'detail': 'User identifier not found'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            images = ImageModel.objects.filter(user_identifier=user_identifier)
            serializer = ImageListSerializer(images, many=True, context={'request': request})
            return Response(serializer.data, status=status.HTTP_200_OK)
        except ImageModel.DoesNotExist:
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
        user_identifier = request.headers.get('user-identifier')
        if not user_identifier:
            return Response({'detail': 'User identifier not found'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            image_instance = ImageModel.objects.filter(uuid=image_id, user_identifier=user_identifier).first()
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
            latest_image = (ImageModel.objects.filter(user_identifier=user_identifier).filter(parent=None,)
                            .order_by("-id").first())
            if not latest_image:
                return Response({'error': 'No image found to process.'}, status=status.HTTP_404_NOT_FOUND)

            image_path = latest_image.image.path
            colors = image_instance.main_colors[:limit_colors]

            # Open image and process it
            with Image.open(image_path) as img:
                img_array = np.array(img.convert('RGB'))

            # Convert hex colors to RGB tuples
            rgb_colors = np.array([self.hex_to_rgb(color['hex']) for color in colors], dtype=np.float32)

            # Detect faces and apply selective color quantization
            updated_img_array = self.apply_selective_quantization(img_array, rgb_colors)

            # Save the updated image as a new image
            updated_image = Image.fromarray(updated_img_array.astype(np.uint8))
            new_image_io = BytesIO()
            updated_image.save(new_image_io, format='JPEG')
            new_image_content = ContentFile(new_image_io.getvalue(), name=f"{latest_image.image.name}")

            # Create a new ImageModel instance with the updated image
            new_image_instance = ImageModel.objects.create(image=new_image_content, colors=list(colors),
                                                           main_colors=image_instance.main_colors,
                                                           parent=image_instance,
                                                           user_identifier=user_identifier)
            new_image_instance.save()

            serializer = ImageListSerializer(new_image_instance, context={'request': request})
            response = Response(serializer.data, status=status.HTTP_200_OK)
            response.set_cookie('user_identifier', user_identifier)
            return response

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @staticmethod
    def hex_to_rgb(hex_color):
        return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))

    def apply_selective_quantization(self, img_array, color_list):
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        scale_factor = 0.25
        small_gray_img = cv2.resize(gray_img, None, fx=scale_factor, fy=scale_factor)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(small_gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        face_mask = np.zeros(img_array.shape[:2], dtype=bool)
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


class UpdateColorsViews(APIView):
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Update colors in an Image instance",
        responses={200: ImageModelSerializer()},
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'color_id': openapi.Schema(type=openapi.TYPE_INTEGER),
                'new_color_hex': openapi.Schema(type=openapi.TYPE_STRING),
            },
            required=['color_id', 'new_color_hex'],
        ),
    )
    def put(self, request, image_id):
        user_identifier = request.headers.get('user-identifier')
        if not user_identifier:
            return Response({'detail': 'User identifier not found'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            image_instance = ImageModel.objects.filter(uuid=image_id, user_identifier=user_identifier).first()

            color_id = request.data.get('color_id')
            new_color_hex = request.data.get('new_color_hex')

            # Check if color_id and new_color_hex are provided
            if not color_id or not new_color_hex:
                return Response({'error': 'color_id and new_color_hex are required fields'},
                                status=status.HTTP_400_BAD_REQUEST)

            # Find the color with the given color_id in image_instance.colors
            old_color_hex = None
            for color in image_instance.colors:
                if color['id'] == color_id:
                    old_color_hex = color['hex']
                    color['hex'] = new_color_hex
                    break

            if not old_color_hex:
                return Response({'error': f'Color with ID {color_id} not found'}, status=status.HTTP_404_NOT_FOUND)

            # Load the image
            image_path = image_instance.image.path
            image = Image.open(image_path)
            image = image.convert('RGBA')

            # Convert hex colors to RGBA
            old_color_rgba = tuple(int(old_color_hex[i:i+2], 16) for i in (1, 3, 5)) + (255,)
            new_color_rgba = tuple(int(new_color_hex[i:i+2], 16) for i in (1, 3, 5)) + (255,)

            # Change the pixels
            data = image.getdata()
            new_data = []
            for item in data:
                if item == old_color_rgba:
                    new_data.append(new_color_rgba)
                else:
                    new_data.append(item)
            image.putdata(new_data)

            # Convert to RGB mode before saving as JPEG
            image = image.convert('RGB')

            # Create the directory if it does not exist
            new_image_dir = os.path.dirname(f"/media/images/updated_{os.path.basename(image_path)}")
            os.makedirs(new_image_dir, exist_ok=True)

            # Save the updated image
            new_image_path = f"/media/images/updated_{os.path.basename(image_path)}"
            image.save(new_image_path, 'JPEG')
            image_instance.image = new_image_path

            # Save the updated instance
            image_instance.save()

            # Serialize and return the updated image data
            serializer = ImageModelSerializer(image_instance)
            return Response(serializer.data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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


class GroupedColorsViews(APIView):
    permission_classes = [AllowAny]  # Define your permissions here

    @swagger_auto_schema(
        operation_description="Grouped colors",
        tags=['Grouped colors'],
        responses={200: ImageListSerializer(many=True)}
    )
    def get(self, request, image_id):
        user_identifier = request.headers.get('user-identifier')

        if not user_identifier:
            return Response({'detail': 'User identifier not found'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            image_instance = ImageModel.objects.filter(uuid=image_id, user_identifier=user_identifier).first()
            if not image_instance:
                return Response({'detail': 'Image not found'}, status=status.HTTP_404_NOT_FOUND)

            # Retrieve colors from JSONField
            colors = image_instance.colors

            # Convert hex to RGB
            def hex_to_rgb(hex):
                hex = hex.lstrip('#')
                return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))

            # Extract RGB values from the JSONField 'colors'
            rgb_colors = [hex_to_rgb(color['hex']) for color in colors]

            # Convert to numpy array
            rgb_array = np.array(rgb_colors)

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=10)  # Adjust the number of clusters as needed
            kmeans.fit(rgb_array)
            labels = kmeans.labels_

            # Group colors by cluster labels
            grouped_colors = {}
            for i, label in enumerate(labels):
                if label not in grouped_colors:
                    grouped_colors[label] = []
                grouped_colors[label].append(colors[i])

            # Sort colors within each cluster
            def sort_colors_by_rgb(colors):
                return sorted(colors, key=lambda c: hex_to_rgb(c['hex']))

            sorted_grouped_colors = []
            for cluster_colors in grouped_colors.values():
                sorted_grouped_colors.extend(sort_colors_by_rgb(cluster_colors))

            # Update the image instance with the sorted grouped colors
            image_instance.colors = sorted_grouped_colors
            image_instance.save()

            # Serialize and return the updated image data
            serializer = ImageListSerializer(image_instance, context={'request': request})
            return Response(serializer.data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ReturningOwnColorsViews(APIView):
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Returning own colors",
        tags=['Returning own colors'],
        responses={200: ImageListSerializer(many=True)}
    )
    def get(self, request, image_id):
        user_identifier = request.headers.get('user-identifier')
        if not user_identifier:
            return Response({'detail': 'User identifier not found'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            image_instance = ImageModel.objects.filter(uuid=image_id, user_identifier=user_identifier).first()

            image_instance.colors = image_instance.main_colors[:len(image_instance.colors)]
            image_instance.save()
            serializers = ImageListSerializer(image_instance, context={'request': request})
            return Response(serializers.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)