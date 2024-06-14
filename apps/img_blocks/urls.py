from django.urls import path
from apps.img_blocks.views import *

urlpatterns = [
    path('upload/', ImageUploadAPIView.as_view(), name='image-upload'),
    path('update-colors/<int:image_id>', UpdateImageColors.as_view(), name='update_image_colors'),
    path('back/process/<int:id>', BackProcessViews.as_view(), name='back_process'),

]