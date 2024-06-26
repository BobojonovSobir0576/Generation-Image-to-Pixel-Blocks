from django.urls import path
from apps.img_blocks.views import *

urlpatterns = [
    path('upload/', ImageUploadAPIView.as_view(), name='image-upload'),
    path('images/<uuid:image_id>/', ImageUploadAPIView.as_view(), name='get_user_images'),
    path('color/update/<uuid:image_id>', UpdateColorsViews.as_view()),
    path('update-colors/<uuid:image_id>', UpdateImageColors.as_view(), name='update_image_colors'),
    path('back/process/<int:id>', BackProcessViews.as_view(), name='back_process'),

    path('grouped/colors/<uuid:image_id>', GroupedColorsViews.as_view()),
    path('return/own-colors/<uuid:image_id>', ReturningOwnColorsViews.as_view())
]