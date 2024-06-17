# models.py
from django.db import models
from jsonfield import JSONField
from PIL import Image
import numpy as np


class ImageModel(models.Model):
    image = models.ImageField(upload_to='images/')
    colors = models.JSONField(null=True, blank=True)
    main_colors = models.JSONField(null=True, blank=True)
    parent = models.ForeignKey("self", on_delete=models.CASCADE, null=True, blank=True, default=None)

    def __str__(self):
        return f'Image {self.id}'
