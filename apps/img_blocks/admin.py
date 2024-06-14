from django.contrib import admin

from apps.img_blocks.models import ImageModel


class ImageModelAdmin(admin.ModelAdmin):
    list_display = ['id', 'parent']


admin.site.register(ImageModel, ImageModelAdmin)