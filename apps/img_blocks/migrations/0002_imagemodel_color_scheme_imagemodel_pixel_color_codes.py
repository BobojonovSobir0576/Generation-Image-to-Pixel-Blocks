# Generated by Django 5.0.6 on 2024-06-29 09:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('img_blocks', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagemodel',
            name='color_scheme',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='imagemodel',
            name='pixel_color_codes',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
