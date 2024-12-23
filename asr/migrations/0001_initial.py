# Generated by Django 5.1.3 on 2024-12-01 15:09

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="AudioFile",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("file", models.FileField(upload_to="audio_files/")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
