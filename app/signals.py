# app/signals.py
import os
from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import User  # Import User model
from datetime import datetime

@receiver(post_save, sender=User)
def create_user_folder(sender, instance, created, **kwargs):

    if created:
        user_folder = os.path.join(settings.MEDIA_ROOT, f"user_{instance.id}")
        os.makedirs(user_folder, exist_ok=True)


        subfolders = ['chat', 'email', 'personal_info', 'whatsapp']
        for subfolder in subfolders:
            subfolder_path = os.path.join(user_folder, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)

