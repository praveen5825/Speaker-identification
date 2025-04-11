from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.exceptions import ValidationError
import os
from django.utils import timezone
from app.enums.all_enum import DeviceType 
import uuid



class User(AbstractUser):
    is_verified = models.BooleanField(default=False)
    name = models.CharField(max_length=255, null=True, blank=True)
    misc = models.JSONField(default = dict)
    def __str__(self):
        return str(self.id)+"-"+str(self.email)

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    photo = models.ImageField(upload_to='profile_photos/', null=True, blank=True)
    gender = models.CharField(max_length=10, null=True, blank=True)
    dob = models.DateField(null=True, blank=True)
    def __str__(self):
        return str(self.user.email)



class Contact(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Link to User
    name = models.CharField(max_length=255)
    phone_number = models.CharField(max_length=20,)
    email = models.EmailField( null=True, blank=True)
    device_type = models.CharField(max_length=10,
        default=DeviceType.ANDROID,
        choices=DeviceType.choices,
        blank=True,)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.phone_number})"


def user_resume_upload_to(instance, filename):
    user_folder = f"user_{instance.user.id}"
    personal_info_folder = os.path.join(user_folder, "personal_info")
    return os.path.join(personal_info_folder, filename)

class UserDocuments(models.Model):
    user = models.ForeignKey('User', on_delete=models.CASCADE) 
    file = models.FileField(upload_to=user_resume_upload_to) 
    extracted_text = models.TextField() 
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Resume of User {self.user.id} - {self.created_at}"


class TextMessage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    text = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return str(self.user.email)


class OTP(models.Model):
    email = models.EmailField(unique=True)
    otp = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)
    def is_expired(self):
        return timezone.now() > self.created_at + timezone.timedelta(minutes=5)  # OTP valid for 5 minutes
    def __str__(self):
        return str(self.email)
    
class TextMessageHistory(models.Model):
    text_message = models.ForeignKey(TextMessage, on_delete=models.DO_NOTHING)
    text = models.TextField()
    created_by = models.ForeignKey(User, related_name='created_histories', on_delete=models.CASCADE)
    updated_by = models.ForeignKey(User, related_name='updated_histories', on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return f"History of {self.text_message.user.email} at {self.timestamp}"
    
class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f'Notification for {self.user.username}: {self.message}'


class Speaker(models.Model):
    name = models.CharField(max_length=100, unique=True)
    audio_file = models.FileField(upload_to="AudioFile/")

    def __str__(self):
        return self.name

class RealTimeTranscription(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    speaker_id = models.CharField(max_length=100, null=True, blank=True)
    text = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Speaker {self.speaker_id}: {self.text[:50]}"



class UserConnection(models.Model): 
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="connections")
    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    whatsapp_number = models.CharField(max_length=20)
    relation = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return self.name



class Transcription(models.Model):
    speaker = models.CharField(max_length=255)
    text = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.speaker}: {self.text[:50]}"

class Chat(models.Model):
    user_id = models.IntegerField()
    user_name = models.CharField(max_length=100)
    speech = models.TextField()
    time_stamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'Chats'
        ordering = ['time_stamp']

class AIChatHistory(models.Model):
    role = models.CharField(max_length=20)
    content = models.TextField()
    time_stamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'AIChatHistory'
        ordering = ['time_stamp']
        
        
        

class Transcript(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="transcripts")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    full_text = models.TextField(blank=True, null=True) 
    speaker = models.CharField(max_length=100)
    text = models.TextField()
    line_order = models.PositiveIntegerField()
    transcript_group = models.CharField(max_length=50, default=uuid.uuid4)

    class Meta:
        ordering = ['transcript_group', 'line_order']
        indexes = [
            models.Index(fields=['transcript_group', 'created_at']),
        ]

    def __str__(self):
        return f"{self.speaker}: {self.text} (Group: {self.transcript_group})"