from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/transcription/$', consumers.TranscriptionConsumer.as_asgi()),
]