from channels.generic.websocket import AsyncWebsocketConsumer
import json
import logging
from .backend import SpeechTranscriberBackend

logger = logging.getLogger(__name__)

class TranscriptionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add('transcription_group', self.channel_name)
        await self.accept()
        logger.info("WebSocket connected")

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard('transcription_group', self.channel_name)
        logger.info("WebSocket disconnected")

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            logger.info("Received binary audio data from client")
            # Forward the audio data to Deepgram via backend
            if SpeechTranscriberBackend.is_recording and SpeechTranscriberBackend.ws_ready and SpeechTranscriberBackend.ws:
                SpeechTranscriberBackend.ws.send(bytes_data, opcode=0x2)
        elif text_data:
            logger.info(f"Received text data: {text_data}")

    async def transcription_message(self, event):
        await self.send(text_data=json.dumps({'message': event['message']}))