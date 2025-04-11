import logging
import threading
import json
from datetime import datetime
import pyaudio
import wave
from websocket import WebSocketApp
from openai import OpenAI
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from .models import Chat, AIChatHistory
from pathlib import Path
from text_manage_app import settings

logger = logging.getLogger(__name__)

# Configuration
SAMPLE_RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16

class SpeechTranscriberBackend:
    def __init__(self):
        self.is_recording = False
        self.ws = None
        self.ws_ready = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_thread = None
        self.audio_file_path = None
        self.speaker_names = {}
        self.transcripts = []
        self.channel_layer = get_channel_layer()
        logger.info("Initialized SpeechTranscriberBackend")

    def start_recording(self):
        self.stop_recording()
        ws_url = f"wss://api.deepgram.com/v1/listen?diarize=true&encoding=linear16&sample_rate={SAMPLE_RATE}"
        self.ws = WebSocketApp(ws_url,
                               header={"Authorization": f"Token {settings.DEEPGRAM_API_KEY}"},
                               on_open=self.on_ws_open,
                               on_message=self.on_ws_message,
                               on_error=self.on_ws_error,
                               on_close=self.on_ws_close)
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
                                      input=True, frames_per_buffer=CHUNK)
        media_dir = Path.cwd() / 'text_manage_app/media/VoiceRecording'
        self.audio_file_path = media_dir / f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        media_dir.mkdir(exist_ok=True)
        logger.info(f"Audio file path: {self.audio_file_path}")
        self.is_recording = True
        self.audio_thread = threading.Thread(target=self.save_audio, daemon=True)
        self.audio_thread.start()
        logger.info("Recording started")

    def stop_recording(self):
        self.is_recording = False
        self.ws_ready = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.ws:
            self.ws.close()
            self.ws = None
        if self.audio_thread:
            self.audio_thread.join()
        logger.info("Recording stopped")

    def save_audio(self):
        wf = wave.open(str(self.audio_file_path), 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)

        while self.is_recording and self.stream:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            wf.writeframes(data)
            if self.ws_ready and self.ws:
                self.ws.send(data, opcode=0x2)
        wf.close()
        logger.info(f"Audio saved to {self.audio_file_path}")

    def on_ws_open(self, ws):
        self.ws_ready = True
        logger.info("WebSocket connected")

    def on_ws_message(self, ws, message):
        data = json.loads(message)
        if 'channel' in data:
            alternatives = data['channel']['alternatives']
            if alternatives and alternatives[0]['transcript']:
                transcript = alternatives[0]['transcript']
                words = alternatives[0].get('words', [])
                if words:
                    speaker_id = words[0]['speaker']
                    speaker_name = self.speaker_names.get(speaker_id, f"Unknown_{speaker_id}")
                    display_text = f"{speaker_name}: {transcript}"
                    self.transcripts.append(display_text)

                    time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    Chat.objects.create(user_id=speaker_id, user_name=speaker_name, speech=transcript, time_stamp=time_stamp)
                    
                    async_to_sync(self.channel_layer.group_send)(
                        'transcription_group',
                        {'type': 'transcription_message', 'message': display_text}
                    )
                    logger.info(display_text)

    def on_ws_error(self, ws, error):
        logger.error(f"WebSocket Error: {error}")

    def on_ws_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket closed")

    def save_speaker_name(self, name):
        # Get the most recent speaker_id from the Chat model
        recent_chat = Chat.objects.order_by('-time_stamp').first()
        if not recent_chat:
            logger.warning("No recent speaker to rename")
            raise ValueError("No recent speaker found to rename")
        speaker_id = recent_chat.user_id
        self.speaker_names[speaker_id] = name
        Chat.objects.filter(user_id=speaker_id, user_name=f"Unknown_{speaker_id}").update(user_name=name)
        for i, transcript in enumerate(self.transcripts):
            if transcript.startswith(f"Unknown_{speaker_id}:"):
                self.transcripts[i] = f"{name}: {transcript.split(':', 1)[1].strip()}"
        logger.info(f"Speaker {speaker_id} renamed to {name}")
        return speaker_id

    def clear_data(self):
        self.transcripts = []
        self.speaker_names.clear()
        Chat.objects.all().delete()
        AIChatHistory.objects.all().delete()
        logger.info("Data cleared")

    def ask_ai(self, query):
        speaker_data = Chat.objects.all()
        ai_chat_data = AIChatHistory.objects.order_by('time_stamp')

        speaker_context = "Speaker conversation history:\n" + (
            "\n".join(f"{row.user_name}: {row.speech} (at {row.time_stamp})" for row in speaker_data)
            if speaker_data else "No speaker conversation history found.\n"
        )
        ai_chat_context = "AI chat history:\n" + (
            "\n".join(f"{row.role}: {row.content} (at {row.time_stamp})" for row in ai_chat_data)
            if ai_chat_data else "No AI chat history found.\n"
        )

        system_instruction = (
            "You are a helpful AI assistant. Use the provided speaker conversation history and AI chat history to answer the user's question."
        )
        messages = [
            {"role": "system", "content": f"{system_instruction}\n{speaker_context}\n{ai_chat_context}"},
            {"role": "user", "content": f"User Question: {query}\nToday's Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}
        ]

        client = OpenAI(api_key=settings.OPEN_AI_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, max_tokens=150, temperature=0.7
        )
        ai_message = response.choices[0].message.content.strip()

        time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        AIChatHistory.objects.create(role="user", content=query, time_stamp=time_stamp)
        AIChatHistory.objects.create(role="assistant", content=ai_message, time_stamp=time_stamp)

        logger.info(f"AI: {ai_message}")
        return ai_message

    def close(self):
        self.stop_recording()
        self.audio.terminate()
        logger.info("Backend closed")