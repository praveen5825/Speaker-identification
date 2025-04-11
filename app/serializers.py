from rest_framework import serializers
from .models import *
import uuid

class UserProfileSerializer(serializers.ModelSerializer):
    email = serializers.SerializerMethodField()
    name = serializers.SerializerMethodField()

    def get_email(self,obj):
        return obj.user.email if obj.user.email else ''

    def get_name(self,obj):
        return obj.user.name if obj.user.name else ''
    
    class Meta:
        model = UserProfile
        fields = ['photo', 'gender', 'dob','email', 'name']


class TextMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = TextMessage
        fields = ('id', 'text', 'timestamp')

class NotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Notification
        fields = ('id', 'message', 'timestamp', 'is_read')


class ContactSerializer(serializers.ModelSerializer):
    class Meta:
        model = Contact
        fields = ['id', 'user', 'name', 'phone_number', 'email', 'device_type', 'timestamp']
        extra_kwargs = {'user': {'read_only': True}} 

    def create(self, validated_data):
        user = self.context['request'].user
        validated_data['user'] = user 
        return super().create(validated_data)



class UserConnectionSerializer(serializers.ModelSerializer):

    class Meta:
        model = UserConnection
        fields = ['id', 'name', 'email', 'whatsapp_number', 'relation',]
        
    def validate(self, data):
        user = self.context['request'].user
        if UserConnection.objects.filter(user=user, name=data.get('name')).exists():
            raise serializers.ValidationError({"name": "A connection with this name already exists for this user"})
        
        if UserConnection.objects.filter(user=user, email=data.get('email')).exists():
            raise serializers.ValidationError({"email": "A connection with this email already exists for this user"})

        whatsapp_number = data.get('whatsapp_number')
        if whatsapp_number and UserConnection.objects.filter(user=user, whatsapp_number=whatsapp_number).exists():
            raise serializers.ValidationError({"whatsapp_number": "A connection with this WhatsApp number already exists for this user"})
            
        return data

class TranscriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Transcription
        fields = '__all__'


class ChatSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chat
        fields = ['id', 'user_id', 'user_name', 'speech', 'time_stamp']

class AIChatHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = AIChatHistory
        fields = ['id', 'role', 'content', 'time_stamp']
        


class TranscriptUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Transcript
        fields = ['id', 'user', 'full_text', 'speaker', 'text', 'line_order', 'transcript_group', 'created_at']
        read_only_fields = ['id', 'created_at', 'updated_at', 'transcript_group']
        
        extra_kwargs = { 
            "speaker": {"required": False},
            "text": {"required": False},
            "line_order": {"required": False}
        }
    def create(self, validated_data):
        full_text = validated_data.get('full_text', '')
        user = validated_data['user']
        transcript_group = str(uuid.uuid4())

        if full_text:
            lines = full_text.split('\n')
            transcript_instances = []
            for order, line in enumerate(lines, start=1):
                if line.strip():
                    try:
                        speaker, text = line.split(': ', 1)
                    except ValueError:
                        speaker = "Unknown"
                        text = line.strip()
                    transcript_instances.append(
                        Transcript(
                            user=user,
                            full_text=full_text,
                            speaker=speaker,
                            text=text,
                            line_order=order,
                            transcript_group=transcript_group
                        )
                    )
            Transcript.objects.bulk_create(transcript_instances)
            return transcript_instances[0]
        return Transcript.objects.create(**validated_data)