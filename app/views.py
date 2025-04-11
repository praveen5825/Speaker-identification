from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from .models import UserProfile, TextMessage, OTP, User, TextMessageHistory, Notification,Contact,Speaker,Chat,AIChatHistory,Transcription
from django.contrib.auth import authenticate
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import *
from .backend import SpeechTranscriberBackend
from .utility import send_otp,extract_text_from_doc,extract_text_from_pdf,get_file_type
import json, traceback
from openai import OpenAI
from text_manage_app import settings
from .serializers import ContactSerializer
from django.db.models import Q
from rest_framework.parsers import MultiPartParser,FormParser,JSONParser
import re
import os
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from scipy.spatial.distance import cosine
from deepgram import Deepgram
import time
import logging
from django.core.cache import cache
from django.shortcuts import render



logger = logging.getLogger(__name__)
backend = SpeechTranscriberBackend()


class RegisterUserView(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        account_types = ('email','gmail')
        try:
            request_data = request.data if isinstance(request.data,dict) else json.loads(request.data)
            name = request_data.get('name','').strip()
            email = request_data.get('email','').strip()
            password = request_data.get('password','').strip()
            account_type = request_data.get('account_type','').strip()
            if not all([email,password]):
                raise ValueError('Email & Password required feilds.')
            if account_type not in account_types:
                raise ValueError(f'account_type required field , It should be `{"/".join(account_types)}` only.')
            user_obj = User.objects.filter(email=email,username=email)

            if account_type == 'gmail':
                if user_obj.filter(is_active=True,is_verified=True).exists():
                    user = user_obj.last()
                elif user_obj.filter(is_active=True,is_verified=False).exists():
                    user = user_obj.last()
                    user.is_active =True
                    user.is_verified =True
                    user.save()
                else:
                    # misc data
                    request_data.pop("password")
                    request_data.pop("email")
                    request_data.pop("name")
                    misc = request_data

                    user = User.objects.create_user(name=name,
                                                    email=email,
                                                    username=email,
                                                    password=password,
                                                    is_active=True,
                                                    is_verified=True,
                                                    misc=misc)
                    UserProfile.objects.create(user=user)

                    refresh = RefreshToken.for_user(user)
                    return Response({
                        'token': str(refresh.access_token),
                        'message': 'Email verified.','status':200}, status=status.HTTP_200_OK)

            if user_obj.filter(is_active=True,is_verified=True).exists():
                return Response({'message': 'Email already exists.','status':400}, status=status.HTTP_400_BAD_REQUEST)
            elif user_obj.filter(is_active=True,is_verified=False).exists():
                user = user_obj.last()
            else:
                # misc data
                request_data.pop("password")
                request_data.pop("email")
                request_data.pop("name")
                misc = request_data

                user = User.objects.create_user(name=name,
                                                email=email,
                                                username=email,
                                                password=password,
                                                is_active=True,
                                                is_verified=False,
                                                misc=misc)
                UserProfile.objects.create(user=user)
                
            otp_sent = send_otp(email)
            if not otp_sent:
                user.delete()
                return Response({'message':'OTP not sent','status':400}, status=status.HTTP_400_BAD_REQUEST)
            
            return Response({'message': f'OTP sent on email `{email}`, please verify.','status':201}, status=status.HTTP_201_CREATED)
        except Exception as E:
            return Response({'message': f'{traceback.format_exc()}','status':400}, status=status.HTTP_400_BAD_REQUEST)


class OTPVerificationView(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        try:
            otp = request.data.get('otp')
            email = request.data.get('email')
            if not all([email,otp]):
                raise ValueError('Email & OTP required feilds.')

            otp_record = OTP.objects.filter(email=email,otp=otp)
            if not otp_record.exists():
                return Response({'message': 'Invalid OTP','status':400}, status=status.HTTP_400_BAD_REQUEST)
            if otp_record.last().is_expired():
                return Response({'message': 'OTP has expired.','status':400}, status=status.HTTP_400_BAD_REQUEST)

            try:
                user = User.objects.get(email=email)
            except User.DoesNotExist:
                user = User.objects.create(email=email,username=email)

            if not request.data.get("is_resetpassword") in ('true','True','TRUE',True):
                user.is_active = True
                user.is_verified = True
                user.save()

            otp_record.delete()
            refresh = RefreshToken.for_user(user)
            return Response({
                'token': str(refresh.access_token),
                'message': 'OTP verified successfully.','status':200}, status=status.HTTP_200_OK)

        except Exception as E:
            return Response({'message': f'{E}','status':400}, status=status.HTTP_400_BAD_REQUEST)


class SignInView(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        try:
            email = request.data.get('email','').strip()
            if not email:
                raise ValueError('Email required feild.')
            otp_sent = send_otp(email)
            if not otp_sent:
                return Response({'message':'OTP not sent','status':400}, status=status.HTTP_400_BAD_REQUEST)
            
            return Response({'message': f'OTP sent on email `{email}`, please verify.','status':201}, status=status.HTTP_201_CREATED)
        except Exception as E:
            return Response({'message': f'{E}','status':200}, status=status.HTTP_400_BAD_REQUEST)

class LogInView(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        try:
            email = request.data.get('email','').strip()
            password = request.data.get('password','').strip()

            if not all([email,password]):
                raise ValueError('Email & Password required feilds.')
            user = authenticate(username=email, password=password)
            if not user or not user.is_active:
                return Response({'message': 'Invalid credentials','status':401},
                                status=status.HTTP_401_UNAUTHORIZED)
                
            if not all([user.is_verified, user.is_active]):
                    return Response({'message': f'OTP sent on email `{email}`, please verify.','status':201}, status=status.HTTP_201_CREATED)

            refresh = RefreshToken.for_user(user)
            
            return Response({
                'token': str(refresh.access_token),
                'message': 'Login successfully.','status':200
            })
            
        except Exception as E:
            return Response({'message': f'{E}'}, status=status.HTTP_400_BAD_REQUEST)

class UserProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user_profile = request.user.userprofile
        serializer = UserProfileSerializer(user_profile)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request):
        try:
            user_profile = request.user.userprofile

            if 'photo' in request.data:
                user_profile.photo = request.data['photo']
            if 'gender' in request.data:
                user_profile.gender = request.data['gender']
            if 'dob' in request.data:
                user_profile.dob = request.data['dob']

            if request.data.get('password'):
                request.user.set_password(request.data.get('password'))

            if request.data.get('name'):
                request.user.name = request.data.get('name')
                
            request.user.save()
            user_profile.save()
            
            return Response({'message': 'Profile updated successfully.','status':200}, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response({'message': str(e),'status':400}, status=status.HTTP_400_BAD_REQUEST)
        
class TextMessageView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            text_messages = TextMessage.objects.filter(user=request.user)
            serializer = TextMessageSerializer(text_messages, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e),'status':400}, status=status.HTTP_400_BAD_REQUEST)
                                   
    def post(self, request):
        try:
            text = request.data.get('text', '').strip()
            if not text:
                return Response({'message': '`text` required field!','status':400}, status=status.HTTP_400_BAD_REQUEST)

            text_message = TextMessage.objects.create(user=request.user, text=text)
            TextMessageHistory.objects.create(
                text_message=text_message,
                text=text,
                created_by=request.user,
                updated_by=request.user)

            return Response({'message': 'Text message created successfully.','status':200}, status=status.HTTP_201_CREATED)
        
        except Exception as e:
            return Response({'message': str(e),'status':400}, status=status.HTTP_400_BAD_REQUEST)

class PushNotificationView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        if not any([request.user.is_superuser,request.user.is_staff]):
            return Response({'message': 'You do not have permission to perform this action.'
                             ,'status':403}, status=status.HTTP_403_FORBIDDEN)
        email = request.data.get('email','').strip()
        message = request.data.get('message','').strip()

        if not all([email,message]):
            raise ValueError('Email & Message required feilds.')

        try:
            user = User.objects.get(username=email,email=email,is_verified=True)

            # Create the notification
            notification = Notification.objects.create(user=user, message=message)
            return Response({'message': 'Notification pushed successfully.', 'status':201}, status=status.HTTP_201_CREATED)

        except User.DoesNotExist:
            return Response({'message': 'User not found.','status':404}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({'message': str(e),'status':400}, status=status.HTTP_400_BAD_REQUEST)
    
    def put(self, request):
        try:
            n_id = request.data.get("id")
            notification_obj = Notification.objects.filter(id=n_id)
            if not notification_obj.exists():
                return Response({'message': 'Invalid Notification Id.','status':404}, status=status.HTTP_404_NOT_FOUND)
            notification_obj.update(is_read=True)
            return Response({'message': 'Notification marked as read successfully.','status':200}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'message': str(e),'status':400}, status=status.HTTP_400_BAD_REQUEST)

class GetNotificationsView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request):
        user = request.user
        notifications = Notification.objects.filter(user=user)
        serializer = NotificationSerializer(notifications, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

class ForgotPasswordView(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        try:
            request_data = request.data if isinstance(request.data,dict) else json.loads(request.data)
            email = request_data.get('email','').strip()
            if not email:
                raise ValueError('Email required feild.')
            user_obj = User.objects.filter(email=email,username=email)
            if not user_obj.exists():
                return Response({'message': 'Invalid Email !!!','status':400}, status=status.HTTP_400_BAD_REQUEST)                

            otp_sent = send_otp(email)            
            return Response({'message': f'OTP sent on email `{email}`, please verify.','status':201}, status=status.HTTP_201_CREATED)
        except Exception as E:
            return Response({'message': f'{traceback.format_exc()}','status':400}, status=status.HTTP_400_BAD_REQUEST)

class FCMNotificationView(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        from pyfcm import FCMNotification
        try:        
            request_data = request.data if isinstance(request.data,dict) else json.loads(request.data)
            device_tokens = request_data.get('device_tokens_list',[])
            title = request_data.get('title','')
            message = request_data.get('message','')
            if not device_tokens:
                raise ValueError('Device Tokens required feild.')

            push_service = FCMNotification(api_key=settings.FIREBASE_SERVER_KEY) 
            result = push_service.notify_multiple_devices(registration_ids=device_tokens, 
                                                        message_title=title, 
                                                        message_body=message) 
            
            return Response({'message': f'Notified successfully.','status':201}, status=status.HTTP_200_OK)
        except Exception as E:
            return Response({'message': f'{traceback.format_exc()}','status':400}, status=status.HTTP_400_BAD_REQUEST)



class DeactivateAccount(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        try:
            request.user.is_active = False
            request.user.is_verified = False
            request.user.save()
            
            return Response({'message': f'User Deactivated successfully.','status':200}, status=status.HTTP_200_OK)
        except Exception as E:
            return Response({'message': f'{traceback.format_exc()}','status':400}, status=status.HTTP_400_BAD_REQUEST)


class AIAssistantView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        try:
            query = request.data.get('query', '').strip().lower()
            user = request.user  # Get the authenticated user

            try:
                user_profile = UserProfile.objects.select_related('user').get(user=user)
            except UserProfile.DoesNotExist:
                return Response({"error": "User profile not found"}, status=status.HTTP_404_NOT_FOUND)

            # Prepare user data for context
            user_data = {
                "name": user.name,
                "email": user.email,
                "dob": str(user_profile.dob),
                "gender": user_profile.gender,
                "is_verified": user.is_verified ,
                "misc": user.misc
            }

            # AI Assistant Prompt
            prompt = f"""
            You are a personal AI assistant. The user details are:
            {user_data}
            User asks: {query}
            Respond accurately based on their details.
            """



            # Groq(api_key="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")

            # response = client.chat.completions.create(
            #     model="llama3-8b-8192",  # Options: "llama3-70b", "gemma-7b"
            #     messages=[{"role": "system", "content": prompt}]
            # )

            # openai_api_key = os.getenv('OPEN_AI_KEY')
            # if not openai_api_key:
            #     raise ValueError("OpenAI API key is not set")

            client = OpenAI(
                api_key=settings.OPEN_AI_KEY
            )

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                store=True,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return Response({"response": completion.choices[0].message.content}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e), "traceback": traceback.format_exc()}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




class SyncContactsView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        contacts_data = request.data.get("contacts", [])
        
        if not contacts_data:
            return Response({"error": "No contacts provided"}, status=status.HTTP_400_BAD_REQUEST)

        user = request.user
        created_contacts = []
        skipped_duplicate_contacts = []

        for contact_data in contacts_data:
            phone_number = contact_data.get("phone_number")
            email = contact_data.get("email")

            existing_contact = Contact.objects.filter(user=user).filter(
                Q(phone_number=phone_number) | Q(email=email)
            ).first()

            if existing_contact:

                skipped_duplicate_contacts.append({
                    "name": existing_contact.name,
                    "phone_number": existing_contact.phone_number,
                    "email": existing_contact.email,
                    "message": "Already exists"
                })
                continue 

            contact_data["user"] = user.id
            serializer = ContactSerializer(data=contact_data, context={'request': request})

            if serializer.is_valid():
                serializer.save()
                created_contacts.append(serializer.data)
            else:
                return Response({"error": "Invalid contact data", "details": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

        return Response({
            "message": "Contacts processed",
            "contacts_saved": created_contacts,
            "contacts_skipped": skipped_duplicate_contacts
        }, status=status.HTTP_201_CREATED)

    def get(self, request):

        contacts = Contact.objects.filter(user=request.user)
        serializer = ContactSerializer(contacts, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)




def beautify_text(extracted_text):
    beautified_text = re.sub(r'\s+', ' ', extracted_text)
    beautified_text = re.sub(r'([a-zA-Z0-9])\n([a-zA-Z])', r'\1 \2', beautified_text)
    beautified_text = beautified_text.strip()

    return beautified_text

class UploadDocumentsView(APIView):
    permission_classes=[IsAuthenticated]
    parser_classes = [
        MultiPartParser,
        FormParser,
        JSONParser,
    ]

    def post(self, request, *args, **kwargs):
            file = request.FILES.get("resume")
            user=request.user
            if not file:
                return Response(data="No file uploaded", status=status.HTTP_400_BAD_REQUEST)

            file_type = get_file_type(file.name)
            if file_type == "PDF":
                resume_text = beautify_text(extract_text_from_pdf(file))
            elif file_type == "Word":
                resume_text = beautify_text(extract_text_from_doc(file))
            else:
                return Response(data=f"Unsupported file type: {file.name}", status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
            document = UserDocuments.objects.create(
                                    user=user,
                                    file=file, 
                                    extracted_text=resume_text  
                                )
            print(document)
            return Response(data={"extracted_text": resume_text}, status=status.HTTP_200_OK)
        




#Initialize SpeechBrain model for speaker recognition
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_dir")

#Directory to store audio files
AUDIO_DIR = os.path.join(settings.MEDIA_ROOT, 'AudioFile')
os.makedirs(AUDIO_DIR, exist_ok=True)  # Ensure directory exists
#Speaker database (dynamically loaded from files)
speaker_db = {}
store_all_chat_after_identify_sepacker={}

def load_speakers():
    """Load speaker audio files dynamically from AUDIO_DIR."""
    global speaker_db
    speaker_db = {}
    for file in os.listdir(AUDIO_DIR):
        if file.endswith(".wav"):
            name = os.path.splitext(file)[0]  # Extract name from filename
            speaker_db[name] = os.path.join(AUDIO_DIR, file)

load_speakers()

#Extract speaker embedding
def get_speaker_embedding(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    signal, fs = torchaudio.load(audio_path)
    embedding = classifier.encode_batch(signal).squeeze().detach().numpy()
    return embedding.flatten()[:192]  # Ensure fixed size (192)

#Identify speaker
def identify_speaker(unknown_audio_path):
    unknown_embedding = get_speaker_embedding(unknown_audio_path)
    best_match = None
    lowest_distance = float("inf")

    for name, sample_audio in speaker_db.items():
        sample_embedding = get_speaker_embedding(sample_audio)
        distance = cosine(sample_embedding, unknown_embedding)
        if distance < lowest_distance:
            lowest_distance = distance
            best_match = name

    confidence = max(0, 1 - lowest_distance)  # Ensure confidence is non-negative
    return best_match if confidence > 0.7 else "Unknown", confidence

#Initialize Deepgram API
deepgram = Deepgram(settings.DEEPGRAM_API_KEY)

def transcribe_audio(audio_path):
    """Transcribe the audio using Deepgram API."""
    with open(audio_path, "rb") as audio:
        response = deepgram.transcription.sync_prerecorded(
            {"buffer": audio, "mimetype": "audio/wav"},
            {"punctuate": True, "diarize": False}
        )
    return response["results"]["channels"][0]["alternatives"][0]["transcript"]

#API View to identify a speaker
class SpeakerIdentificationAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        """Identify the speaker from an uploaded audio file."""
        audio_file = request.FILES.get('file')
        if not audio_file:
            return Response({'error': 'No audio file provided'}, status=400)

        temp_dir = os.path.join(settings.MEDIA_ROOT, 'tmp_dir')
        os.makedirs(temp_dir, exist_ok=True)
        audio_path = os.path.join(temp_dir, audio_file.name)

        with open(audio_path, 'wb') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        speaker, confidence = identify_speaker(audio_path)
        transcription = transcribe_audio(audio_path)

        return Response({
            'speaker': speaker,
            'confidence': round(confidence, 4),
            'transcription': transcription
        })

#API View to add a speaker
class AddSpeakerAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        """Add a new speaker to the database by uploading an audio sample."""
        audio_file = request.FILES.get('file')
        speaker_name = request.data.get('name')

        if not audio_file or not speaker_name:
            return Response({'error': 'Speaker name and file are required'}, status=400)

        #Append timestamp to filename to prevent overwriting
        timestamp = int(time.time())  # Example: 1710941234
        file_name = f"{speaker_name}_{timestamp}.wav"
        speaker_audio_path = os.path.join(AUDIO_DIR, file_name)

        #Save the file
        with open(speaker_audio_path, 'wb') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        #Save speaker info in the database
        speaker, created = Speaker.objects.get_or_create(name=speaker_name)
        speaker.audio_file = f"AudioFile/{file_name}"  # Update file path
        speaker.save()

        load_speakers()
        return Response({'message': f'Speaker {speaker_name} added successfully!', 'created': created})





class CreateUserConnectionAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = UserConnectionSerializer(
            data=request.data,
            context={'request': request}  
        )
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def get(self, request):
        userconnection = UserConnection.objects.filter(user=request.user)
        serializer = UserConnectionSerializer(userconnection, many=True)
        return Response({
            "data": serializer.data,
        }, status=status.HTTP_200_OK)



@api_view(['GET'])
def get_transcriptions(request):
    transcriptions = Transcription.objects.all().order_by('-timestamp')
    serializer = TranscriptionSerializer(transcriptions, many=True)
    return Response(serializer.data)
        
        

class TranscriptUploadView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        transcript_text = request.data.get('transcript', '')
        if not transcript_text:
            return Response({"error": "Transcript is required"}, status=status.HTTP_400_BAD_REQUEST)

        user = request.user
        data = {'full_text': transcript_text, 'user': user.id}
        serializer = TranscriptUploadSerializer(data=data)
        if serializer.is_valid():
            serializer.save(user=user)
            return Response(
                {"message": "Transcript uploaded successfully", "transcript": serializer.data},
                status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class AskAIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        query = request.data.get('query', '').strip()
        transcript_group = request.data.get('transcript_group', None)

        if not query:
            return Response({"error": "Query is required"}, status=status.HTTP_400_BAD_REQUEST)

        user = request.user
        # cache_key = f"ai_query_{user.id}_{query}_{transcript_group or 'all'}"
        # cached_response = cache.get(cache_key)
        # if cached_response:
        #     return Response(cached_response, status=status.HTTP_200_OK)

        # Fetch transcripts
        if transcript_group:
            transcripts = Transcript.objects.filter(user=user, transcript_group=transcript_group).order_by('line_order')
        else:
            transcripts = Transcript.objects.filter(user=user).order_by('created_at', 'line_order')

        if not transcripts.exists():
            return Response({"error": "No transcripts found"}, status=status.HTTP_404_NOT_FOUND)

        # Prepare context for AI
        context = "Transcript data:\n"
        for t in transcripts:
            context += f"{t.created_at.strftime('%Y-%m-%d %H:%M:%S')} - {t.speaker}: {t.text}\n"

        # AI prompt
        prompt = f"""
        You are an intelligent assistant analyzing transcripts. Based on this data:
        {context}
        Answer the user's query: "{query}"
        Provide a concise, accurate, and natural language response. If the query is unclear, make reasonable assumptions.
        """

        # Call OpenAI API
        client = OpenAI(api_key=settings.OPEN_AI_KEY)
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            ai_response = response.choices[0].message.content.strip()

            # Serialize transcript data
            # serializer = TranscriptSerializer(transcripts, many=True)
            response_data = {
                "message": "Query processed successfully",
                "ai_response": ai_response,
            }

            # Cache the response for 1 hour
            # cache.set(cache_key, response_data, timeout=3600)
            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": f"AI processing failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        




class RecordingView(APIView):
    def post(self, request, action):
        try:
            if action == 'start':
                backend.start_recording()
                return Response({"message": "Recording started"}, status=status.HTTP_200_OK)
            elif action == 'stop':
                backend.stop_recording()
                return Response({"message": "Recording stopped"}, status=status.HTTP_200_OK)
            return Response({"error": "Invalid action"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SaveSpeakerNameView(APIView):
    def post(self, request):
        try:
            name = request.data.get('name')
            if not name:
                logger.warning("No name provided for save_speaker_name")
                return Response({"error": "Name is required"}, status=status.HTTP_400_BAD_REQUEST)
            speaker_id = backend.save_speaker_name(name)
            return Response({"message": f"Speaker {speaker_id} renamed to {name}"}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Save speaker name error: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class AskAIViewTemp(APIView):
    def post(self, request):
        try:
            query = request.data.get('query')
            if not query:
                logger.warning("No query provided for ask_ai")
                return Response({"error": "Query required"}, status=status.HTTP_400_BAD_REQUEST)
            response = backend.ask_ai(query)
            return Response({"response": response}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Ask AI error: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ClearDataView(APIView):
    def post(self, request):
        try:
            backend.clear_data()
            return Response({"message": "Data cleared"}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Clear data error: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class HistoryView(APIView):
    def get(self, request, type):
        try:
            if type == 'chat':
                chats = Chat.objects.all()
                serializer = ChatSerializer(chats, many=True)
            elif type == 'ai':
                history = AIChatHistory.objects.all()
                serializer = AIChatHistorySerializer(history, many=True)
            else:
                return Response({"error": "Invalid type"}, status=status.HTTP_400_BAD_REQUEST)
            return Response(serializer.data)
        except Exception as e:
            logger.error(f"History fetch error: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        
def index_view(request):
    return render(request, 'index.html')