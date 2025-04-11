from django.contrib import admin
from .models import *


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ['email','is_verified']
    search_fields = ['email']

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user','gender','dob']

@admin.register(TextMessage)
class TextMessageAdmin(admin.ModelAdmin):
    list_display = ['user','text','timestamp']

@admin.register(OTP)
class OTPAdmin(admin.ModelAdmin)    :
    list_display = ['email','created_at','otp']

@admin.register(TextMessageHistory)
class TextMessageHistoryAdmin(admin.ModelAdmin):
    list_display = ['text_message','text','timestamp','created_by']

@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'message', 'timestamp', 'is_read')
    list_filter = ('user', 'is_read')  
    search_fields = ('message',)


@admin.register(Contact)
class ContactAdmin(admin.ModelAdmin):
    list_display = ['id','user','timestamp']

@admin.register(UserDocuments)
class DisplayUserDocument(admin.ModelAdmin):
    list_display = ['id','user','created_at','extracted_text','file']
    
@admin.register(Speaker)
class DisplaySpeaker(admin.ModelAdmin):
    list_display=["id","name","audio_file"]
    
    
@admin.register(Transcript)
class DisplayTranscript(admin.ModelAdmin):
    list_display=["id","user","created_at","updated_at","full_text","speaker","text","line_order","transcript_group"]
    
@admin.register(UserConnection)
class DisplayUserConnection(admin.ModelAdmin):
    list_display=["user","name","email","whatsapp_number","relation"]
    
@admin.register(Chat)
class DisplayChat(admin.ModelAdmin):
    list_display=["user_id","user_name","speech","time_stamp"]