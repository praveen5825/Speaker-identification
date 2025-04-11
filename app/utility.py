from django.core.mail import send_mail
from django.conf import settings
from .models import UserProfile, TextMessage, OTP, User
from django.utils import timezone
import random
from PyPDF2 import PdfReader 
import docx2txt
import os

def generate_otp():
    otp = str(random.randint(100000, 999999))
    return otp

def send_otp(email):
    otp = generate_otp()
    try:
        response = send_mail(
                    'OTP',
                    f'Enter OTP {otp} to verify your email.',
                    settings.EMAIL_HOST_USER,
                    [email],
                    fail_silently=False,
                    )
        response = bool(response)
    except:
        response = False
    
    if response:
        otp_obj = OTP.objects.filter(email=email)
        if otp_obj.exists():
            otp_obj = otp_obj.last()
            otp_obj.otp = otp
            otp_obj.created_at = timezone.now()
            otp_obj.save()
        else:
            OTP.objects.create(email=email, otp=otp)
    return response



def get_file_type(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == ".pdf":
        return "PDF"
    elif file_extension.lower() in (".doc", ".docx"):
        return "Word"
    else:
        return "Unknown"


def extract_text_from_pdf(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_doc(doc_path):
    text = docx2txt.process(doc_path)
    return text