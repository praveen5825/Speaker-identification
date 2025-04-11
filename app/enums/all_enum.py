from django.db.models import IntegerChoices,TextChoices
from django.utils.translation import gettext_lazy as _

class DeviceType(TextChoices):
    ANDROID = "ANDROID", _("Android")
    IOS = "IOS", _("Ios")