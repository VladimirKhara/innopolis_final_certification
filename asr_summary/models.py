from django.db import models
from django.core.validators import FileExtensionValidator

class AudioFile(models.Model):
    id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=100)
    audiofile = models.FileField(upload_to='audio/', validators=[FileExtensionValidator(allowed_extensions=['mp3', 'wav'])])

    def __str__(self):
        return self.title
