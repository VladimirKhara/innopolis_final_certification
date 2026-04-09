from django import forms

from asr_summary.models import AudioFile

class AudioFileForm(forms.ModelForm):
    class Meta:
        model = AudioFile
        fields = ['title', 'audiofile']
