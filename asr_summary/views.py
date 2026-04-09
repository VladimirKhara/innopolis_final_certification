import traceback

from django.http import HttpResponse
from django.shortcuts import redirect, render
from datasets import Audio
from django.urls import reverse_lazy
from asr_summary.ai_engine import start_asr, start_summatization
from asr_summary.forms import AudioFileForm

from django.views.generic import ListView, DeleteView
from asr_summary.models import AudioFile

class AudioListView(ListView):
    model = AudioFile
    template_name = 'index.html'


def processing(request, pk):
    audio = AudioFile.objects.get(id=pk)
    filename = audio.audiofile
    asr_result = start_asr(filename)
    summarization_result = start_summatization(asr_result.transcription)
    return render(request, 'processing.html', context={'file': audio.audiofile,'asr_result': asr_result, 'summarization_result' : summarization_result})

def upload_audio(request):
    if request.method == 'POST':
        form = AudioFileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('main')
        else:
            form = AudioFileForm()
    else:
        form = AudioFileForm()
    return render(request, 'upload.html', {'form': form})

class AudioDeleteView(DeleteView):
    model = AudioFile
    template_name = "confirm_delete.html"
    success_url = reverse_lazy('main')

    def form_valid(self, form):
        obj = self.get_object()
        try:
            obj.audiofile.delete(save=False)
        except Exception:
            traceback.print_exc()
        return super().form_valid(form)