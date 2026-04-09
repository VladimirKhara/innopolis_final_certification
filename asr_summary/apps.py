from django.apps import AppConfig
import torch
from evaluate import load
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration,  WhisperProcessor
from transformers import AutoTokenizer, T5ForConditionalGeneration

class AsrSummaryConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'asr_summary'
    asr_model = None
    asr_processor = None
    device = 'cpu'
    summarization_model = None
    summarization_tokenizer= None

    def ready(self):
        if AsrSummaryConfig.asr_model is None:
            AsrSummaryConfig.device = "cuda" if torch.cuda.is_available() else "cpu"
            AsrSummaryConfig.asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(AsrSummaryConfig.device)
        if AsrSummaryConfig.asr_processor is None:
            AsrSummaryConfig.asr_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        if AsrSummaryConfig.summarization_model is None:
            AsrSummaryConfig.summarization_model = T5ForConditionalGeneration.from_pretrained('IlyaGusev/rut5_base_sum_gazeta')
        if AsrSummaryConfig.summarization_tokenizer is None:
            AsrSummaryConfig.summarization_tokenizer = AutoTokenizer.from_pretrained('IlyaGusev/rut5_base_sum_gazeta')
