import librosa
import torch
from asr_summary.apps import AsrSummaryConfig
import time
from dataclasses import dataclass

@dataclass
class ASRResult:
    transcription: str
    wer: float
    elapsed_time: int

def start_asr(filename):
    forced_decoder_ids = AsrSummaryConfig.asr_processor.get_decoder_prompt_ids(language="russian", task="transcribe")
    audio, sr = librosa.load(filename, sr=16000)
    print(time.strftime("START: %H:%M:%S", time.localtime()))
    input_features = AsrSummaryConfig.asr_processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(AsrSummaryConfig.device)
    print(time.strftime("After processing: %H:%M:%S", time.localtime()))
    with torch.no_grad():
        predicted_ids = AsrSummaryConfig.asr_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    print(time.strftime("After generation: %H:%M:%S", time.localtime()))

    transcription = AsrSummaryConfig.asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print("Транскрипция:", transcription[0]) 
    print(time.strftime("END: %H:%M:%S", time.localtime()))
    
    return transcription[0]

def start_summatization(text):
    input_ids = AsrSummaryConfig.summarization_tokenizer(
        [text],
        max_length=600,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )["input_ids"]

    output_ids = AsrSummaryConfig.summarization_model.generate(
        input_ids=input_ids,
        no_repeat_ngram_size=4
    )[0]
    summary = AsrSummaryConfig.summarization_tokenizer.decode(output_ids, skip_special_tokens=True)
    print(summary)
    return summary