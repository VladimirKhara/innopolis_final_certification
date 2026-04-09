import gc

import librosa
import torch
from asr_summary.apps import AsrSummaryConfig
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ASRResult:
    transcription: str
    elapsed_time: float

@dataclass
class SummatizationResult:
    summatization_text: str
    elapsed_time: float

def start_asr(filename):
    print(AsrSummaryConfig.device)
    forced_decoder_ids = AsrSummaryConfig.asr_processor.get_decoder_prompt_ids(language="russian", task="transcribe")
    start = datetime.now()
    audio, sr = librosa.load(filename, sr=16000)
    audio_copy  = audio.copy()
    input_features = AsrSummaryConfig.asr_processor(audio_copy, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(AsrSummaryConfig.device)
    with torch.no_grad():
        predicted_ids = AsrSummaryConfig.asr_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = AsrSummaryConfig.asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)
    end = datetime.now()
    diff = (end-start).total_seconds()
    del audio
    torch.cuda.empty_cache()  
    gc.collect()
    return ASRResult(transcription[0], diff)

def start_summatization(text):
    start = datetime.now()
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
    end = datetime.now()
    diff = (end-start).total_seconds()
    return SummatizationResult(summary, diff)