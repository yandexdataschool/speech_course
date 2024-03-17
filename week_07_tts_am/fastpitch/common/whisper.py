import string

import librosa
import numpy as np
import torch
import torchmetrics
import transformers

from week_07_tts_am.fastpitch.data import Wav


def resample_audio(wav: Wav) -> np.array:
    return librosa.resample(wav.data, orig_sr=wav.sr, target_sr=16000)


class WhisperModel:
    def __init__(self, model_name: str = 'openai/whisper-medium', device: str = 'cpu', lang: str = 'en'):
        self.device = device
        self.lang = lang
        self.model, self.processor, self.forced_decoder_ids = self._load_models(model_name)

    def _load_models(self, model_name: str):
        model = transformers.WhisperForConditionalGeneration.from_pretrained(model_name)
        model.to(self.device)
        processor = transformers.AutoProcessor.from_pretrained(model_name)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=self.lang, task="transcribe")
        return model, processor, forced_decoder_ids

    def predict(self, audios: list[Wav]) -> list[str]:
        processed_audios = [resample_audio(audio) for audio in audios]
        inputs = self.processor(processed_audios, return_tensors="pt", sampling_rate=16000).to(self.device)
        generated_ids = self.model.generate(inputs=inputs.input_features, forced_decoder_ids=self.forced_decoder_ids)
        transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return transcriptions
    

def remove_punctuation(s: str) -> str:
    """
    See for comments and details:
    https://stackoverflow.com/a/23318457/21371044
    """
    return " ".join("".join([" " if ch in string.punctuation else ch for ch in s]).split())


def postprocess_text(text: str) -> str:
    text = remove_punctuation(text)
    text = text.strip().lower()
    return text


def postprocess_texts(texts: list[str]) -> list[str]:
    return [postprocess_text(text) for text in texts]


def calculate_wer(transcriptions: list[str], gt_texts: list[str]) -> torch.FloatTensor:
    wer = torchmetrics.text.WordErrorRate()
    return wer(
        postprocess_texts(transcriptions),
        postprocess_texts(gt_texts)
    )
