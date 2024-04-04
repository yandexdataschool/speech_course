import librosa
import numpy as np
import pyloudnorm as pyln
import torch
import torch.nn.functional as F
from codec.codec import CodecModel
from dp.phonemizer import Phonemizer as DPPhonemizer
from pyannote.audio import Inference
from torch.utils.data import Dataset


class Phonemizer:
    phoneset = ['<pad>', '<unk>', '<bos>', '<eos>', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', ',', '.', '!', '?', '-', ' ']

    def __init__(self, model_path):
        self.phonemizer = DPPhonemizer.from_checkpoint(model_path)
        self.phones_remapper = self._create_phone_remapper()        

    def _create_phone_remapper(self):
        phones_remapper = {ph: ph for ph in self.phoneset}

        phones_remapper[' ,'] = ','
        phones_remapper[' .'] = '.'
        phones_remapper[' !'] = '!'
        phones_remapper[' ?'] = '?'

        phones_remapper[', '] = ','
        phones_remapper['. '] = '.'
        phones_remapper['! '] = '!'
        phones_remapper['? '] = '?'

        phones_remapper[' , '] = ','
        phones_remapper[' . '] = '.'
        phones_remapper[' ! '] = '!'
        phones_remapper[' ? '] = '?'
        return phones_remapper

    def phonemize(self, text):
        phonemes = self.phonemizer(text, lang='en_us')
        phonemes = phonemes.replace('[', '_').replace(']', '_')
        phonemes = [ph for ph in phonemes.split("_") if ph != '']
        phonemes = [self.phones_remapper.get(ph, '<unk>') for ph in phonemes]
        return phonemes
    
    def tokenize(self, text):
        phonemes = self.phonemize(text)
        phoneme_ids = [self.phoneset.index(ph) for ph in phonemes]
        return np.array(phoneme_ids)


class BioembModel:
    HUGGING_FACE_TOKEN = "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"
    SR = 16000

    def __init__(self, device):
        self.device = device
        self.speaker_embedder = Inference(
            model="pyannote/embedding",
            window="whole",
            use_auth_token=self.HUGGING_FACE_TOKEN,
        ).to(self.device)
    
        self.meter = pyln.Meter(self.SR)

    def __call__(self, waveform, sr):
        assert isinstance(waveform, np.ndarray), f"{type(waveform)=}"
        assert len(waveform.shape) == 1, "Waveform should have one dimension"

        if sr != self.SR:
            waveform = librosa.resample(
                waveform,
                orig_sr=sr,
                target_sr=self.SR,
            )
            sr = self.SR

        waveform_for_loud = waveform
        while waveform_for_loud.shape[0] <= self.SR:
            waveform_for_loud = np.concatenate((waveform_for_loud, waveform_for_loud), axis=0)
        loudness = self.meter.integrated_loudness(waveform_for_loud)
    
        waveform = pyln.normalize.loudness(waveform, loudness, -20.0)

        embedding = self.speaker_embedder({
            'waveform': torch.tensor(waveform, device=self.device, dtype=torch.float).unsqueeze(dim=0),
            'sample_rate': sr,
        })

        embedding = F.normalize(torch.tensor(embedding), dim=-1)
        embedding = embedding.numpy(force=True)
           
        return embedding


class CodecApplier:
    def __init__(self, config_path, ckpt_path, sample_rate=16000, device=torch.device("cpu")):
        self.device = device
        self.model = CodecModel(config_path, ckpt_path, sample_rate).to(device)
        self.sample_rate = sample_rate

        self.n_codebooks, self.n_codes = 4, 160

        self.end_token_idx = self.n_codes
        self.start_token_idx = self.n_codes + 1
        self.repetition_token_idx = self.n_codes + 2
    
    def encode(self, waveform, sample_rate):
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy(force=True)

        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=self.sample_rate)
        input = torch.tensor([waveform], device=self.device) # .unsqueeze(dim=0)
        encoded = self.model.encode(input)
        encoded = encoded.squeeze(dim=0).numpy(force=True)

        start = np.full((1, self.n_codebooks), self.start_token_idx, dtype=np.int16)
        audio_tokens = np.concatenate([start, encoded], axis=0)
        return audio_tokens

    def decode(self, encoded, spkr):
        if isinstance(encoded, np.ndarray):
            encoded = torch.tensor(encoded, device=self.device)

        decoded = self.model.decode(encoded, spkr)
        decoded = decoded.squeeze(dim=0).numpy(force=True)
        return decoded


class CodecsDataset(Dataset):
    def __init__(self, dataset, phonemizer, bioemb_model, codec_model):
        self.dataset = dataset
        self.phonemizer = phonemizer
        self.bioemb_model = bioemb_model
        self.codec_model = codec_model
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        waveform, sample_rate, text, *_ = item

        waveform = waveform.squeeze().numpy()
        phonemes = self.phonemizer.phonemize(text)
        phoneme_ids = self.phonemizer.tokenize(text)
        codecs = self.codec_model.encode(waveform, sample_rate)
        bioemb = self.bioemb_model(waveform, sample_rate)

        return phonemes, phoneme_ids, codecs, bioemb

    def __len__(self):
        return len(self.dataset)
