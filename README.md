# YSDA Speech Processing Course

- Materials for each week are in ./week* folders

## Course program

- Week 1: [Slides](https://docs.google.com/presentation/d/1Cte6w0t8yTJRFirde6GPxKB29VX3SrX1mhAkKYEN-n4/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/i/E-cGC7pH46UjWQ) | [Seminar](https://disk.yandex.ru/i/j4MsAAliJ5ri_A)
    - Lecture: Intro to Digital Signal Processing (DSP)
    - Seminar: Implement DSP pipeline
    - Homework (5pt): Implement mel-spectrogram transformations
- Week 2 [Slides](https://docs.google.com/presentation/d/1M3AULJazLVuwJpKsgb5UowcI_XAT6AB8I22ofd1fJY8/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/i/gyROgNpX5LZNZQ) | [Seminar](https://disk.yandex.ru/i/EwrqSMFQHICVRA):
    - Lecture: Introduction to speech NN discriminative models. Voice Activity Detection (VAD) and Sound Event Detection (SED) tasks
    - Seminar: Train VAD models
    - Homework (15pt): Train SED models
- Week 3 [Slides](https://docs.google.com/presentation/d/1IppXDfOI2Du5bMfnjxosAgg6Uu0VjXEVRr_8nkUxAus/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/i/uo6scU9T7R5V9Q) | [Seminar](https://disk.yandex.ru/i/NE6mekhYEQoPGA):
    - Lecture: Keyword Spotting and Speech Biometrics tasks
    - Seminar: Train Biometrics model and look at embeddings
    - Homework (20pt): Train Biometrics model ECAPA-TDNN with contrastive loss
- Week 4 [Slides](https://docs.google.com/presentation/d/1dU9NasfSDCfgldqJzaaEF2xdamejnGUb4tzu4wbTCPU) | [Lecture](https://disk.yandex.ru/i/_KgUj9Snbl9BsA) | [Seminar](https://disk.yandex.ru/d/e6n-TwqNPyacOw):
    - Lecture: Speech Recognition I
    - Seminar: CTC forward-backward, soft alignment
    - Homework (10pt): CTC/RNN-T decoding, RNN-T forward-backward
- Week 5 [Slides](https://github.com/yandexdataschool/speech_course/blob/main/week_05_pretraining_asr/lecture.pdf) | [Lecture](https://disk.yandex.ru/d/vownNQ4ZuBARPg) | [Seminar](https://disk.yandex.ru/i/7BtUnWcme2ryag):
    - Lecture: Speech Recognition II, Pretraining
    - Homework (5pt): Finetune Wav2Vec2
- Week 6: [Slides](https://docs.google.com/presentation/d/1BjEEduYCH7z0yKIbL3-t3jaEv90c2r_O6aaMWVXhtRg/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/i/smC9fVJmXGWOeA) | [Seminar](https://disk.yandex.ru/i/vMn3gnPuH9j3mA)
    - Lecture: ASR Inference
    - Seminar: Streaming ASR
    - Homework (5pt): Seminar continuation
- Week 7: [Slides](https://docs.google.com/presentation/d/1MAAAc_2vRS2jhxZlqfpx0m7Z3MY9J1jyXgdl7XStFf8/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/d/7w6n6ad8PGxV7w)
    - Lecture: Text-to-Speech I, intro, preprocessor, metrics
- Week 8: [Slides](https://docs.google.com/presentation/d/1hR4koanl61qFXNAk2SRp45gYcgxUAc5Xt6_UQJMJYmM/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/i/THrniTW4cyBnrQ)
    - Lecture: Text-to-Speech II, Acoustic models and vocoding
    - Seminar (5pt): Pitch estimation, Monotonic Alignment Search for phoneme duration estimation
    - Homework (10pt): Train FastPitch model
- Week 9: [Slides](https://docs.google.com/presentation/d/13pfNuGipGjxlAB754rtQvv_goSFkJR5uTUq4_nCUfWE/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/i/XCo4vLs3z8UhLw) | [Seminar](https://disk.yandex.ru/i/EVqLrmq1VOyedw)
    - Lecture: Text-to-Speech III, Codecs
    - Seminar: Vector Quantizaton, Residual Vector Quantization
- Week 10: [Slides](https://docs.google.com/presentation/d/1MldH8pnho6BiMHyW3qU8aXspdKarV_sRC1mvvs77FbE/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/i/uoemA7ga_Rz2mQ)
    - Lecture: Text-to-Speech IV, Tortoise and other tranformers for TTS
    - Homework (15pt): write inference for CLM with two transformers
- Week 11: [Slides](https://docs.google.com/presentation/d/1zfo-vvYHKIFZniDuh-afKif8v8O3272uLSAXm7cFyL8/edit?slide=id.g34b931ff1ac_0_118#slide=id.g34b931ff1ac_0_118) | [Lecture](https://disk.yandex.ru/i/hl_jSEu2fLgWGA)
    - Lecture: Multimodality, How to build a big GPT with voice capabilities
- Week 12: [Slides](https://drive.google.com/open?id=1uBAnbciOyoD0VoVdkudPuW3ZeTuAh8wJ&usp=drive_fs) | [Lecture](https://disk.yandex.ru/i/tEFqArlw8vH53g) | [Seminar](https://disk.yandex.ru/i/4ZY96Fsb-si_XQ)
    - Lecture: noise reduction
    - Seminar: Streaming STFT and ISTFT
    - Homework (15pt): Noise reduction model implementation
- Week 13: [Slides 1](https://drive.google.com/open?id=1tzsZlPC4UPM8wrpuD7LzV3JBJCiW5vmp&usp=drive_fs) | [Slides 2](https://drive.google.com/open?id=1u2WKIt9iPpxFQQUJgxXUeq29YGwz7zMh&usp=drive_fs) | [Lecture+Seminar](https://disk.yandex.ru/i/3gEl2GV22QvEkA)
    - Lecture: Acoustic Echo Cancelation (AEC) and Beamforming
    - Homework (5pt): Basic AEC implementation



<details>
<summary>
Course program for spring 2024
</summary>

- Week 1: [Slides](https://docs.google.com/presentation/d/1IkVFw8PgWPjn74918rFbuahd7Q38O0f04_bk5_fWPhE/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/i/eL9PZKCT1O7yhw) | [Seminar](https://disk.yandex.ru/i/ILE1n2aVIWkxEA)
    - Lecture: Intro to Digital Signal Processing (DSP)
    - Seminar: Implement DSP pipeline
- Week 2: [Slides](https://docs.google.com/presentation/d/10cPD8k2oVL2D4wfp4eMBGvSVOShIAfAUgFuqwjedJF4/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/i/8IwvU8PXAwDKSg) | [Seminar](https://disk.yandex.ru/i/PHaDf7Gzo0LNkA)
    - Lecture: Introduction to speech NN discriminative models. Voice Activity Detection (VAD) and Sound Event Detection (SED) tasks
    - Seminar: Train VAD models
    - Homework: Train SED models
- Week 3: [Slides](https://docs.google.com/presentation/d/1q5bL4Pdp9MRLJHRuZeabgoHt1KsSg-qGqoEXb_hokW0/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/i/pGDEBo68YDjenQ) | [Seminar](https://disk.yandex.ru/d/BBpCSG2JLlxf6A)
    - Lecture: Keyword Spotting and Speech Biometrics tasks
    - Seminar: Train Biometrics model and look at embeddings
    - Homework: Train Biometrics model to better quality
- Week 4: [Slides](https://docs.google.com/presentation/d/1WLjwnJHwjfWfrl31Q3CwwkSEtM5z47LCZ5lXAFk6_Gw/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/i/QHO1SnqQSkRY6A) | [Seminar](https://disk.yandex.ru/d/87Lrfi1VoH-F8w)
    - Lecture: Speech Recognition I
    - Seminar: Metrics and augmentations for speech recognition
    - Homework: Implement CTC algorithm
- Week 5: [Slides](https://docs.google.com/presentation/d/1JI8NEjZLNQhbUuO5py3OlYIqMgTPI4LSR-CU3-Rnp9g/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/d/uT9o9bKt60w79g)
    - Lecture: Speech Recognition II, Pretraining
    - Homework: Finetune Wav2Vec2
- Week 6: [Slides](https://docs.google.com/presentation/d/1MAAAc_2vRS2jhxZlqfpx0m7Z3MY9J1jyXgdl7XStFf8/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/i/X6Se5K14FF91Ow)
    - Lecture: Text-to-Speech I, intro, preprocessor, metrics
- Week 7: [Slides](https://docs.google.com/presentation/d/1CO1_5xzZb7mYLfQfdhqN0350dNCkgLt6hHB7smUjdGA/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/i/YW_TVQMGKbuYag)
    - Lecture: Text-to-Speech II, Acoustic models
    - Seminar: Pitch estimation, Monotonic Alignment Search for phoneme duration estimation
    - Homework: Train FastPitch model
- Week 8: [Slides, p1](https://docs.google.com/presentation/d/1QU5sUe8_uGEiFs-IFua7EU5_imsZK2TRuKJ_6IY4O9k/edit?usp=sharing) | [Lecture, p1](https://disk.yandex.ru/d/R4p0hupEJrF02g) | [Slides, p2](https://docs.google.com/presentation/d/143qUGId_yvMKx3IDOcErF5M1V6NXHmHp30GuRryhHxs/edit?usp=sharing) | [Lecture, p2](https://disk.yandex.ru/d/abw48YXapfwvfw) | [Seminar](https://disk.yandex.ru/i/XSr1jKD_ah4hkg)
    - Lecture, p1: Text-to-Speech III, Vocoding
    - Lecture, p2: Vector Quantization, Codecs
    - Seminar: Vector Quantizaton, Residual Vector Quantization
- Week 9: [Slides](https://docs.google.com/presentation/d/1ARlJHMr_c0R2g5Od-66ZTTuZGRxdTCjAzX2LQ9XPFdQ/edit#slide=id.g1f4de8b9e93_0_1414) | [Lecture, p1](https://disk.yandex.ru/i/80GAO85GUjRzKA) | [Lecture, p2](https://disk.yandex.ru/i/icrNEsu17jL7hA)
    - Lecture: Tranformers for TTS
    - Homework: write inference for pre-trained transformer
- Week 10: [Slides](https://docs.google.com/presentation/d/1qq67ydUQpe40Xv5B-lEUWdhB2UMt0rjFuL1--EcqEzU/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/d/olPQZPtFyaTOCw) | [Seminar](https://disk.yandex.ru/i/Dg19n05qIiHT8g)
    - Lecture: noise reduction
    - Seminar: Streaming STFT and ISTFT
    - Homework: Noise reduction model implementation
- Week 11: [Slides](https://docs.google.com/presentation/d/1JsPBlITpc-a_I_1foWM1tWyO6B3y6OC_AIfmM6CLMr4/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/i/vGOufpHs8x5mZQ)
    - Lecture: Acoustic Echo Cancelation (AEC) and Beamforming
- Week 12: [Slides](https://docs.google.com/presentation/d/1KfiPechci9fmcgi8K9I1_MybAdBo_MwrcK1AYqT4vLI/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/i/JpgbsaLbpN22Qw) | [Seminar](https://disk.yandex.ru/i/1erd90ueHJjjzw)
    - Lecture: ASR Inference
    - Seminar: Streaming ASR
- Week 13: [Slides](https://docs.google.com/presentation/d/1iwGzbmdJDulfjEvGhw1_oGle8IAjFlalUr_lHWFsVM8/edit?usp=sharing) | [Lecture](https://disk.yandex.ru/i/pXO7D-7JJCjlhQ)
    - Lecture: Flow based TTS + Voice Conversion

</details>

## Contributors & course staff

Current:
- Pavel Mazaev - VAD, SED
- Alex Rak - spotter, biometry
- Daniil Volgin - ASR
- Dzmitry Soupel - ASR
- Stepan Kargaltsev - ASR
- Evgeniia Elistratova - TTS
- Roman Kail - TTS
- Vladimir Platonov - TTS
- Ivan Matvienko - TTS
- Ravil Khisamov - VQE
- Anton Porfirev - AEC

Previous iteration:
- Andrey Malinin - Course admin, lectures, seminars, homeworks
- Vladimir Kirichenko - lectures, seminars, homeworks
- Segey Dukanov - lecures, seminars, homeworks
- Evgenii Shabalin - lecture and homework on conversion
- Mikhail Andreev - ASR
