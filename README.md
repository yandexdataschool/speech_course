# YSDA Speech Processing Course

- Materials for each week are in ./week* folders

## Course program

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

## Contributors & course staff

Current:
- Alex Rak - VAD, spotter, biometry
- Mikhail Andreev - ASR
- Stepan Kargaltsev - ASR
- Evgeniia Elistratova - TTS
- Roman Kail - TTS
- Vladimir Platonov - TTS
- Evgenii Shabalin - TTS
- Ravil Khisamov - VQE

Previous iteration:
- Andrey Malinin - Course admin, lectures, seminars, homeworks
- Vladimir Kirichenko - lectures, seminars, homeworks
- Segey Dukanov - lecures, seminars, homeworks
