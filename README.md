# YSDA Speech Processing Course

- Materials for each week are in ./week* folders

## Course program

- Week 1 - Introduction to Speech
    - Lecture: In this lecture we introduce the area of speech processing, discuss historical background and current trends. In the second half of the lecture we introduce the concept fo speech as a separate modality from text or images and foreshadow concepts from later lectures.
  
- Week 2 - Digital Signal Processing
    - Lecture: In this lecture we discuss how to transform an audio signal into a form which is convenient for use in Speech Recognition and Synthesis. We discuss: how an audio wave is sampled and digitized; The Fourier Transform and the Discrete Fourier Transform and how they can be used to obtain the frequency spectrum of the signal; How to use the Short-Time-Fourier-Transform to represent sound as a Spectrogram; finally, we discuss the Mel-Scale and how to obtain a Mel-Spectrogram. 
    - Seminar: In part 1 we will implement the Short-Time-Fourier-Transform and obtain a Mel-Spectrogram. In part 2 we will: recover a Spectrogram from a Mel-Spectrogram. Reconstruct the original audio signal via the Griffin-Lim algorithm and do some simple voice warping. 
    - Homework: Audio-MNIST: Implement a Neural Network model to do simple digit classification based on a mel-spectrogram. 

- Week 3 - ASR I: Introduction to Speech Recognition
    - Lecture: In this lecture we aim to draw a map of the general area of ASR. We do a quick recap of how audio is processed into a convenient form to work with (Mel-Spectrogram or MFCCs). Then we discuss how to process text into sub-word speech units, such as graphemes and phonemes and how to assess ASR quality using Word Error Rate, Character Error Rate and Phone Error Rate using the Levenstein Algorithm. Then we examine the problem aligning the sequence of acoustic features and sub-word speech units using either state-space models or attention mechanisms. Finally, we take a look at a high-level taxonomy of ASR systems.
    - Seminar: You have to implement the recursive and matrix Levenstein Algorithms


- Week 4- ASR II: Discriminative State-Space ASR Models
    - Lecture: In this lecture we discuss Discriminative State-Space ASR systems, specifically Connectionist Temporal Classification. We discuss important difference in model structure and speech units between HMMs and CTC, inference and training trellises and the Forward-Backward algorithm for CTC. We close with a discussion where we contrast HMMs and CTC, and show the primary similarities and differences. 
    - Seminar: Implement CTC Forward-Backward Algorithm, training CTC model.

- Week 5 - ASR III: Context Modelling and Language Model Fusion
    - Lecture: In this lecture we analyse the errors typically made by a CTC system and introduce Prefix Search Decoding as an improved inference scheme for CTC. We introduce an alternative model type, called the RNN-Transducer, examine it's training and inference and compare it with the CTC approach.
    - Seminar + Homework:  Implement RNN-Transducer loss and inference
    
- Week 6 - ASR IV:  Context Modelling, Language Models and Attention-based ASR
    - Lecture: We introduce language models and how they are typically evaluated. Then we take a closer look at N-Gram and Neural Language Models. Then we look at how we can use N-Gram Language models to do Prefix Search Decoding for CTC models. We close with a discussion of attention-based Autoregressive ASR models. We discuss models such as Listen, Attend and Spell, examine advantages and limitations of such models and how they can be overcome. 

- Week 7 - ASR V: Attention-based ASR Systems
    - Lecture: We close our discussion of ASR systems by examining recent developments in ASR, including SOTA language modelling approaches, semi-supervised learning via noisy teacher-student training, and speech representation learning via Wav2Vec and VQ-Wav2Vec.

- Week 8 - TTS I: Introduction to TTS
    - Lecture: In this lecture we introduce the problem of Speech Synthesis (Text-to-Speech), take a look at the history of TTS and provide an overview of the current state of TTS.
    - Seminar: Assistance with Homework

- Week 9 - TTS II : Vocoders I
    - Lecture: In this lecture we examine the Vocoder component of TTS systems. We introduce the concept of a vocoder - a component which takes a _synthesised_ Mel-Spectrogram and produces an acoustic waveform. We provide an overview of modern vocoder types and then focus on WaveNet, WaveRNN and LPCNet architectures.
    - Seminar: Assistance with Homework

- Week 10 - TTS III: Vocoders II
    - Lecture: In this lecture we examine Normalising-Flow, GAN and Diffusion based vocoder models. We introduce the basics of Normalising Flows and discuss the WaveGlow vocoder. Then we move onto HIFI-GAN - a vocoder based on Generative Adversarial Networks. We closer with a discussion of diffusion-based vocoders, such as WaveGrad.
    - Seminar: Assistance with Homework

- Week 10 - TTS VI: Acoustics Models I
    - Lecture: In this lecture we examine acoustic models for speech synthesis - components which take marked-up text and produce a synthesised Mel-Spectrogram. We examine the SOTA, but slow Tacotron2 model and the some faster alternatives, such as FastSpeech, FastPitch and Parallel Tacotron. We also take a look at intonation control.
    - Seminar: Assistance with Homework

- Week 10 - TTS V: Acoustics Models II
    - Lecture: In this lecture we examine how to extend the previously introduced architectures to allow for control of speech style as well as  multi-speaker and multi-lingual speech synthesis.
    - Seminar: Assistance with Homework

- - Week 10 - TTS VI: Speech to Speech
    - Lecture: We close the course with an overview lecture on some of the most exciting frontiers off research in TTS - Speech-to-Speech models. We examine the tasks of Speech Enhancement, Speech Separation, Voice Cloning, Voice conversion and speech translation. 
    - Seminar: Assistance with Homework
## Contributors & course staff

- Andrey Malinin - Course admin, lectures, seminars, homework
- Vladimir Kirichenko - lectures, seminars, homework
- Segey Dukanov - lecures, seminars, homework
- Evgeniia Elistratova - seminars, homework
- Ekaterina Ermishkina - seminars, homework
- Anastasia Demidova - seminars, homework
