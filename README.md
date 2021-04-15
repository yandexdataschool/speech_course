# YSDA Speech Processing Course

- Materials for each week are in ./week* folders

## Course program

- Week 1: Introduction to Speech
    - Lecture: In this lecture we introduce the area of speech processing, discuss historical background and current trends. In the second half of the lecture we introduce the concept fo speech as a separate modality from text or images and foreshadow concepts from later lectures.
  
- [Week 2](https://github.com/yandexdataschool/speech_course/blob/main/week_02): Digital Signal Processing
    - Lecture: In this lecture we discuss how to transform an audio signal into a form which is convenient for use in Speech Recognition and Synthesis. We discuss: how an audio wave is sampled and digitized; The Fourier Transform and the Discrete Fourier Transform and how they can be used to obtain the frequency spectrum of the signal; How to use the Short-Time-Fourier-Transform to represent sound as a Spectrogram; finally, we discuss the Mel-Scale and how to obtain a Mel-Spectrogram. 
    - Seminar: In part 1 we will implement the Short-Time-Fourier-Transform and obtain a Mel-Spectrogram. In part 2 we will: recover a Spectrogram from a Mel-Spectrogram. Reconstruct the original audio signal via the Griffin-Lim algorithm and do some simple voice warping. 
    - Homework: Audio-MNIST: Implement a Neural Network model to do simple digit classification based on a mel-spectrogram. 

- [Week 3](https://github.com/yandexdataschool/speech_course/blob/main/week_03): Introduction to Speech Recognition
    - Lecture: In this lecture we aim to draw a map of the general area of ASR. We do a quick recap of how audio is processed into a convenient form to work with (Mel-Spectrogram or MFCCs). Then we discuss how to process text into sub-word speech units, such as graphemes and phonemes, and how to align between sequence of acoustic features and sub-word speech units using either state-space models or attention mechanisms. We compare how to decode ASR using discriminative and generative models. Finally, we discuss how to assess ASR quality using Word Error Rate, Character Error Rate and Phone Error Rate using the Levenstein Algorithm.
    - Seminar: You have to implement the recusive and matrix Levenstein Algorithms

- [Week 4]: Generative State-Space ASR Models
    - Lecture: In this lecture we discuss HMM and HMM-DNN ASR systems. We introduct the concept of a Trellis, specify that there are inference and training Trellises, and discuss how to run Dynamic Programming algorithms on them. Specficially, we discuss the Forward, Backward, Forward-Backward and Viterbi Algorithms on the Trellises. We close with a discussion of Baum-Welch training and HMM-DNN systems.
    - Seminar: How to Present Papers.

- [Week 5](https://github.com/yandexdataschool/speech_course/blob/main/week_05): Discriminative State-Space ASR Models
    - Lecture: In this lecture we dicuss Discriminative State-Space ASR systems, specifically Connectionist Temporal Classification. We discuss important difference in model structure and speech units between HMMs and CTC, inference and training trellises and the Forward-Backward algorithm for CTC. We close with a discussion where we contrast HMMs and CTC, and show the primary similarities and differences. 
    - Seminar: Implement CTC Forward-Backward Algorithm

- [Week 6](https://github.com/yandexdataschool/speech_course/blob/main/week_05): Context Modelling and Language Model Fusion
    - Lecture: In this lecture we analyse the errors typically made by a CTC system and use this to motivate a need for language modelling. We introduct language models and how they are typically evaluated. Then we take a closer look at N-Gram and Neural Language Models. Then we look at how we can use N-Gram Language models to do Prefix Search Decoding for CTC models. 
    - Seminar + Homework: You have to implement the recusive and matrix Levenstein Algorithms
    
- [Week 7]: Keynote Lecture from Professor Hermann Ney 
    - Lecture: Professor Hermann Ney dicusses Statistical Decision Theory in the context of ASR. 
    - Seminar: Assistance with Homework

- [Week 8]: Attention-based ASR Systems
    - Lecture: We close our discussion of ASR systems by examining Attention-based Autoregressive ASR models. We dicuss models such as Listen, Attend and Spell, examine advantages and limitations of such models and how they can be overcome. 
    - Seminar: Assistance with Homework



## Contributors & course staff

- Andrey Malinin - Course admin, lectures, seminars, homeworks
- Vladimir Kirichenko - lectures, seminars, homeworks
- Segey Dukanov - lecures, seminars, homeworks
- Yulia Gusak - seminars
- Ivan Provilkov - seminars
- Michael Solotky - seminars
- JustHeustic - seminars 
