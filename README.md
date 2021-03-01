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
    

## Contributors & course staff

- Andrey Malinin - Course admin, lectures, seminars, homeworks
- Vladimir Kirichenko - lectures, seminars, homeworks
- Segey Dukanov - lecures, seminars, homeworks
- Yulia Gusak - seminars
- Ivan Provilkov - seminars
- Michael Solotky - seminars
- JustHeustic - seminars 
