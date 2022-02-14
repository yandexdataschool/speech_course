# YSDA Speech Processing Course

- Materials for each week are in ./week* folders

## Course program

- Week 1: Introduction to Speech
    - Lecture: In this lecture we introduce the area of speech processing, discuss historical background and current trends. In the second half of the lecture we introduce the concept fo speech as a separate modality from text or images and foreshadow concepts from later lectures.
  
- [Week 2](https://github.com/yandexdataschool/speech_course/blob/main/week_02): Digital Signal Processing
    - Lecture: In this lecture we discuss how to transform an audio signal into a form which is convenient for use in Speech Recognition and Synthesis. We discuss: how an audio wave is sampled and digitized; The Fourier Transform and the Discrete Fourier Transform and how they can be used to obtain the frequency spectrum of the signal; How to use the Short-Time-Fourier-Transform to represent sound as a Spectrogram; finally, we discuss the Mel-Scale and how to obtain a Mel-Spectrogram. 
    - Seminar: In part 1 we will implement the Short-Time-Fourier-Transform and obtain a Mel-Spectrogram. In part 2 we will: recover a Spectrogram from a Mel-Spectrogram. Reconstruct the original audio signal via the Griffin-Lim algorithm and do some simple voice warping. 
    - Homework: Audio-MNIST: Implement a Neural Network model to do simple digit classification based on a mel-spectrogram. 



## Contributors & course staff

- Andrey Malinin - Course admin, lectures, seminars, homeworks
- Vladimir Kirichenko - lectures, seminars, homeworks
- Segey Dukanov - lecures, seminars, homeworks
