# Homework Assingnment on Acoustic models

This is a seminar and homework assignment where you will get acquainted with the FastPitch architecture and related concepts. The notebooks for the seminar and homework are located in the folder `week_07_tts_am/notebooks`.

## How to Run the Assignment Code

### On the Beleriand Cluster
- Access Beleriand and follow the general instructions for Shad's clusters to enable Jupyter, TensorBoard, and set up Anaconda.
- Clone the course repository: `git clone https://github.com/yandexdataschool/speech_course.git`
- From the `week_07_tts_am` folder, execute the following command:
  ```
  chmod +x setup_env.sh && ./setup_env.sh
  ```
  This script will set up the FastPitch Conda environment with all necessary libraries.
- You can then use it in Jupyter by selecting a kernel named "fastpitch".

### In Google Colab
To ensure everything works in Colab, uncomment the corresponding lines in the notebooks. Note two significant limitations in Colab:

- If there's no interaction with the Colab window for more than 90 minutes, the session is interrupted.
- GPU usage time is severely limited per day.  

Therefore, it's recommended to debug everything on CPU and only switch to GPU when confident everything will work.