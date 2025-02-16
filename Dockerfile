FROM ubuntu:focal


RUN apt-get update -y && apt-get install make -y && apt-get install unzip -y && apt-get install git -y

RUN apt-get update -y && apt-get install python3-pip -y

RUN useradd -ms /bin/bash --uid 1000 jupyter

RUN apt update && apt install build-essential

RUN pip3 install \
    numpy \
    pandas \
    Pillow \
    scikit-learn \
    scikit-image \
    scipy \
    torch \
    torchaudio \
    torchvision \
    librosa \
    catboost \
    tqdm \
    matplotlib \
    jupyterlab \
    editdistance \
    transformers==4.31.0 \
    soundfile==0.12.1 \
    jiwer \
    datasets \
    g2p_en==2.1.0 \
    praat-parselmouth==0.4.3 \
    tensorboardX==2.6.2 \
    torchmetrics==0.11.1 \
    ipykernel \
    ipywidgets \
    lightning \
    deep-phonemizer \
    pyannote.audio \
    pyloudnorm \
    sacrebleu \
    protobuf \
    nltk \
    && apt-get clean && rm -rf /tmp/*
