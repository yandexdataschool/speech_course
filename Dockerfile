FROM ubuntu:22.04
ENV DEBIAN_FRONTEND noninteractive

RUN useradd -ms /bin/bash --uid 1000 jupyter\
    && apt update\
    && apt install -y python3.11 python3.11-dev python3.11-distutils curl\
    && ln -s /usr/bin/python3.11 /usr/local/bin/python3\
    && curl https://bootstrap.pypa.io/get-pip.py | python3

RUN apt-get update -y && apt-get install make -y && apt-get install unzip -y && apt-get install git -y && apt-get install wget -y && apt-get install libsndfile1 -y

RUN apt update -y && apt install build-essential -y

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
    soundfile==0.13.1 \
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