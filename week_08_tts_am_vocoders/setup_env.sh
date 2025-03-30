#!/bin/bash

set -euo pipefail

conda-activate

conda create -y -n fastpitch python=3.9.16
source activate fastpitch

conda install -y -c conda-forge cudatoolkit==11.6.0
conda install -y -c conda-forge cudnn==8.4.1.50

pip install --no-input --extra-index-url https://download.pytorch.org/whl/cu116 \
  torch==1.13.1 \
  torchaudio==0.13.1

pip install --no-input \
librosa==0.9.0 \
matplotlib==3.7.1 \
g2p_en==2.1.0 \
numpy==1.23.5 \
pandas==1.2.2 \
praat-parselmouth==0.4.3 \
soundfile==0.12.1 \
tensorboardX==2.6.2 \
torchmetrics==0.11.1 \
tqdm==4.65.0 \
transformers==4.31.0

pip install --no-input ipykernel==6.13.0 ipywidgets==8.0.6
python -m ipykernel install --user --name fastpitch --display-name "fastpitch"