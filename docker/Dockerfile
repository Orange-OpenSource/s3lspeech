# Software Name : s3lspeech
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.

# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-04.html#rel-23-04
FROM nvcr.io/nvidia/pytorch:23.04-py3
WORKDIR /workspace
ENV PIP_INDEX_URL https://repos.tech.orange/artifactory/api/pypi/pythonproxy/simple
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 sox && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir \ 
    nvidia-cudnn-cu12==8.9.2.26 nvidia-cublas-cu12==12.1.3.1  
RUN python3 -m pip install --no-cache-dir \ 
    torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0  
RUN python3 -m pip install --no-cache-dir \
    sox==1.4.1 huggingface-hub==0.18.0 hyperpyyaml==1.2.2 \
    sentencepiece==0.1.99 gensim==4.3.2 webdataset==0.2.57 \
    pytorch-lightning==2.1.0
RUN python3 -m pip install --no-cache-dir --no-deps \
    speechbrain==0.5.15 torchtext==0.16
RUN python3 -m pip install --no-cache-dir --force-reinstall llvmlite==0.40.0 
RUN python3 -m pip install --no-build-isolation --no-cache-dir \
    flash-attn==2.2.0
RUN python3 -m pip install --no-cache-dir \
    omegaconf==2.3.0 flashlight-text==0.0.4