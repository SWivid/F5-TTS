FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

USER root

ARG DEBIAN_FRONTEND=noninteractive

LABEL github_repo="https://github.com/SWivid/F5-TTS"

RUN set -x \
    && apt-get update \
    && apt-get -y install wget curl man git less openssl libssl-dev unzip unar build-essential aria2 tmux vim \
    && apt-get install -y openssh-server sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy the entire repository into the container
COPY . /workspace/F5-TTS

WORKDIR /workspace/F5-TTS

# Install dependencies for evaluation
RUN pip install -e .[eval]

ENV SHELL=/bin/bash

# Set entrypoint and command
# ENTRYPOINT ["/bin/bash", "/workspace/F5-TTS/entrypoint.sh"]
CMD ["bash", "-c", "f5-tts_infer-gradio --host 0.0.0.0 --port 7860"]
