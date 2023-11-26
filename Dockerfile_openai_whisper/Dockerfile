FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt update && apt -y upgrade
RUN apt install -y python3-pip

RUN apt-get update && apt-get install -y ffmpeg

# download model
RUN apt-get install -y wget
RUN mkdir -p /root/.cache/whisper/
# Reference: URL to download (https://github.com/openai/whisper/blob/main/whisper/__init__.py)
RUN wget -P /root/.cache/whisper/ https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt
RUN wget -P /root/.cache/whisper/ https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt
RUN wget -P /root/.cache/whisper/ https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt
RUN wget -P /root/.cache/whisper/ https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt
RUN wget -P /root/.cache/whisper/ https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt
RUN wget -P /root/.cache/whisper/ https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt
RUN wget -P /root/.cache/whisper/ https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt

# install packages
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
