FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt update && apt -y upgrade
RUN apt install -y python3-pip

RUN apt-get update && apt-get install -y ffmpeg

# install packages
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# download model
COPY download_model.py .
RUN python3 download_model.py tiny
RUN python3 download_model.py base
RUN python3 download_model.py small
RUN python3 download_model.py medium
RUN python3 download_model.py large-v1
RUN python3 download_model.py large-v2
RUN python3 download_model.py large-v3
