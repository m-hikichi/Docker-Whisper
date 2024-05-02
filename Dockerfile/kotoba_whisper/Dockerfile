FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt update && apt -y upgrade
RUN apt install -y python3-pip

RUN apt-get update && apt-get install -y ffmpeg

# install packages
COPY requirements.txt .
RUN pip install -r requirements.txt
