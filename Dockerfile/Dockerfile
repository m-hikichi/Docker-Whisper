FROM python:3.10

RUN apt-get update && apt-get install -y ffmpeg

RUN pip install --upgrade pip

# install packages
COPY requirements.txt .
RUN pip install -r requirements.txt
