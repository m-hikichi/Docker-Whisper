FROM python:3.10

RUN apt-get update && apt-get install -y ffmpeg

# install packages
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# download model
COPY download_model.py .
RUN python download_model.py
