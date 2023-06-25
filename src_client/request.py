import requests


if __name__=="__main__":
    url = "http://whisper_server:5000/whisper"
    files = {"audio_file": open("/app/voice_recordings/voicevox.wav", "rb")}
    response = requests.post(url, files=files)
    print(response.text)