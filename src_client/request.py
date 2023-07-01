import requests
import base64


if __name__=="__main__":
    url = "http://whisper_server:5000/whisper"
    with open("/app/voice_recordings/voicevox.wav", "rb") as f:
        audio = f.read()
        b64_audio = base64.b64encode(audio)

    files = {"b64_audio": b64_audio}
    response = requests.post(url, files=files)
    print(response.text)