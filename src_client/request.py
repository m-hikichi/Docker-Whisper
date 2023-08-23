import requests
import base64


if __name__=="__main__":
    url = "http://whisper_server:5000/whisper"
    with open("/app/voice_recordings/voicevox.wav", "rb") as f:
        audio = f.read()
        b64_audio = base64.b64encode(audio).decode("utf-8")

    """
    正常ケース
    """
    data = {"b64_audio": b64_audio}
    response = requests.post(url, json=data)
    print(response.text)

    data = {"b64_audio": b64_audio, "model_name": "medium"}
    response = requests.post(url, json=data)
    print(response.text)

    """
    エラーケース
    """
    data = {"b64_audio": b64_audio, "model_name": "test"}
    response = requests.post(url, json=data)
    print(response.text)

    data = {}
    response = requests.post(url, json=data)
    print(response.text)

    data = {"model_name": "test"}
    response = requests.post(url, json=data)
    print(response.text)

    data = {"b64_audio": "", "model_name": "small"}
    response = requests.post(url, json=data)
    print(response.text)

    data = {"b64_audio": "", "model_name": "test"}
    response = requests.post(url, json=data)
    print(response.text)

    data = {"b64_audio": "a", "model_name": "small"}
    response = requests.post(url, json=data)
    print(response.text)

    data = {"b64_audio": "a", "model_name": "test"}
    response = requests.post(url, json=data)
    print(response.text)