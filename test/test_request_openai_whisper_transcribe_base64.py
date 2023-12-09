import pytest
import requests
import base64


@pytest.fixture()
def base64_audio():
    with open("/app/test/voicevox.wav", "rb") as f:
        audio = f.read()
        b64_audio = base64.b64encode(audio).decode("utf-8")
    return b64_audio


@pytest.mark.parametrize(
    "model_name",
    [
        "tiny",
        "base",
        "small",
        "medium",
        "large-v1",
        "large-v2",
        "large-v3",
    ]
)
def test_successful_request(base64_audio, model_name):
    # GIVEN
    url = "http://openai_whisper_server:5000/transcribe_base64"
    data = {"b64_audio": base64_audio, "model_name": model_name}

    # WHEN
    response = requests.post(url, json=data)

    # THEN
    assert response.status_code == 200


@pytest.mark.parametrize(
    "model_name",
    [
        "test",
    ]
)
def test_invalid_model_name_request(base64_audio, model_name):
    # GIVEN
    url = "http://openai_whisper_server:5000/transcribe_base64"
    data = {"b64_audio": base64_audio, "model_name": model_name}

    # WHEN
    response = requests.post(url, json=data)

    # THEN
    assert response.status_code == 400


@pytest.mark.parametrize(
    "invalid_base64_audio, model_name",
    [
        ("", "test"),
        ("a", "test"),
        ("abc", "test"),
    ]
)
def test_invalid_base64_audio_request(invalid_base64_audio, model_name):
    # GIVEN
    url = "http://openai_whisper_server:5000/transcribe_base64"
    data = {"b64_audio": invalid_base64_audio, "model_name": model_name}

    # WHEN
    response = requests.post(url, json=data)

    # THEN
    assert response.status_code == 400
