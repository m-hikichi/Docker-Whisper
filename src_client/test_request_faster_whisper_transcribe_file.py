import pytest
import requests


@pytest.fixture()
def audio():
    with open("/app/src/voicevox.wav", "rb") as f:
        audio = f.read()
    return audio


@pytest.mark.parametrize(
    "model_name",
    [
        "small",
        "medium",
    ]
)
def test_successful_request(audio, model_name):
    # GIVEN
    url = "http://faster_whisper_server:5000/transcribe_file"
    files = {"file" : audio}
    data = {"model_name": model_name}

    # WHEN
    response = requests.post(url, files=files, data=data)

    # THEN
    assert response.status_code == 200
