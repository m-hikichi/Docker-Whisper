import pytest
from gradio_client import Client


@pytest.fixture()
def client():
    client = Client("http://faster_whisper_server:7860/")
    return client


# テスト音声には voicevox の「ずんだもん（ノーマル）」を使用
@pytest.mark.parametrize(
    "model_name, transcribe_result",
    [
        ("tiny", "おはようございます。"),
        ("base", "おはようございます"),
        ("small", "おはようございます"),
        ("medium", "おはようございます"),
        ("large-v1", "おはようございます"),
        ("large-v2", "おはようございます"),
        ("large-v3", "おはようございます"),
    ]
)
def test_successful_request(client, model_name, transcribe_result):
    # GIVEN
    api_name = "/transcribe_audio_file"
    audio_filepath = "/app/test/voicevox.wav"

    # WHEN
    result = client.predict(
        audio_filepath,
        model_name,
        api_name=api_name,
    )

    # THEN
    assert transcribe_result == result
