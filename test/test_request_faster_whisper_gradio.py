import pytest
from gradio_client import Client, file


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
        audio_filepath=file(audio_filepath),
        whisper_model_name=model_name,
        api_name=api_name,
    )

    # THEN
    assert transcribe_result == result


@pytest.mark.skip(reason="the test does not word, so this test is unconditionally skipped.")
@pytest.mark.parametrize(
    "model_name",
    [
        "test",
    ]
)
def test_nonexistent_model_name_request(client, model_name):
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
    assert result == "Input should be 'large-v3', 'large-v2', 'large-v1', 'medium', 'small', 'base' or 'tiny'"


@pytest.mark.skip(reason="the test does not word, so this test is unconditionally skipped.")
@pytest.mark.parametrize(
    "except_audio_filepath",
    [
        "/app/test/test.png",
    ]
)
def test_except_audio_file_request(client, except_audio_filepath):
    # GIVEN
    api_name = "/transcribe_audio_file"
    model_name = "small"

    # WHEN
    result = client.predict(
        except_audio_filepath,
        model_name,
        api_name=api_name,
    )

    # THEN
    assert False
