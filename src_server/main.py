from flask import Flask, abort, request
import whisper
import tempfile
from pathlib import Path


app = Flask(__name__)

@app.route("/")
def index():
    """
    ホームページの表示
    """
    display_text = """
    <h1>Welcome to the Whisper API!</h1>
    """
    return display_text


@app.route("/whisper", methods=["POST"])
def whisper_handler():
    """
    whisperのリクエストハンドリグ
    """
    if not request.files:
        # リクエストにファイルが含まれていない場合は, HTTP Exception 400を返す
        abort(400)

    # temporary storage of received audio file
    audio_file = request.files["audio_file"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_filepath = Path(temp_file.name)
        audio_file.save(temp_filepath)

    # load whisper-model
    model = whisper.load_model("small")

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(temp_filepath)
    audio = whisper.pad_or_trim(audio)

    # delete temporary file
    temp_filepath.unlink()

    # make log-Mel spectrogram and move to the same devices as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f"Detected language: {detected_language}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    return result.text


if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
