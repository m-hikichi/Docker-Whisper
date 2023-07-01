from flask import Flask, abort, request
import whisper
import tempfile
from pathlib import Path
import base64


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
    b64_audio = request.files["b64_audio"]
    audio = base64.b64decode(b64_audio.read())
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_filepath = Path(temp_file.name)
    with open(str(temp_filepath), "wb") as f:
        f.write(audio)

    # load whisper-model
    model = whisper.load_model("small")

    # transcribe speech in audio file
    result = model.transcribe(str(temp_filepath))

    # delete temporary file
    temp_filepath.unlink()

    return result["text"]


if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
