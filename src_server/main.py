from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import  BaseModel
import whisper
import tempfile
from pathlib import Path
import base64


app = FastAPI(
    title="whisper API",
    description="",
)


class WhisperRequestModel(BaseModel):
    b64_audio: str
    model_name: str = "small"


class TranscribeTextModel(BaseModel):
    transcribe_text: str


@app.get(
    "/",
    description="ホームページの表示"
)
def index():
    display_text = """
    <h1>Welcome to the Whisper API!</h1>
    """
    return HTMLResponse(content=display_text)


@app.post(
    "/whisper",
    response_model=TranscribeTextModel,
    description="base64形式の音声ファイルを受け取り, 文字起こしした文章を返す"
)
def whisper_handler(request: WhisperRequestModel):
    if not request.b64_audio:
        # return HTTP Exception 400 if b64_audio is blank
        raise HTTPException(status_code=400, detail="audio file is None")

    # temporary storage of received audio file
    b64_audio = request.b64_audio
    audio = base64.b64decode(b64_audio)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_filepath = Path(temp_file.name)
    with open(str(temp_filepath), "wb") as f:
        f.write(audio)

    # load whisper-model
    try:
        model = whisper.load_model(request.model_name)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # transcribe speech in audio file
    result = model.transcribe(str(temp_filepath))

    # delete temporary file
    temp_filepath.unlink()

    # return JSONResponse(content={"transcribe_text": result["text"]})
    return TranscribeTextModel(transcribe_text=result["text"])
