from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import  BaseModel
import faster_whisper
import tempfile
from pathlib import Path
import base64


app = FastAPI(
    title="faster whisper API",
    description="",
)


class WhisperRequestModel(BaseModel):
    b64_audio: str = ""
    model_name: str = "small"


class TranscribeTextModel(BaseModel):
    transcribe_text: str


@app.get(
    "/",
    description="ホームページの表示"
)
async def index():
    display_text = """
    <h1>Welcome to the faster whisper API!</h1>
    """
    return HTMLResponse(content=display_text)


@app.post(
    "/transcribe_file",
    response_model=TranscribeTextModel,
    description="音声ファイルを受け取り, 文字起こしした文章を返す"
)
async def transcribe_file(file: UploadFile = File(...)):
    # temporary storage of received audio file
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as temp_file:
        temp_filepath = Path(temp_file.name)
    temp_filepath.write_bytes(contents)

    # transcribe
    try:
        model = model = faster_whisper.WhisperModel("small", device="cpu", compute_type="int16")
        segments, info = model.transcribe(str(temp_filepath), beam_size=5)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # delete temporary file
        temp_filepath.unlink()

    # response
    transcribe_text = ""
    for segment in segments:
        transcribe_text += segment.text
    return TranscribeTextModel(transcribe_text=transcribe_text)


@app.post(
    "/transcribe_base64",
    response_model=TranscribeTextModel,
    description="base64形式の音声ファイルを受け取り, 文字起こしした文章を返す"
)
async def transcribe_base64(request: WhisperRequestModel):
    # return HTTP Exception 400 if b64_audio is blank
    if not request.b64_audio:
        raise HTTPException(status_code=400, detail="b64_audio is Empty")

    # decode received audio file
    try:
        b64_audio = request.b64_audio
        audio = base64.b64decode(b64_audio)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    # temporary storage of received audio file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_filepath = Path(temp_file.name)
    with open(str(temp_filepath), "wb") as f:
        f.write(audio)

    # transcribe
    try:
        model = model = faster_whisper.WhisperModel(request.model_name, device="cpu", compute_type="int16")
        segments, info = model.transcribe(str(temp_filepath), beam_size=5)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # delete temporary file
        temp_filepath.unlink()

    # response
    transcribe_text = ""
    for segment in segments:
        transcribe_text += segment.text
    return TranscribeTextModel(transcribe_text=transcribe_text)
