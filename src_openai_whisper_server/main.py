from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from pydantic import  BaseModel
from enum import Enum
import whisper
import torch
import tempfile
from pathlib import Path
import base64
import logging.config


logging.config.fileConfig("/app/logging.conf")
logger = logging.getLogger("openai_whisper")


app = FastAPI(
    title="openai whisper API",
    description="",
)


class WhisperRequestModel(BaseModel):
    b64_audio: str = ""
    model_name: str = "small"


class TranscribeTextModel(BaseModel):
    transcribe_text: str


class ModelName(Enum):
    largeV3 = "large-v3"
    largeV2 = "large-v2"
    largeV1 = "large-v1"
    medium = "medium"
    small = "small"
    base = "base"
    tiny = "tiny"


@app.get(
    "/",
    description="ホームページの表示"
)
async def index():
    display_text = """
    <h1>Welcome to the openai whisper API!</h1>
    """
    return HTMLResponse(content=display_text)


@app.post(
    "/transcribe_file",
    response_model=TranscribeTextModel,
    description="音声ファイルを受け取り, 文字起こしした文章を返す"
)
async def transcribe_file(file: UploadFile = File(...), model_name: ModelName = Form(...)):
    logger.info("/transcribe_file accessed")
    
    # temporary storage of received audio file
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as temp_file:
        temp_filepath = Path(temp_file.name)
    temp_filepath.write_bytes(contents)

    # transcribe
    try:
        model = load_model(model_name.value)
        result = model.transcribe(str(temp_filepath))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # delete temporary file
        temp_filepath.unlink()

    # response
    return TranscribeTextModel(transcribe_text=result["text"])


@app.post(
    "/transcribe_base64",
    response_model=TranscribeTextModel,
    description="base64形式の音声ファイルを受け取り, 文字起こしした文章を返す"
)
async def transcribe_base64(request: WhisperRequestModel):
    logger.info("/transcribe_base64 accessed")

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
        model = load_model(request.model_name)
        result = model.transcribe(str(temp_filepath))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # delete temporary file
        temp_filepath.unlink()

    # response
    return TranscribeTextModel(transcribe_text=result["text"])


def load_model(model_name: str):
    try:
        if torch.cuda.is_available():
            model = whisper.load_model(model_name, device="cuda")
        else:
            model = whisper.load_model(model_name, device="cpu")
    except torch.cuda.OutOfMemoryError as e:
        # CUDA out of memory
        model = whisper.load_model(model_name, device="cpu")
    except Exception as e:
        raise e

    return model
