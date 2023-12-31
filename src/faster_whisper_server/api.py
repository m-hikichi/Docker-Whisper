from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from pathlib import Path
import base64
import logging.config

from utils.util import write_to_temporary_file
from faster_whisper_server.faster_whisper_tools import (
    ModelName,
    load_faster_whisper_model,
    faster_whisper_transcribe,
)


logging.config.fileConfig("/app/logging.conf")
logger = logging.getLogger("faster_whisper")


app = FastAPI(
    title="faster whisper API",
    description="",
)


class WhisperRequestModel(BaseModel):
    b64_audio: str = ""
    model_name: str = "small"


class TranscribeTextModel(BaseModel):
    transcribe_text: str


@app.get("/", description="ホームページの表示")
async def index():
    display_text = """
    <h1>Welcome to the faster whisper API!</h1>
    """
    return HTMLResponse(content=display_text)


@app.post(
    "/transcribe_file",
    response_model=TranscribeTextModel,
    description="音声ファイルを受け取り, 文字起こしした文章を返す",
)
async def transcribe_file(
    file: UploadFile = File(...), model_name: ModelName = Form(...)
):
    logger.info("/transcribe_file accessed")

    # temporary storage of received audio file
    contents = await file.read()
    temp_filepath = write_to_temporary_file(contents, Path(file.filename).suffix)

    # transcribe
    try:
        model = load_faster_whisper_model(model_name.value)
        transcribe_text = faster_whisper_transcribe(model, str(temp_filepath))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # delete temporary file
        temp_filepath.unlink()

    # response
    logger.info("response transcribe text")
    return TranscribeTextModel(transcribe_text=transcribe_text)


@app.post(
    "/transcribe_base64",
    response_model=TranscribeTextModel,
    description="base64形式の音声ファイルを受け取り, 文字起こしした文章を返す",
)
async def transcribe_base64(request: WhisperRequestModel):
    logger.info("/transcribe_base64 accessed")

    # return HTTP Exception 400 if b64_audio is blank
    if not request.b64_audio:
        raise HTTPException(status_code=400, detail="b64_audio is Empty")

    # decode received audio file
    try:
        b64_audio = request.b64_audio
        contents = base64.b64decode(b64_audio)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    # temporary storage of received audio file
    temp_filepath = write_to_temporary_file(contents, ".wav")

    # transcribe
    try:
        model = load_faster_whisper_model(request.model_name)
        transcribe_text = faster_whisper_transcribe(model, str(temp_filepath))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # delete temporary file
        temp_filepath.unlink()

    # response
    logger.info("response transcribe text")
    return TranscribeTextModel(transcribe_text=transcribe_text)
