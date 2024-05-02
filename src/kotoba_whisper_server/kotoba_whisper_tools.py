import logging.config
import torch
from enum import Enum, unique
from transformers import pipeline


logging.config.fileConfig("/app/logging.conf")
logger = logging.getLogger("kotoba_whisper")


def load_kotoba_whisper_model() -> pipeline:
    logger.info("Start to load kotoba whisper model")

    model_dir_path = "/app/src/models/kotoba-whisper-v1.0"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_dir_path,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs=model_kwargs,
    )

    logger.info(f"Finish to load kotoba whisper model")
    return pipe


def kotoba_whisper_transcribe(pipe: pipeline, audio_filepath: str) -> str:
    logger.info("Start to transcribe")

    generate_kwargs = {"language": "japanese", "task": "transcribe"}
    result = pipe(audio_filepath, generate_kwargs=generate_kwargs)

    logger.info("Finish to transcribe")
    return result["text"]
