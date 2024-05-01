import logging.config
import torch
import whisper
from enum import Enum
from pathlib import Path


logging.config.fileConfig("/app/logging.conf")
logger = logging.getLogger("openai_whisper")


class ModelName(Enum):
    largeV3 = "large-v3"
    largeV2 = "large-v2"
    largeV1 = "large-v1"
    medium = "medium"
    small = "small"
    base = "base"
    tiny = "tiny"


def load_openai_whisper_model(model_name: str):
    logger.info(f"Start to load OpenAI {model_name} model")
    model_filepath = Path("/app/src/openai_whisper_server/models") / (model_name + ".pt")
    try:
        if torch.cuda.is_available():
            model = whisper.load_model(model_filepath, device="cuda")
        else:
            model = whisper.load_model(model_filepath, device="cpu")
    except torch.cuda.OutOfMemoryError as e:
        # CUDA out of memory
        model = whisper.load_model(model_filepath, device="cpu")
    except Exception as e:
        raise e

    logger.info(f"Finish to load OpenAI {model_name} model")
    return model
