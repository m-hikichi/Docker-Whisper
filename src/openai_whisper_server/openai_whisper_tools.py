import logging.config
from enum import Enum

import torch
import whisper


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

    logger.info(f"load {model_name} model")
    return model
