import logging.config
from enum import Enum, unique

import torch
from faster_whisper import WhisperModel


logging.config.fileConfig("/app/logging.conf")
logger = logging.getLogger("faster_whisper")


@unique
class ModelName(Enum):
    largeV3 = "large-v3"
    largeV2 = "large-v2"
    largeV1 = "large-v1"
    medium = "medium"
    small = "small"
    base = "base"
    tiny = "tiny"


def load_faster_whisper_model(model_name):
    try:
        if torch.cuda.is_available():
            model = WhisperModel(model_name, device="cuda", compute_type="float16", local_files_only=True)
        else:
            model = WhisperModel(model_name, device="cpu", compute_type="int8", local_files_only=True)
    except RuntimeError as e:
        # CUDA out of memory
        model = WhisperModel(model_name, device="cpu", compute_type="int8", local_files_only=True)
    except Exception as e:
        raise e

    logger.info(f"load {model_name} model")
    return model
