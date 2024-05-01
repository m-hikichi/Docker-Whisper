import logging.config
import torch
from enum import Enum, unique
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions


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


def load_faster_whisper_model(model_name: str) -> WhisperModel:
    try:
        if torch.cuda.is_available():
            model = WhisperModel(
                model_name, device="cuda", compute_type="float16", local_files_only=True
            )
            logger.info("Device: GPU")
        else:
            model = WhisperModel(
                model_name, device="cpu", compute_type="int8", local_files_only=True
            )
            logger.info("Device: CPU")
    except RuntimeError as e:
        logger.warning(e)
        # CUDA out of memory
        model = WhisperModel(
            model_name, device="cpu", compute_type="int8", local_files_only=True
        )
        logger.info("Device: CPU")
    except Exception as e:
        raise e

    logger.info(f"Finish to load {model_name} model")
    return model


def faster_whisper_transcribe(model: WhisperModel, audio_filepath: str) -> str:
    logger.info("Start to transcribe")
    transcribe_text = ""

    vad_options = VadOptions(min_silence_duration_ms=1000)
    segments, info = model.transcribe(
        audio_filepath,
        beam_size=5,
        without_timestamps=True,
        vad_filter=True,
        vad_parameters=vad_options,
    )

    transcribe_text = "\n".join(segment.text for segment in segments)

    logger.info("Finish to transcribe")
    return transcribe_text
