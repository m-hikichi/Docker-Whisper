import tempfile
from pathlib import Path
import logging.config


logging.config.fileConfig("/app/logging.conf")
logger = logging.getLogger("faster_whisper")

def write_to_temporary_file(contents, suffix):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_filepath = Path(temp_file.name)
    temp_filepath.write_bytes(contents)
    logger.info("write to temporary file")

    return temp_filepath
