import faster_whisper


for model_name in ["tiny", "base", "small", "medium", "large-v1", "large-v2"]:
    _ = faster_whisper.WhisperModel(model_name, device="cpu", compute_type="int8")
