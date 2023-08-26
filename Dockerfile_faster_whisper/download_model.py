import faster_whisper


for model_name in ["tiny", "base", "small", "medium"]:
    _ = faster_whisper.WhisperModel(model_name, device="cpu", compute_type="int16")
