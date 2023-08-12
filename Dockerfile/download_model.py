import whisper


MODEL_DIR = "/root/.cache/whisper"

for model_name in ["tiny", "base", "small", "medium"]:
    _ = whisper.load_model(model_name, download_root=MODEL_DIR)
