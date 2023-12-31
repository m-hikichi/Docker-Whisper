import gradio as gr
import logging.config
import argparse
from pathlib import Path
from faster_whisper_server.faster_whisper_tools import (
    ModelName,
    load_faster_whisper_model,
    faster_whisper_transcribe,
)
import requests
import json


# ロギング設定
logging.config.fileConfig("/app/logging.conf")
logger = logging.getLogger("faster_whisper")


def parse_arguments() -> argparse.Namespace:
    """
    コマンドライン引数を解析.

    Returns:
        argparse.Namespace: 解析されたコマンドライン引数.
    """
    parser = argparse.ArgumentParser(description="Launches a simple web server that serves the demo.")
    parser.add_argument("--share", action="store_true", help="Whether to create a publicly shareable link for the interface.")
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default="7860")
    return parser.parse_args()


def transcribe_audio_file(audio_filepath: str, whisper_model_name: str) -> str:
    url = "http://faster_whisper_server:5000/transcribe_file"
    audio_filepath = Path(audio_filepath)

    with open(audio_filepath, "rb") as f:
        audio = f.read()
    files = {"file" : audio}
    data = {"model_name": whisper_model_name}

    audio_filepath.unlink()

    response = requests.post(url, files=files, data=data)
    if response.status_code == 200:
        response_dict = json.loads(response.text)
        return response_dict["transcribe_text"]
    else:
        response_dict = json.loads(response.text)
        return response_dict["detail"][0]["msg"]


def build_transcribe_ui() -> gr.blocks.Blocks:
    with gr.Blocks() as app:
        gr.Markdown(
            """
            # Faster Whisper Transcribe
            """
        )
        with gr.Column():
            with gr.Row():
                audio = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Audio",
                )
                whisper_model_names = tuple(model_name.value for model_name in ModelName)
                model_name_dropdown = gr.Dropdown(
                    choices=whisper_model_names,
                    value=whisper_model_names[0],
                    label="Whisper Model Name",
                    info="文字起こしをする際に使用するwhisperのモデルを指定してください"
                )
            with gr.Row():
                transcribe_button = gr.Button("transcribe")
            with gr.Row():
                transcribe_textbox = gr.Textbox(
                    label="Transcribe Text",
                    info="文字起こしされたテキストが表示されます"
                )

        transcribe_button.click(
            fn=transcribe_audio_file,
            inputs=[audio, model_name_dropdown],
            outputs=transcribe_textbox
        )

    return app


if __name__=="__main__":
    args = parse_arguments()
    app = build_transcribe_ui()
    app.launch(share=args.share, server_name=args.server_name, server_port=args.server_port)