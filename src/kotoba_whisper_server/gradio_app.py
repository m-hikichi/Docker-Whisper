import logging.config
import argparse
import gradio as gr
from pathlib import Path
from kotoba_whisper_tools import (
    load_kotoba_whisper_model,
    kotoba_whisper_transcribe,
)


# ロギング設定
logging.config.fileConfig("/app/logging.conf")
logger = logging.getLogger("kotoba_whisper")


def parse_arguments() -> argparse.Namespace:
    """
    コマンドライン引数を解析.

    Returns:
        argparse.Namespace: 解析されたコマンドライン引数.
    """
    parser = argparse.ArgumentParser(
        description="Launches a simple web server that serves the demo."
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to create a publicly shareable link for the interface.",
    )
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default="7860")
    return parser.parse_args()


def transcribe_audio_file(audio_filepath: str) -> str:
    audio_filepath = Path(audio_filepath)

    whisper_model = load_kotoba_whisper_model()
    transcribe_text = kotoba_whisper_transcribe(whisper_model, str(audio_filepath))

    audio_filepath.unlink()

    return transcribe_text


def build_transcribe_ui() -> gr.blocks.Blocks:
    with gr.Blocks() as app:
        gr.Markdown(
            """
            # kotoba Whisper Transcribe
            """
        )
        with gr.Column():
            audio = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio",
            )
            
            transcribe_button = gr.Button("transcribe")

            transcribe_textbox = gr.Textbox(
                label="Transcribe Text",
                info="文字起こしされたテキストが表示されます",
            )

        transcribe_button.click(
            fn=transcribe_audio_file,
            inputs=[audio],
            outputs=transcribe_textbox,
        )

    return app


if __name__ == "__main__":
    args = parse_arguments()
    app = build_transcribe_ui()
    app.launch(
        share=args.share, server_name=args.server_name, server_port=args.server_port
    )
