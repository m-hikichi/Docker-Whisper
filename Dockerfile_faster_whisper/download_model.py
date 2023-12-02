import argparse
from faster_whisper import WhisperModel


def download_model(model_name: str, device: str = "cpu", compute_type: str = "int8") -> WhisperModel:
    """
    指定されたモデルをダウンロード.

    Parameters:
        model_name (str): ダウンロードするモデルの名前.
        device (str, optional): モデルのデプロイ先デバイス. デフォルトは "cpu".
        compute_type (str, optional): モデルの計算タイプ. デフォルトは "int8".

    Returns:
        WhisperModel: ダウンロードされたWhisperモデル.
    """
    whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
    return whisper_model


def parse_arguments() -> argparse.Namespace:
    """
    コマンドライン引数を解析.

    Returns:
        argparse.Namespace: 解析されたコマンドライン引数.
    """
    parser = argparse.ArgumentParser(description="Download the specified model.")
    parser.add_argument("model_name", help="Specify the model to be downloaded.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    model = download_model(args.model_name, device="cpu", compute_type="int8")


if __name__ == "__main__":
    main()
