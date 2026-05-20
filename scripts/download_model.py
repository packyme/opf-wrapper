from huggingface_hub import snapshot_download

from app.config import MODEL_ID, MODEL_PATH, ONNX_FILE, ONNX_SUBFOLDER, REVISION


def main() -> None:
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=str(MODEL_PATH),
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "viterbi_calibration.json",
            f"{ONNX_SUBFOLDER}/{ONNX_FILE}",
            f"{ONNX_SUBFOLDER}/{ONNX_FILE}_data*",
        ],
    )

    print(f"Downloaded {MODEL_ID}@{REVISION} to {MODEL_PATH}")
    print(f"ONNX file: {ONNX_SUBFOLDER}/{ONNX_FILE}")


if __name__ == "__main__":
    main()
