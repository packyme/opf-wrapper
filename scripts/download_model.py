from huggingface_hub import snapshot_download

from app.config import MODEL_FILE, MODEL_ID, MODEL_PATH, REVISION


def main() -> None:
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=str(MODEL_PATH),
        allow_patterns=[
            "config.json",
            MODEL_FILE,
            "tokenizer.json",
            "tokenizer_config.json",
            "viterbi_calibration.json",
        ],
    )

    print(f"Downloaded {MODEL_ID}@{REVISION} to {MODEL_PATH}")
    print(f"Model file: {MODEL_FILE}")


if __name__ == "__main__":
    main()
