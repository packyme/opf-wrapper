import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = os.getenv("OPF_MODEL_ID", "openai/privacy-filter")
REVISION = os.getenv("OPF_REVISION", "main")
MODEL_PATH = Path(os.getenv("OPF_MODEL_PATH", "./model")).expanduser()
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH
MODEL_PATH = MODEL_PATH.resolve()
MODEL_FILE = os.getenv("OPF_MODEL_FILE", "model.safetensors")
DEVICE = os.getenv("OPF_DEVICE", "cpu")
N_CTX = os.getenv("OPF_N_CTX")
INFERENCE_BATCH_SIZE = os.getenv("OPF_INFERENCE_BATCH_SIZE", "1")
