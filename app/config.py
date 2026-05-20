import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = os.getenv("OPF_MODEL_ID", "openai/privacy-filter")
REVISION = os.getenv("OPF_REVISION", "main")
MODEL_PATH = Path(os.getenv("OPF_MODEL_PATH", "./model")).expanduser()
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH
MODEL_PATH = MODEL_PATH.resolve()
ONNX_SUBFOLDER = os.getenv("OPF_ONNX_SUBFOLDER", "onnx")
ONNX_FILE = os.getenv("OPF_ONNX_FILE", "model_q4.onnx")
PROVIDER = os.getenv("OPF_PROVIDER", "CPUExecutionProvider")
N_CTX = os.getenv("OPF_N_CTX")
