from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from app.config import DEVICE, MODEL_FILE, MODEL_ID, MODEL_PATH, PROFILE
from app.privacy_filter import get_classifier, is_classifier_loaded, load_classifier, redact_text, run_detection
from app.schemas import DetectRequest, DetectResponse, RedactRequest, RedactResponse


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    load_classifier()
    yield


app = FastAPI(title="OpenAI Privacy Filter Wrapper", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    runtime = get_classifier() if is_classifier_loaded() else None

    return {
        "status": "ok",
        "model": MODEL_ID,
        "model_path": str(MODEL_PATH),
        "weights": MODEL_FILE,
        "model_loaded": str(is_classifier_loaded()).lower(),
        "backend": "pytorch",
        "device": DEVICE,
        "actual_device": str(runtime.model.device) if runtime is not None else "",
        "n_ctx": str(runtime.n_ctx) if runtime is not None else "",
        "inference_batch_size": str(runtime.inference_batch_size) if runtime is not None else "",
        "decoder": runtime.decoder_mode if runtime is not None else "",
        "viterbi_backend": runtime.viterbi_backend if runtime is not None else "",
        "decoder_backend": decoder_backend(runtime),
        "profile": str(PROFILE).lower(),
    }


def decoder_backend(runtime: Any | None) -> str:
    if runtime is None:
        return ""
    if runtime.decoder_mode == "argmax":
        return "argmax"
    if runtime.viterbi_backend == "cuda" and runtime.model.device.type == "cuda":
        return "torch_cuda"
    if runtime.viterbi_backend == "dense":
        return "numpy_dense"
    return "numpy_sparse"


@app.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest) -> DetectResponse:
    try:
        detections = run_detection(request.text, request.threshold)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return DetectResponse(detections=detections)


@app.post("/redact", response_model=RedactResponse)
def redact(request: RedactRequest) -> RedactResponse:
    try:
        detections = run_detection(request.text, request.threshold)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    redacted_text = redact_text(request.text, detections, request.mask)
    return RedactResponse(text=redacted_text, detections=detections)
