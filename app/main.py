from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import MODEL_ID, MODEL_PATH, ONNX_FILE, ONNX_SUBFOLDER, PROVIDER
from app.privacy_filter import is_classifier_loaded, load_classifier, redact_text, run_detection
from app.schemas import DetectRequest, DetectResponse, RedactRequest, RedactResponse


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    load_classifier()
    yield


app = FastAPI(title="OpenAI Privacy Filter Wrapper", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "model_path": str(MODEL_PATH),
        "model_loaded": str(is_classifier_loaded()).lower(),
        "onnx": f"{ONNX_SUBFOLDER}/{ONNX_FILE}",
        "provider": PROVIDER,
    }


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
