from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    text: str = Field(min_length=1)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class RedactRequest(DetectRequest):
    mask: str | None = None


class Detection(BaseModel):
    label: str
    text: str
    score: float
    start: int
    end: int


class DetectResponse(BaseModel):
    detections: list[Detection]


class RedactResponse(BaseModel):
    text: str
    detections: list[Detection]
