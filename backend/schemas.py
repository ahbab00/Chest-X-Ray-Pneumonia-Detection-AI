from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_available: bool


class PredictionResponse(BaseModel):
    label: Literal["NORMAL", "PNEUMONIA"]
    confidence: float = Field(ge=0, le=1)
    normal_probability: float = Field(ge=0, le=1)
    pneumonia_probability: float = Field(ge=0, le=1)
    threshold: float = Field(gt=0, lt=1)
    disclaimer: str
