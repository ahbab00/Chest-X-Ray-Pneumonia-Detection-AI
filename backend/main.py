import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from backend.config import (
    ALLOWED_CONTENT_TYPES,
    DISCLAIMER,
    FRONTEND_DIR,
    MAX_UPLOAD_BYTES,
    METRICS_PATH,
    MODEL_PATH,
)
from backend.schemas import HealthResponse, PredictionResponse
from backend.services.model_service import InvalidImageError, ModelUnavailableError, predict

app = FastAPI(
    title="Pneumonia CNN Research API",
    version="1.0.0",
    description="Research-only chest X-ray binary classification API.",
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(model_available=MODEL_PATH.is_file())


@app.get("/api/metrics")
def metrics():
    if not METRICS_PATH.is_file():
        return JSONResponse({"available": False})
    try:
        payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        evaluated_model = payload.get("model_path")
        if not evaluated_model or MODEL_PATH.resolve() != Path(evaluated_model).resolve():
            return JSONResponse({"available": False})
        return {"available": True, **payload}
    except (json.JSONDecodeError, OSError, TypeError):
        return JSONResponse({"available": False})


@app.post("/api/predict", response_model=PredictionResponse)
async def classify_xray(file: UploadFile = File(...)) -> PredictionResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only JPEG and PNG images are supported.",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image size must not exceed 10 MB.")

    try:
        result = await run_in_threadpool(predict, content)
    except InvalidImageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=500, detail="The model returned an invalid score.") from error

    return PredictionResponse(**result, disclaimer=DISCLAIMER)


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
