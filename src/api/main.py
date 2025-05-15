"""
News Classifier API Version 0.2
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import joblib
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------------------
load_dotenv()

API_KEY = os.getenv("API_KEY")

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = next(MODEL_DIR.glob("news_classifier_*.joblib"), None)
model = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    if MODEL_PATH and MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    yield


# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------
app = FastAPI(
    title="News Classifier API",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Metadata"},
        {"name": "Inference"},
    ],
)

# ---------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------
Instrumentator().instrument(app).expose(app)


# ---------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(api_key_from_header: str = Security(api_key_header)):
    if (
        not API_KEY
    ):  # Allow access if API_KEY is not set in the environment (e.g. for local dev without .env)
        return
    if not api_key_from_header:
        raise HTTPException(
            status_code=403, detail="Not authenticated: X-API-Key header missing."
        )
    if api_key_from_header != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key_from_header


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------
class NewsRequest(BaseModel):
    title: str


class Prediction(BaseModel):
    category: str
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class Info(BaseModel):
    model_loaded: bool
    model_version: Optional[str] = None
    classes: Optional[List[str]] = None


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/info", response_model=Info, tags=["Metadata"])
def info():
    if model is None:
        return {
            "model_loaded": False,
        }

    version = MODEL_PATH.stem.split("_")[-1]
    return {
        "model_loaded": True,
        "model_version": version,
        "classes": list(getattr(model, "classes_", [])),
    }


@app.post(
    "/predict",
    response_model=Prediction,
    tags=["Inference"],
    dependencies=[Depends(get_api_key)],
)
def predict(req: NewsRequest):
    if model is None:
        raise HTTPException(503, "Model not available")

    pred = model.predict([req.title])[0]
    conf = (
        float(model.predict_proba([req.title])[0].max())
        if hasattr(model, "predict_proba")
        else None
    )
    return {"category": pred, "confidence": conf}


# ---------------------------------------------------------------------
# Local run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="localhost", port=8000, reload=True)
