"""
News Classifier API Version 0.4
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import joblib
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
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
thread_pool = ThreadPoolExecutor(max_workers=4)  # Adjust based on your CPU cores


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    if MODEL_PATH and MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    yield
    thread_pool.shutdown()


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

# Mount static files
templates_dir = Path(__file__).parent / "templates"
app.mount("/static", StaticFiles(directory=templates_dir), name="static")

# ---------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------
Instrumentator().instrument(app).expose(app)


# ---------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(api_key_from_header: str = Security(api_key_header)):
    if not API_KEY:  # Allow access if API_KEY is not set in the environment
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
# Helper Functions
# ---------------------------------------------------------------------
async def run_in_threadpool(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(thread_pool, func, *args)


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/info", response_model=Info, tags=["Metadata"])
async def info():
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
async def predict(req: NewsRequest):
    if model is None:
        raise HTTPException(503, "Model not available")

    # Run prediction in threadpool to avoid blocking
    pred = await run_in_threadpool(model.predict, [req.title])
    pred = pred[0]

    # Get confidence if available
    conf = None
    if hasattr(model, "predict_proba"):
        proba = await run_in_threadpool(model.predict_proba, [req.title])
        conf = float(proba[0].max())

    return {"category": pred, "confidence": conf}


@app.get("/", response_class=HTMLResponse, tags=["Metadata"])
async def root():
    template_path = templates_dir / "index.html"
    return template_path.read_text()


# ---------------------------------------------------------------------
# Local run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    # Run with multiple workers
    uvicorn.run(
        "src.api.main:app",
        host="localhost",
        port=8000,
        workers=4,
        log_level="info",
        access_log=True,
        timeout_keep_alive=30,
    )
