"""
News Classifier API Version 0.5
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import mlflow
import yaml
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
MODEL_NAME = os.getenv("MODEL_NAME", "news_classifier_logistic")
MODEL_VERSION = os.getenv("MODEL_VERSION", 1)

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "mlflow_config.yaml"
model = None
current_model_version = MODEL_VERSION
thread_pool = ThreadPoolExecutor(max_workers=4)  # Adjust based on your CPU cores


def load_mlflow_config():
    """Load MLflow configuration from YAML file"""
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_model_from_registry(model_name: str, version: str = None) -> Tuple[object, str]:
    """
    Load the latest model from MLflow Model Registry.

    Args:
        model_name: Name of the model in the registry

    Returns:
        Tuple of (loaded_model, version_number)
    """
    config = load_mlflow_config()
    mlflow.set_tracking_uri(config["tracking_uri"])

    try:
        if not current_model_version:
            return None, None

        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Failed to load model from MLflow: {e}")
        return None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, current_model_version
    try:
        # Try loading from MLflow registry first
        model = load_model_from_registry(MODEL_NAME, current_model_version)
        print(
            f"Loaded {MODEL_NAME} version {current_model_version} from MLflow registry"
        )
        if model is None:
            # Fallback to local model if MLflow loading fails
            model_path = next(MODEL_DIR.glob("news_classifier_*.joblib"), None)
            if model_path and model_path.exists():
                model = joblib.load(model_path)
                current_model_version = "local"
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        current_model_version = None
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
    model_name: Optional[str] = None
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

    return {
        "model_loaded": True,
        "model_name": MODEL_NAME,
        "model_version": current_model_version,
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
        # workers=4,
        log_level="info",
        access_log=True,
        timeout_keep_alive=30,
    )
