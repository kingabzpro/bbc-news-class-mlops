"""
News Classifier API v0.6
───────────────────────
FastAPI-based micro-service for text-classification inference.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import mlflow
import yaml
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Response, Security
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings

# ───────────────────────────────────────────────────────────────────────────────
# Settings
# ───────────────────────────────────────────────────────────────────────────────
load_dotenv()  # allow .env overrides for local dev


class Settings(BaseSettings):
    model_config = ConfigDict(extra="ignore")  # ignore stray env vars

    api_key: str | None = Field(default=None, json_schema_extra={"env": "API_KEY"})
    model_name: str = Field(
        "news_classifier_logistic", json_schema_extra={"env": "MODEL_NAME"}
    )
    model_version: str | int = Field(1, json_schema_extra={"env": "MODEL_VERSION"})
    cache_ttl: int = Field(
        3600, json_schema_extra={"env": "CACHE_TTL"}
    )  # Cache TTL in seconds (1 hour)
    tracking_uri: str | None = Field(
        None, json_schema_extra={"env": "MLFLOW_TRACKING_URI"}
    )

    @field_validator("model_version", mode="before")
    def _coerce_version(cls, v):
        return str(v) if v is not None else v

    @field_validator("cache_ttl", mode="before")
    def _parse_cache_ttl(cls, v):
        if isinstance(v, str):
            # Remove any comments and strip whitespace
            v = v.split("#")[0].strip()
        return int(v)


settings = Settings()
logger = logging.getLogger("news-api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent.parent / "models"
CONFIG_YAML = BASE_DIR.parent.parent / "configs" / "mlflow_config.yaml"

# Shared thread pool for every CPU-bound call
EXECUTOR = ThreadPoolExecutor(max_workers=4)


# ───────────────────────────────────────────────────────────────────────────────
# In-memory timed cache
# ───────────────────────────────────────────────────────────────────────────────
class TTLCache:
    def __init__(self, ttl_seconds: int):
        self._ttl = timedelta(seconds=ttl_seconds)
        self._store: Dict[str, Tuple[datetime, dict]] = {}

    def get(self, key: str) -> Optional[dict]:
        ts_val = self._store.get(key)
        if not ts_val:
            return None
        ts, val = ts_val
        if datetime.utcnow() - ts > self._ttl:
            del self._store[key]
            return None
        return val

    def set(self, key: str, value: dict) -> None:
        self._store[key] = (datetime.utcnow(), value)


prediction_cache = TTLCache(settings.cache_ttl)


# ───────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ───────────────────────────────────────────────────────────────────────────────
def _load_mlflow_cfg() -> dict:
    if CONFIG_YAML.exists():
        with CONFIG_YAML.open() as fh:
            return yaml.safe_load(fh)
    return {}


def _load_from_registry(name: str, version: str) -> Tuple[object, str]:
    """Return (model, resolved_version).  Raises on failure."""
    cfg = _load_mlflow_cfg()

    uri = cfg.get("tracking_uri") or settings.tracking_uri
    if not uri:
        raise RuntimeError("MLflow tracking URI not configured")
    mlflow.set_tracking_uri(uri)

    # Always use "latest" version regardless of input version
    model_uri = f"models:/{name}/latest"
    logger.info("Loading latest model from registry: %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)
    return model, "latest"


def _load_local_fallback() -> Tuple[object, str]:
    pattern = (MODELS_DIR / "news_classifier_*.joblib").as_posix()
    local_path = next(Path().glob(pattern), None)
    if not local_path:
        raise FileNotFoundError("No local model found matching pattern")
    logger.warning("Falling back to local model: %s", local_path)
    return joblib.load(local_path), "local"


# ───────────────────────────────────────────────────────────────────────────────
# FastAPI app & lifecycle
# ───────────────────────────────────────────────────────────────────────────────
app: FastAPI  # forward ref for typing
model: object | None = None
model_version: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, model_version
    try:
        # MLflow first, else disk
        model, model_version = await asyncio.to_thread(
            _load_from_registry, settings.model_name, settings.model_version
        )
    except Exception as e:
        logger.error("Registry load failed: %s", e)
        try:
            model, model_version = await asyncio.to_thread(_load_local_fallback)
        except Exception as inner:
            logger.critical("No model available: %s", inner)
            model = model_version = None
    yield
    EXECUTOR.shutdown(wait=True)


# ------------------------------------------------------------------------------
# App description (appears in /docs and /openapi.json)
# ------------------------------------------------------------------------------
APP_DESCRIPTION = """
Async micro-service that classifies short news headlines into predefined
categories using a scikit-learn model.  
•  **/info** — model & class metadata  
•  **/predict** — JSON inference endpoint (title → category + confidence)  
•  Prometheus metrics exposed at **/metrics**
"""

# ------------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------------
app = FastAPI(
    title="News Classifier API",
    version="0.6",
    description=APP_DESCRIPTION,  #  👈  added
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Metadata", "description": "Service health & model details"},
        {"name": "Inference", "description": "Predict category for a headline"},
    ],
)

# Static HTML/JS assets
templates_dir = BASE_DIR / "templates"
app.mount("/static", StaticFiles(directory=templates_dir), name="static")


# Prometheus metrics
class Metrics:
    _instance = None
    _initialized = False
    _registry = CollectorRegistry()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Metrics, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not Metrics._initialized:
            # Request metrics
            self.request_count = Counter(
                "http_requests_total",
                "Total number of HTTP requests",
                ["method", "endpoint", "status"],
                registry=self._registry,
            )

            self.request_latency = Histogram(
                "http_request_duration_seconds",
                "HTTP request latency in seconds",
                ["method", "endpoint"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
                registry=self._registry,
            )

            # Cache metrics
            self.cache_hits = Counter(
                "prediction_cache_hits_total",
                "Total number of cache hits for predictions",
                ["category"],
                registry=self._registry,
            )

            self.cache_misses = Counter(
                "prediction_cache_misses_total",
                "Total number of cache misses for predictions",
                ["category"],
                registry=self._registry,
            )

            # Prediction metrics
            self.prediction_counter = Counter(
                "news_predictions_total",
                "Total number of predictions made",
                ["category"],
                registry=self._registry,
            )

            self.prediction_confidence = Histogram(
                "news_prediction_confidence",
                "Confidence scores of predictions",
                ["category"],
                buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                registry=self._registry,
            )

            self.prediction_rate = Gauge(
                "news_prediction_rate",
                "Current prediction rate per category",
                ["category"],
                registry=self._registry,
            )
            Metrics._initialized = True


# Initialize metrics
metrics = Metrics()


# Metrics middleware
@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    # Record request metrics
    metrics.request_count.labels(
        method=request.method, endpoint=request.url.path, status=response.status_code
    ).inc()

    metrics.request_latency.labels(
        method=request.method, endpoint=request.url.path
    ).observe(duration)

    return response


# Metrics endpoint
@app.get("/metrics")
async def metrics_endpoint():
    return Response(generate_latest(metrics._registry), media_type=CONTENT_TYPE_LATEST)


# ───────────────────────────────────────────────────────────────────────────────
# Auth
# ───────────────────────────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(header: str = Security(api_key_header)):
    if settings.api_key and header != settings.api_key:
        detail = (
            "X-API-Key header missing" if header is None else "Invalid API key supplied"
        )
        raise HTTPException(status_code=403, detail=detail)
    return header


# ───────────────────────────────────────────────────────────────────────────────
# Schemas
# ───────────────────────────────────────────────────────────────────────────────
class NewsRequest(BaseModel):
    title: str = Field(..., min_length=3)


class Prediction(BaseModel):
    category: str
    confidence: float | None = Field(None, ge=0, le=1)


class Info(BaseModel):
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    classes: list[str] | None = None


# ───────────────────────────────────────────────────────────────────────────────
# Helper: run sync in threadpool
# ───────────────────────────────────────────────────────────────────────────────
async def _run_blocking(func, *args):
    return await asyncio.to_thread(func, *args)


# ───────────────────────────────────────────────────────────────────────────────
# End-points
# ───────────────────────────────────────────────────────────────────────────────
@app.get("/info", response_model=Info, tags=["Metadata"])
async def info() -> Info:
    return Info(
        model_loaded=model is not None,
        model_name=settings.model_name if model else None,
        model_version=model_version,
        classes=list(getattr(model, "classes_", [])) if model else None,
    )


@app.post(
    "/predict",
    response_model=Prediction,
    tags=["Inference"],
    dependencies=[Depends(get_api_key)],
)
async def predict(req: NewsRequest) -> Prediction:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    cached = prediction_cache.get(req.title)
    if cached:
        # Record cache hit
        metrics.cache_hits.labels(category=cached["category"]).inc()
        # Record prediction metrics for cache hits too
        metrics.prediction_counter.labels(category=cached["category"]).inc()
        if cached["confidence"] is not None:
            metrics.prediction_confidence.labels(category=cached["category"]).observe(
                cached["confidence"]
            )
        metrics.prediction_rate.labels(category=cached["category"]).inc()
        return Prediction(**cached)

    try:
        pred = await _run_blocking(model.predict, [req.title])
        cat = pred[0]
        conf = None
        if hasattr(model, "predict_proba"):
            proba = await _run_blocking(model.predict_proba, [req.title])
            conf = float(proba[0].max())

            # Record cache miss
            metrics.cache_misses.labels(category=cat).inc()

            # Record prediction metrics
            metrics.prediction_counter.labels(category=cat).inc()
            if conf is not None:
                metrics.prediction_confidence.labels(category=cat).observe(conf)
            metrics.prediction_rate.labels(category=cat).inc()

        result = {"category": cat, "confidence": conf}
        prediction_cache.set(req.title, result)
        return Prediction(**result)
    except Exception as e:
        logger.exception("Inference failure")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/", response_class=HTMLResponse, tags=["Metadata"])
async def root():
    return (templates_dir / "index.html").read_text()


# ───────────────────────────────────────────────────────────────────────────────
# Local dev entry-point
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="localhost",
        port=7860,
        # workers=4,
        reload=True,
        log_level="info",
    )
