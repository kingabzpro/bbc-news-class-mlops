"""
Minimal FastAPI app for serving the latest trained model.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = next(MODEL_DIR.glob("news_classifier_*.joblib"), None)

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if MODEL_PATH and MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        model = None
    yield


app = FastAPI(title="News Classifier API", lifespan=lifespan)


class NewsRequest(BaseModel):
    title: str


@app.post("/predict")
def predict(request: NewsRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available.")
    pred = model.predict([request.title])[0]
    return {"category": pred}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="localhost", port=8000, reload=True)
