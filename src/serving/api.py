"""
FastAPI application for serving ML models from MLflow.

This module provides a RESTful API for making predictions using
models stored in MLflow Model Registry.
"""

import os
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.serving.schemas import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelMetadata,
    ErrorResponse
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("model-serving-api")

# Get configurations from environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "default-model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
API_VERSION = os.getenv("API_VERSION", "1.0.0")

# Connect to MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Initialize FastAPI app
app = FastAPI(
    title="Model Serving API",
    description="API for serving machine learning models from MLflow Model Registry",
    version=API_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the loaded model
_model = None
_model_version = None
_model_load_time = None
_model_metadata = None


def get_model():
    """
    Load the model from MLflow if not already loaded.
    
    Returns:
        The loaded model
    """
    global _model, _model_version, _model_load_time, _model_metadata
    
    if _model is None:
        try:
            logger.info(f"Loading model '{MODEL_NAME}' in stage '{MODEL_STAGE}' from MLflow")
            
            # Get model version info
            client = mlflow.tracking.MlflowClient()
            model_version_infos = client.search_model_versions(f"name='{MODEL_NAME}'")
            filtered_mvs = [mv for mv in model_version_infos if mv.current_stage == MODEL_STAGE]
            
            if not filtered_mvs:
                logger.error(f"No model version found for '{MODEL_NAME}' in stage '{MODEL_STAGE}'")
                raise ValueError(f"No model version found for '{MODEL_NAME}' in stage '{MODEL_STAGE}'")
            
            # Use the latest version if multiple exist in the same stage
            model_version_info = sorted(filtered_mvs, key=lambda mv: int(mv.version), reverse=True)[0]
            _model_version = model_version_info.version
            
            # Load the model
            _model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
            _model_load_time = datetime.now().isoformat()
            
            # Extract model metadata
            run_id = model_version_info.run_id
            run = client.get_run(run_id)
            metrics = run.data.metrics
            tags = run.data.tags
            
            _model_metadata = {
                "name": MODEL_NAME,
                "version": _model_version,
                "description": tags.get("mlflow.note.content", "No description available"),
                "creation_time": model_version_info.creation_timestamp,
                "input_schema": {"features": tags.get("input_features", "Unknown")},
                "output_schema": {"prediction": tags.get("output_type", "Unknown")},
                "metrics": metrics,
            }
            
            logger.info(f"Model '{MODEL_NAME}' version {_model_version} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    
    return _model


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle request validation errors.
    """
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation error",
            error_code="VALIDATION_ERROR",
            details={"errors": str(exc)}
        ).dict(),
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check if the API and model are healthy.
    """
    global _model, _model_version
    
    model_loaded = _model is not None
    
    try:
        # Try to load the model if not loaded
        if not model_loaded:
            get_model()
            model_loaded = True
    except Exception as e:
        logger.warning(f"Health check: Model not loaded: {str(e)}")
        
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        model_loaded=model_loaded,
        model_version=_model_version if model_loaded else None
    )


@app.get("/metadata", response_model=ModelMetadata, tags=["Metadata"])
async def get_model_metadata():
    """
    Get metadata about the currently loaded model.
    """
    global _model_metadata
    
    # Ensure model is loaded
    get_model()
    
    if not _model_metadata:
        raise HTTPException(status_code=500, detail="Model metadata not available")
    
    return ModelMetadata(**_model_metadata)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a single prediction with the model.
    """
    try:
        # Ensure model is loaded
        model = get_model()
        
        # Extract features
        features = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction_start = time.time()
        prediction_result = model.predict(features)
        prediction_time = time.time() - prediction_start
        
        # Extract the result (assuming the model returns a numpy array or list)
        prediction_value = prediction_result[0]
        
        # Handle probabilities if available (for classification models)
        probability = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(features)[0]
                # For binary classification, return probability of the positive class
                if len(proba) == 2:
                    probability = float(proba[1])
                # For multiclass, return probability of the predicted class
                else:
                    probability = float(proba[prediction_value])
            except Exception as e:
                logger.warning(f"Could not get prediction probability: {str(e)}")
        
        # Create response
        response = PredictionResponse(
            prediction=prediction_value,
            probability=probability,
            request_id=request.request_id or str(uuid.uuid4()),
            model_version=_model_version,
            prediction_time=datetime.now().isoformat(),
            additional_info={
                "prediction_latency_ms": round(prediction_time * 1000, 2)
            }
        )
        
        return response
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=f"Prediction failed: {str(e)}",
                error_code="PREDICTION_ERROR",
                details={"exception": str(e)}
            ).dict()
        )


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Make batch predictions with the model.
    """
    try:
        # Ensure model is loaded
        model = get_model()
        
        # Extract features from all instances
        features_list = [instance.features for instance in request.instances]
        features_array = np.array(features_list)
        
        # Make predictions
        prediction_start = time.time()
        prediction_results = model.predict(features_array)
        prediction_time = time.time() - prediction_start
        
        # Handle probabilities if available (for classification models)
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(features_array)
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {str(e)}")
        
        # Create individual prediction responses
        predictions = []
        for i, instance in enumerate(request.instances):
            prediction_value = prediction_results[i]
            
            # Extract probability for this prediction
            probability = None
            if probabilities is not None:
                proba = probabilities[i]
                # For binary classification, return probability of the positive class
                if len(proba) == 2:
                    probability = float(proba[1])
                # For multiclass, return probability of the predicted class
                else:
                    probability = float(proba[prediction_value])
            
            predictions.append(
                PredictionResponse(
                    prediction=prediction_value,
                    probability=probability,
                    request_id=instance.request_id or str(uuid.uuid4()),
                    model_version=_model_version,
                    prediction_time=datetime.now().isoformat(),
                    additional_info={
                        "prediction_latency_ms": round((prediction_time * 1000) / len(request.instances), 2)
                    }
                )
            )
        
        # Create batch response
        response = BatchPredictionResponse(
            predictions=predictions,
            batch_id=request.batch_id or str(uuid.uuid4())
        )
        
        return response
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=f"Batch prediction failed: {str(e)}",
                error_code="BATCH_PREDICTION_ERROR",
                details={"exception": str(e)}
            ).dict()
        )


@app.post("/reload-model", response_model=HealthResponse, tags=["Admin"])
async def reload_model():
    """
    Force reloading of the model from MLflow.
    Requires authentication in a production environment.
    """
    global _model, _model_version, _model_load_time, _model_metadata
    
    # Reset model state
    _model = None
    _model_version = None
    _model_load_time = None
    _model_metadata = None
    
    # Load the model
    try:
        get_model()
        return HealthResponse(
            status="ok",
            version=API_VERSION,
            model_loaded=True,
            model_version=_model_version
        )
    except Exception as e:
        logger.error(f"Model reload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=f"Model reload failed: {str(e)}",
                error_code="MODEL_RELOAD_ERROR"
            ).dict()
        )


if __name__ == "__main__":
    import uvicorn
    
    # Load the model on startup
    try:
        get_model()
    except Exception as e:
        logger.warning(f"Could not load model on startup: {str(e)}")
    
    # Run with uvicorn when script is executed directly
    uvicorn.run(
        "src.serving.api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )

