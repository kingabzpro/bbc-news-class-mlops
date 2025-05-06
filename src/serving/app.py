"""
FastAPI application for serving machine learning models via a REST API.

This module provides endpoints for:
- Model predictions (single and batch)
- Health checks
- Model metadata
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request, status
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("model-serving")

# Initialize FastAPI app
app = FastAPI(
    title="MLOps Model API",
    description="API for serving machine learning models",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify your allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "mlops-model")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")  # Use Production, Staging, or None
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Global variables for the model
model = None
model_version = None
model_info = None
model_load_time = None


def get_model():
    """
    Get the loaded model. If not loaded, attempt to load it.
    
    Returns:
        The loaded MLflow model.
        
    Raises:
        HTTPException: If the model cannot be loaded.
    """
    global model, model_version, model_info, model_load_time
    
    if model is None:
        try:
            # Set up MLflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
            # Load the model from MLflow Model Registry
            if MODEL_STAGE:
                model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
                logger.info(f"Loading model from {model_uri}")
                model = mlflow.pyfunc.load_model(model_uri)
            else:
                # If no stage is specified, load the latest version
                client = mlflow.tracking.MlflowClient()
                latest_versions = client.get_latest_versions(MODEL_NAME)
                if not latest_versions:
                    raise ValueError(f"No versions found for model {MODEL_NAME}")
                
                latest_version = latest_versions[0]
                model_version = latest_version.version
                logger.info(f"Loading model version {model_version}")
                model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{model_version}")
            
            # Get model metadata
            client = mlflow.tracking.MlflowClient()
            model_versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE] if MODEL_STAGE else None)
            
            if model_versions:
                model_info = model_versions[0]
                model_version = model_info.version
                logger.info(f"Model {MODEL_NAME} version {model_version} loaded successfully")
            else:
                logger.warning(f"No model metadata found for {MODEL_NAME}")
            
            model_load_time = datetime.now().isoformat()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model could not be loaded: {str(e)}",
            )
    
    return model


def log_prediction(request_data: Dict, response_data: Dict, batch: bool = False):
    """
    Log prediction requests and responses for monitoring.
    
    Args:
        request_data: The prediction request data.
        response_data: The prediction response data.
        batch: Whether this is a batch prediction.
    """
    try:
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request": request_data,
            "response": response_data,
            "batch": batch,
            "model_name": MODEL_NAME,
            "model_version": model_version,
        }
        
        # Log to file system (could be replaced with more sophisticated logging)
        log_dir = os.environ.get("LOG_DIR", "./logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Use current date for log file name
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"predictions_{current_date}.jsonl")
        
        # Append log entry to the file
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # In a production system, you might send logs to:
        # - Evidently AI for drift detection
        # - Prometheus for metrics
        # - Kafka for stream processing
        # - MLflow for experiment tracking
        # - Database for structured storage
        
        logger.debug(f"Logged prediction: {log_entry}")
    except Exception as e:
        # Log but don't fail the request if logging fails
        logger.error(f"Error logging prediction: {e}")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors from Pydantic models."""
    details = {}
    for error in exc.errors():
        loc = ".".join(str(x) for x in error["loc"])
        details[loc] = error["msg"]
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Request validation error",
            error_code="VALIDATION_ERROR",
            details=details
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle any unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            details={"message": str(exc)}
        ).dict(),
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API is running and model is loaded.
    """
    global model, model_version, model_load_time
    
    is_model_loaded = model is not None
    
    try:
        # Try to load the model if not already loaded
        if not is_model_loaded:
            _ = get_model()
            is_model_loaded = True
    except Exception:
        # If model loading fails, still return status but indicate model not loaded
        is_model_loaded = False
    
    return HealthResponse(
        status="ok",
        version=app.version,
        model_loaded=is_model_loaded,
        model_version=model_version if is_model_loaded else None
    )


@app.get("/metadata", response_model=ModelMetadata, tags=["Metadata"])
async def get_model_metadata():
    """
    Get metadata about the deployed model.
    """
    # Ensure model is loaded to get its metadata
    current_model = get_model()
    
    if not model_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metadata not available",
        )
    
    # Get model run from MLflow to extract metrics
    client = mlflow.tracking.MlflowClient()
    model_run = client.get_run(model_info.run_id)
    
    # Extract model schema if available
    model_signature = getattr(current_model, 'metadata', {}).get('signature', None)
    input_schema = {}
    output_schema = {}
    
    if model_signature:
        input_schema = {
            "type": str(model_signature.inputs),
            "features": [str(col) for col in model_signature.inputs.input_names()] if hasattr(model_signature.inputs, 'input_names') else []
        }
        output_schema = {
            "type": str(model_signature.outputs),
            "features": [str(col) for col in model_signature.outputs.input_names()] if hasattr(model_signature.outputs, 'input_names') else []
        }
    else:
        # Provide default schema if model signature not available
        input_schema = {
            "type": "List of numerical features",
            "features": "Unknown"
        }
        output_schema = {
            "type": "Model prediction",
            "format": "Unknown"
        }
    
    # Extract metrics from the run
    metrics = model_run.data.metrics if model_run and hasattr(model_run.data, 'metrics') else {}
    
    return ModelMetadata(
        name=MODEL_NAME,
        version=model_version,
        description=model_info.description if model_info.description else "No description available",
        creation_time=model_info.creation_timestamp.isoformat() if hasattr(model_info, 'creation_timestamp') else model_load_time,
        input_schema=input_schema,
        output_schema=output_schema,
        metrics=metrics
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    model_instance: Any = Depends(get_model)
):
    """
    Make a prediction with the loaded model.
    """
    prediction_time = datetime.now()
    prediction_time_iso = prediction_time.isoformat()
    
    try:
        # Prepare features for prediction
        features = np.array(request.features).reshape(1, -1)
        
        # If feature names are provided, create a DataFrame with column names
        if request.feature_names:
            features_df = pd.DataFrame([request.features], columns=request.feature_names)
            prediction_result = model_instance.predict(features_df)
        else:
            # Otherwise use numpy array
            prediction_result = model_instance.predict(features)
        
        # Extract prediction from result (handle different return types)
        if isinstance(prediction_result, np.ndarray):
            if len(prediction_result.shape) > 1 and prediction_result.shape[1] > 1:
                # Multi-class probabilities
                prediction_value = int(np.argmax(prediction_result[0]))
                probability = float(np.max(prediction_result[0]))
            else:
                # Binary or regression prediction
                prediction_value = float(prediction_result[0]) if isinstance(prediction_result[0], (np.float32, np.float64)) else int(prediction_result[0])
                probability = None
        else:
            # Handle other return types
            prediction_value = prediction_result
            probability = None
        
        # Try to get probability if model has predict_proba method
        if probability is None and hasattr(model_instance, 'predict_proba'):
            try:
                if request.feature_names:
                    proba = model_instance.predict_proba(features_df)
                else:
                    proba = model_instance.predict_proba(features)
                
                if isinstance(proba, np.ndarray) and proba.shape[1] >= 2:
                    # For binary classification, typically use the probability of class 1
                    probability = float(proba[0, 1])
                elif isinstance(proba, np.ndarray):
                    probability = float(proba[0])
            except Exception as e:
                logger.warning(f"Failed to get probability: {e}")
                probability = None
        
        # Create response
        response = PredictionResponse(
            prediction=prediction_value,
            probability=probability,
            request_id=request.request_id,
            model_version=model_version,
            prediction_time=prediction_time_iso,
            additional_info={}
        )
        
        # Log prediction in background
        background_tasks.add_task(
            log_prediction,
            request_data=request.dict(),
            response_data=response.dict(),
            batch=False
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    model_instance: Any = Depends(get_model)
):
    """
    Make batch predictions with the loaded model.
    """
    try:
        # Process each instance in the batch
        predictions = []
        for instance in request.instances:
            prediction_time = datetime.now()
            prediction_time_iso = prediction_time.isoformat()
            
            # Prepare features for prediction
            features = np.array(instance.features).reshape(1, -1)
            
            # Make prediction
            if instance.feature_names:
                features_df = pd.DataFrame([instance.features], columns=instance.feature_names)
                prediction_result = model_instance.predict(features_df)
            else:
                prediction_result = model_instance.predict(features)
            
            # Extract prediction
            if isinstance(prediction_result, np.ndarray):
                if len(prediction_result.shape) > 1 and prediction_result.shape[1] > 1:
                    # Multi-class probabilities
                    prediction_value = int(np.argmax(prediction_result[0]))
                    probability = float(np.max(prediction_result[0]))
                else:
                    # Binary or regression prediction
                    prediction_value = float(prediction_result[0]) if isinstance(prediction_result[0], (np.float32, np.float64)) else int(prediction_result[0])
                    probability = None
            else:
                prediction_value = prediction_result
                probability = None
            
            # Try to get probability
            if probability is None and hasattr(model_instance, 'predict_proba'):
                try:
                    if instance.feature_names:
                        proba = model_instance.predict_proba(features_df)
                    else:
                        proba = model_instance.predict_proba(features)
                    
                    if isinstance(proba, np.ndarray) and proba.shape[1] >= 2:
                        probability = float(proba[0, 1])
                    elif isinstance(proba, np.ndarray):
                        probability = float(proba[0])
                except Exception as e:
                    logger.warning(f"Failed to get probability: {e}")
                    probability = None
            
            # Create response for this instance
            prediction_response = PredictionResponse(
                prediction=prediction_value,
                probability=probability,
                request_id=instance.request_id,
                model_version=model_version,
                prediction_time=prediction_time_iso,
                additional_info={}
            )
            
            predictions.append(prediction_response)
        
        # Create batch response
        batch_response = BatchPredictionResponse(
            predictions=predictions,
            batch_id=request.batch_id
        )
        
        # Log batch prediction in background
        background_tasks.add_task(
            log_prediction,
            request_data=request.dict(),
            response_data=batch_response.dict(),
            batch=True
        )
        
        return batch_response
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making batch prediction: {str(e)}",
        )


@app.on_event("startup")
async def startup_event():
    """
    Initialize resources when the application starts.
    """
    logger.info("Starting up model serving application")
    try:
        # Attempt to load the model at startup (optional)
        if os.environ.get("PRELOAD_MODEL", "true").lower() == "true":
            logger.info("Preloading model...")
            _ = get_model()
            logger.info("Model preloaded successfully")
    except Exception as e:
        # Log but don't fail startup, model will be loaded on first request
        logger.warning(f"Failed to preload model: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up resources when the application shuts down.
    """
    logger.info("Shutting down model serving application")
    # Add any cleanup code here if needed


def main():
    """
    Run the FastAPI application using uvicorn.
    
    This function is the entry point when running the app directly.
    """
    import uvicorn
    
    # Get configuration from environment variables or use defaults
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    log_level = os.environ.get("LOG_LEVEL", "info")
    
    # Run the server
    uvicorn.run(
        "src.serving.app:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=os.environ.get("DEBUG", "false").lower() == "true"
    )


if __name__ == "__main__":
    main()

