"""
Pydantic schemas for the FastAPI model serving application.

This module defines the data models used for request/response validation in the API.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
import numpy as np


class PredictionRequest(BaseModel):
    """
    Schema for prediction request data.
    
    Attributes:
        features (List[float]): List of feature values for a single prediction.
        feature_names (Optional[List[str]]): Names of the features (optional).
        request_id (Optional[str]): Unique identifier for the request for tracking.
    """
    features: List[float] = Field(..., description="List of feature values for prediction")
    feature_names: Optional[List[str]] = Field(None, description="Names of the features (optional)")
    request_id: Optional[str] = Field(None, description="Unique identifier for the request")
    
    @validator('features')
    def validate_features_length(cls, v):
        """Validate that features list is not empty."""
        if len(v) == 0:
            raise ValueError("Features list cannot be empty")
        return v
    
    @validator('feature_names')
    def validate_feature_names_length(cls, v, values):
        """Validate that feature_names matches features length if provided."""
        if v is not None and 'features' in values and len(v) != len(values['features']):
            raise ValueError(f"Feature names length ({len(v)}) must match features length ({len(values['features'])})")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0.1, 0.2, 0.3, 0.4, 0.5],
                "feature_names": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
                "request_id": "abc-123-xyz"
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Schema for batch prediction requests.
    
    Attributes:
        instances (List[PredictionRequest]): List of prediction request instances.
        batch_id (Optional[str]): Unique identifier for the batch request.
    """
    instances: List[PredictionRequest] = Field(..., description="List of prediction request instances")
    batch_id: Optional[str] = Field(None, description="Unique identifier for the batch request")
    
    @validator('instances')
    def validate_instances(cls, v):
        """Validate that instances list is not empty."""
        if len(v) == 0:
            raise ValueError("Instances list cannot be empty")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "instances": [
                    {
                        "features": [0.1, 0.2, 0.3, 0.4, 0.5],
                        "feature_names": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
                        "request_id": "abc-123-xyz"
                    },
                    {
                        "features": [0.5, 0.4, 0.3, 0.2, 0.1],
                        "feature_names": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
                        "request_id": "def-456-uvw"
                    }
                ],
                "batch_id": "batch-789"
            }
        }


class PredictionResponse(BaseModel):
    """
    Schema for prediction response data.
    
    Attributes:
        prediction (Union[int, float, str]): The predicted value.
        probability (Optional[float]): Probability or confidence score (0-1) if applicable.
        request_id (Optional[str]): Echo of the request ID from the request.
        model_version (str): Version of the model used for prediction.
        prediction_time (str): Timestamp of when the prediction was made.
        additional_info (Optional[Dict[str, Any]]): Any additional information to include.
    """
    prediction: Union[int, float, str] = Field(..., description="The predicted value")
    probability: Optional[float] = Field(None, description="Probability or confidence score (0-1) if applicable")
    request_id: Optional[str] = Field(None, description="Echo of the request ID from the request")
    model_version: str = Field(..., description="Version of the model used for prediction")
    prediction_time: str = Field(..., description="Timestamp of when the prediction was made")
    additional_info: Optional[Dict[str, Any]] = Field(None, description="Any additional information to include")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.95,
                "request_id": "abc-123-xyz",
                "model_version": "1",
                "prediction_time": "2025-05-07T00:20:19.497301+00:00",
                "additional_info": {"feature_importance": {"feature_1": 0.2, "feature_2": 0.8}}
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Schema for batch prediction responses.
    
    Attributes:
        predictions (List[PredictionResponse]): List of prediction responses.
        batch_id (Optional[str]): Echo of the batch ID from the request.
    """
    predictions: List[PredictionResponse] = Field(..., description="List of prediction responses")
    batch_id: Optional[str] = Field(None, description="Echo of the batch ID from the request")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": 1,
                        "probability": 0.95,
                        "request_id": "abc-123-xyz",
                        "model_version": "1",
                        "prediction_time": "2025-05-07T00:20:19.497301+00:00",
                        "additional_info": {"feature_importance": {"feature_1": 0.2, "feature_2": 0.8}}
                    },
                    {
                        "prediction": 0,
                        "probability": 0.12,
                        "request_id": "def-456-uvw",
                        "model_version": "1",
                        "prediction_time": "2025-05-07T00:20:20.123456+00:00",
                        "additional_info": {"feature_importance": {"feature_1": 0.7, "feature_2": 0.3}}
                    }
                ],
                "batch_id": "batch-789"
            }
        }


class HealthResponse(BaseModel):
    """
    Schema for health check response.
    
    Attributes:
        status (str): Status of the API ("ok", "error", etc.).
        version (str): API version.
        model_loaded (bool): Whether the model is loaded and ready.
        model_version (str): Version of the loaded model.
    """
    status: str = Field(..., description="Status of the API")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")
    model_version: Optional[str] = Field(None, description="Version of the loaded model")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "ok",
                "version": "1.0.0",
                "model_loaded": True,
                "model_version": "1"
            }
        }


class ModelMetadata(BaseModel):
    """
    Schema for model metadata.
    
    Attributes:
        name (str): Name of the model.
        version (str): Version of the model.
        description (Optional[str]): Description of the model.
        creation_time (str): When the model was created.
        input_schema (Dict[str, Any]): Description of the expected input.
        output_schema (Dict[str, Any]): Description of the model output.
        metrics (Optional[Dict[str, Any]]): Performance metrics of the model.
    """
    name: str = Field(..., description="Name of the model")
    version: str = Field(..., description="Version of the model")
    description: Optional[str] = Field(None, description="Description of the model")
    creation_time: str = Field(..., description="When the model was created")
    input_schema: Dict[str, Any] = Field(..., description="Description of the expected input")
    output_schema: Dict[str, Any] = Field(..., description="Description of the model output")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics of the model")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "classification-model",
                "version": "1",
                "description": "Binary classification model trained on synthetic data",
                "creation_time": "2025-05-01T12:00:00+00:00",
                "input_schema": {
                    "features": "array of 5 numerical values",
                    "feature_names": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
                },
                "output_schema": {
                    "prediction": "binary classification (0 or 1)",
                    "probability": "confidence score between 0 and 1"
                },
                "metrics": {
                    "accuracy": 0.95,
                    "f1": 0.94,
                    "auc": 0.97
                }
            }
        }


class ErrorResponse(BaseModel):
    """
    Schema for error responses.
    
    Attributes:
        error (str): Error message.
        error_code (str): Error code for programmatic handling.
        details (Optional[Dict[str, Any]]): Additional error details if available.
    """
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code for programmatic handling")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Invalid input: Features list has incorrect length",
                "error_code": "INVALID_INPUT",
                "details": {
                    "expected_length": 5,
                    "received_length": 3
                }
            }
        }

