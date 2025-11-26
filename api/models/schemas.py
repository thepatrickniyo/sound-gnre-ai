"""
Pydantic schemas for API request/response models
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    pass  # File upload handled separately


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    genre: str = Field(..., description="Predicted genre")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)
    all_predictions: Dict[str, float] = Field(..., description="Confidence scores for all genres")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_files: int = Field(..., description="Total number of files processed")
    successful: int = Field(..., description="Number of successful predictions")
    failed: int = Field(..., description="Number of failed predictions")


class TrainingRequest(BaseModel):
    """Request model for training."""
    model_type: str = Field(default="random_forest", description="Type of model to train")
    test_size: float = Field(default=0.2, description="Proportion for test set", ge=0.1, le=0.5)
    tune_hyperparameters: bool = Field(default=False, description="Enable hyperparameter tuning")
    version: Optional[str] = Field(default=None, description="Model version name")


class TrainingResponse(BaseModel):
    """Response model for training."""
    status: str = Field(..., description="Training status")
    model_version: Optional[str] = Field(None, description="Model version")
    message: str = Field(..., description="Status message")
    job_id: Optional[str] = Field(None, description="Job ID for async training")


class RetrainingRequest(BaseModel):
    """Request model for retraining."""
    model_version: Optional[str] = Field(None, description="Model version to retrain (None = latest)")
    combine_strategy: str = Field(default="append", description="How to combine datasets: append, replace, merge")
    use_existing_data: bool = Field(default=True, description="Whether to use existing training data")
    new_version: Optional[str] = Field(None, description="Version name for new model")


class RetrainingResponse(BaseModel):
    """Response model for retraining."""
    status: str = Field(..., description="Retraining status")
    old_version: Optional[str] = Field(None, description="Previous model version")
    new_version: Optional[str] = Field(None, description="New model version")
    message: str = Field(..., description="Status message")
    job_id: Optional[str] = Field(None, description="Job ID for async retraining")


class TrainingStatusResponse(BaseModel):
    """Response model for training status."""
    job_id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Current status: pending, running, completed, failed")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)", ge=0, le=100)
    message: Optional[str] = Field(None, description="Status message")
    result: Optional[Dict[str, Any]] = Field(None, description="Training results if completed")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="API status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")


class ModelStatusResponse(BaseModel):
    """Response model for model status."""
    model_version: str = Field(..., description="Current model version")
    model_type: str = Field(..., description="Model type")
    training_date: Optional[str] = Field(None, description="Training date")
    test_accuracy: Optional[float] = Field(None, description="Test accuracy")
    is_loaded: bool = Field(..., description="Whether model is loaded")


class MetricsResponse(BaseModel):
    """Response model for model metrics."""
    accuracy: float = Field(..., description="Accuracy")
    precision: float = Field(..., description="Precision (weighted)")
    recall: float = Field(..., description="Recall (weighted)")
    f1_score: float = Field(..., description="F1-Score (weighted)")
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Per-class metrics")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

