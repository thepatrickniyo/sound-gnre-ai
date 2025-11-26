"""
Monitoring routes for API and model status
"""
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status

from api.models.schemas import ModelStatusResponse, MetricsResponse, HealthResponse

router = APIRouter()

# API statistics (in production, use database or Redis)
_api_stats = {
    'total_requests': 0,
    'predictions': 0,
    'training_jobs': 0,
    'errors': 0,
    'start_time': datetime.now().isoformat()
}


def update_stats(endpoint: str, success: bool = True):
    """Update API statistics."""
    _api_stats['total_requests'] += 1
    if endpoint == 'predict':
        _api_stats['predictions'] += 1
    elif endpoint == 'train' or endpoint == 'retrain':
        _api_stats['training_jobs'] += 1
    if not success:
        _api_stats['errors'] += 1


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with API status
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )


@router.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status():
    """
    Get current model status and version.
    
    Returns:
        ModelStatusResponse with model information
    """
    try:
        from api.routes.prediction import _model, _processor
        
        if _model is None:
            return ModelStatusResponse(
                model_version="none",
                model_type="none",
                training_date=None,
                test_accuracy=None,
                is_loaded=False
            )
        
        # Get model metadata
        models_dir = Path("models")
        version_dirs = sorted([d for d in models_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('v')])
        
        if not version_dirs:
            return ModelStatusResponse(
                model_version="unknown",
                model_type=_model.model_type,
                training_date=None,
                test_accuracy=None,
                is_loaded=True
            )
        
        # Load metadata from latest version
        latest_version_dir = version_dirs[-1]
        model_files = list(latest_version_dir.glob("genre_classifier_*.pkl"))
        
        if model_files:
            model_type = model_files[0].stem.replace("genre_classifier_", "")
            metadata_file = latest_version_dir / f"genre_classifier_{model_type}_metadata.json"
            
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                return ModelStatusResponse(
                    model_version=metadata.get('version', latest_version_dir.name),
                    model_type=metadata.get('model_type', model_type),
                    training_date=metadata.get('training_date'),
                    test_accuracy=metadata.get('test_accuracy'),
                    is_loaded=True
                )
        
        return ModelStatusResponse(
            model_version=latest_version_dir.name,
            model_type=_model.model_type,
            training_date=None,
            test_accuracy=None,
            is_loaded=True
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model status: {str(e)}"
        )


@router.get("/metrics", response_model=MetricsResponse)
async def get_model_metrics():
    """
    Get model performance metrics.
    
    Returns:
        MetricsResponse with model metrics
    """
    try:
        # Load latest model metadata
        models_dir = Path("models")
        version_dirs = sorted([d for d in models_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('v')])
        
        if not version_dirs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No model found"
            )
        
        latest_version_dir = version_dirs[-1]
        model_files = list(latest_version_dir.glob("genre_classifier_*.pkl"))
        
        if not model_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No model file found"
            )
        
        model_type = model_files[0].stem.replace("genre_classifier_", "")
        metadata_file = latest_version_dir / f"genre_classifier_{model_type}_metadata.json"
        
        if not metadata_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model metadata not found"
            )
        
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Get per-class metrics if available
        per_class_metrics = None
        if 'test_precision' in metadata and 'test_recall' in metadata:
            # Try to get per-class metrics from training report
            report_files = list(Path("report").glob("training_summary_*.json"))
            if report_files:
                with open(report_files[-1], 'r') as f:
                    report = json.load(f)
                if 'per_class_metrics' in report:
                    per_class = report['per_class_metrics']
                    class_names = report['dataset_info']['class_names']
                    per_class_metrics = {
                        class_names[i]: {
                            'precision': per_class['precision'][i],
                            'recall': per_class['recall'][i],
                            'f1': per_class['f1'][i]
                        }
                        for i in range(len(class_names))
                    }
        
        return MetricsResponse(
            accuracy=metadata.get('test_accuracy', 0.0),
            precision=metadata.get('test_precision', 0.0),
            recall=metadata.get('test_recall', 0.0),
            f1_score=metadata.get('test_f1', 0.0),
            per_class_metrics=per_class_metrics
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting metrics: {str(e)}"
        )


@router.get("/stats")
async def get_api_stats():
    """
    Get API usage statistics.
    
    Returns:
        Dictionary with API statistics
    """
    uptime_seconds = (datetime.now() - datetime.fromisoformat(_api_stats['start_time'])).total_seconds()
    uptime_hours = uptime_seconds / 3600
    
    return {
        'total_requests': _api_stats['total_requests'],
        'predictions': _api_stats['predictions'],
        'training_jobs': _api_stats['training_jobs'],
        'errors': _api_stats['errors'],
        'uptime_seconds': uptime_seconds,
        'uptime_hours': round(uptime_hours, 2),
        'start_time': _api_stats['start_time'],
        'requests_per_hour': round(_api_stats['total_requests'] / uptime_hours, 2) if uptime_hours > 0 else 0
    }


@router.get("/monitoring/status")
async def monitoring_status():
    """Get monitoring service status."""
    return {
        "status": "active",
        "endpoints": {
            "health": "/api/v1/health",
            "model_status": "/api/v1/model/status",
            "metrics": "/api/v1/metrics",
            "stats": "/api/v1/stats"
        }
    }
