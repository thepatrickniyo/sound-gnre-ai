"""
Prediction routes for music genre classification
"""
import time
import tempfile
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

from api.models.schemas import PredictionResponse, BatchPredictionResponse, ErrorResponse

router = APIRouter()

# Global model variable
_model = None
_processor = None
_class_names = None


def load_model():
    """Load model for prediction (called on startup)."""
    global _model, _processor, _class_names
    try:
        from src.model import GenreClassifier
        from src.preprocessing import DataProcessor
        
        # Try to load latest model
        models_dir = Path("models")
        version_dirs = sorted([d for d in models_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('v')])
        if not version_dirs:
            return False
            
        model_file = list(version_dirs[-1].glob("genre_classifier_*.pkl"))[0]
        model_type = model_file.stem.replace("genre_classifier_", "")
        _model = GenreClassifier(model_type=model_type)
        _model.load_model(model_file)
        
        # Load processor
        _processor = DataProcessor(models_dir="models")
        _processor.load_processor()
        
        # Get class names from label encoder (most reliable source)
        if _processor.label_encoder and hasattr(_processor.label_encoder, 'classes_'):
            _class_names = _processor.label_encoder.classes_.tolist()
        else:
            # Fallback: Try to get from model metadata
            metadata_file = model_file.parent / f"genre_classifier_{model_type}_metadata.json"
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                _class_names = metadata.get('class_names', [])
            else:
                # Try to get from processor metadata
                metadata_path = Path("models/data_processor_metadata.json")
                if metadata_path.exists():
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    _class_names = metadata.get('class_names', [])
                else:
                    _class_names = None
        
        return True
    except Exception as e:
        print(f"Could not load model: {e}")
        return False


def _validate_audio_file(file: UploadFile) -> bool:
    """Validate uploaded audio file."""
    allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    file_ext = Path(file.filename).suffix.lower()
    return file_ext in allowed_extensions


def _extract_features_from_audio(audio_bytes: bytes, filename: str) -> np.ndarray:
    """Extract features from audio bytes."""
    from src.feature_extraction import FeatureExtractor
    
    extractor = FeatureExtractor()
    features = extractor.extract_features_from_upload(audio_bytes, filename=filename)
    
    # Remove metadata columns
    exclude_cols = ['filename', 'length']
    feature_dict = {k: v for k, v in features.items() if k not in exclude_cols}
    
    # Convert to array in correct order
    if _processor and _processor.feature_columns:
        feature_array = np.array([feature_dict.get(col, 0) for col in _processor.feature_columns])
    else:
        # Fallback: use all numeric features
        feature_array = np.array([v for k, v in feature_dict.items() if isinstance(v, (int, float))])
    
    return feature_array.reshape(1, -1)


def _make_prediction(features: np.ndarray) -> dict:
    """Make prediction using loaded model."""
    if _model is None or _processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure a model is trained and available."
        )
    
    # Scale features
    features_scaled = _processor.transform_features(
        pd.DataFrame(features, columns=_processor.feature_columns)
    )
    features_array = features_scaled.values
    
    # Make prediction
    prediction = _model.predict(features_array)[0]
    probabilities = _model.predict_proba(features_array)[0]
    
    # Get class names
    if _class_names:
        predicted_genre = _class_names[prediction]
        all_predictions = {_class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
    else:
        predicted_genre = f"class_{prediction}"
        all_predictions = {f"class_{i}": float(prob) for i, prob in enumerate(probabilities)}
    
    return {
        'genre': predicted_genre,
        'confidence': float(probabilities[prediction]),
        'all_predictions': all_predictions
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(file: UploadFile = File(...)):
    """
    Predict genre from a single audio file.
    
    Args:
        file: Audio file (WAV, MP3, FLAC, M4A, OGG)
        
    Returns:
        PredictionResponse with predicted genre and confidence scores
    """
    start_time = time.time()
    
    # Validate file
    if not _validate_audio_file(file):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format. Allowed: WAV, MP3, FLAC, M4A, OGG"
        )
    
    try:
        # Read file content
        audio_bytes = await file.read()
        
        # Extract features
        features = _extract_features_from_audio(audio_bytes, file.filename)
        
        # Make prediction
        prediction_result = _make_prediction(features)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return PredictionResponse(
            genre=prediction_result['genre'],
            confidence=prediction_result['confidence'],
            all_predictions=prediction_result['all_predictions'],
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing audio file: {str(e)}"
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict genres from multiple audio files.
    
    Args:
        files: List of audio files
        
    Returns:
        BatchPredictionResponse with predictions for all files
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    predictions = []
    successful = 0
    failed = 0
    
    for file in files:
        try:
            # Validate file
            if not _validate_audio_file(file):
                failed += 1
                continue
            
            start_time = time.time()
            
            # Read file content
            audio_bytes = await file.read()
            
            # Extract features
            features = _extract_features_from_audio(audio_bytes, file.filename)
            
            # Make prediction
            prediction_result = _make_prediction(features)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            predictions.append(PredictionResponse(
                genre=prediction_result['genre'],
                confidence=prediction_result['confidence'],
                all_predictions=prediction_result['all_predictions'],
                processing_time_ms=processing_time
            ))
            successful += 1
        
        except Exception as e:
            failed += 1
            continue
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_files=len(files),
        successful=successful,
        failed=failed
    )


@router.get("/prediction/status")
async def prediction_status():
    """Get prediction service status."""
    return {
        "status": "ready" if _model is not None else "not_ready",
        "model_loaded": _model is not None,
        "processor_loaded": _processor is not None,
        "class_names": _class_names if _class_names else []
    }
