"""
Training and retraining routes
"""
import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse

from api.models.schemas import (
    TrainingRequest, TrainingResponse, RetrainingRequest, RetrainingResponse,
    TrainingStatusResponse
)

router = APIRouter()

# Training job storage (in production, use Redis or database)
_training_jobs = {}


def _train_model_async(job_id: str, request: TrainingRequest):
    """Background task for training model."""
    try:
        _training_jobs[job_id]['status'] = 'running'
        _training_jobs[job_id]['message'] = 'Training started...'
        
        from src.model import GenreClassifier
        from src.preprocessing import DataAcquisition, DataProcessor
        
        # Load and process data
        data_acq = DataAcquisition(base_path="data/dataset")
        df = data_acq.load_csv_features()
        train_df, test_df = data_acq.split_data(
            df, 
            test_size=request.test_size,
            random_state=42
        )
        
        processor = DataProcessor(models_dir="models")
        X_train, y_train, metadata = processor.process_training_data(train_df)
        X_test, y_test = processor.process_test_data(test_df)
        
        # Train model
        classifier = GenreClassifier(model_type=request.model_type)
        
        if request.tune_hyperparameters:
            _training_jobs[job_id]['message'] = 'Tuning hyperparameters...'
            classifier.tune_hyperparameters(X_train, y_train)
        
        _training_jobs[job_id]['message'] = 'Training model...'
        train_metrics = classifier.train(X_train, y_train, use_cross_validation=True)
        
        _training_jobs[job_id]['message'] = 'Evaluating model...'
        test_metrics = classifier.evaluate(X_test, y_test, class_names=metadata['class_names'])
        
        # Save model
        model_version = classifier.save_model(version=request.version)
        
        _training_jobs[job_id]['status'] = 'completed'
        _training_jobs[job_id]['message'] = 'Training completed successfully'
        _training_jobs[job_id]['result'] = {
            'model_version': model_version,
            'train_accuracy': train_metrics['train_accuracy'],
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1_weighted']
        }
        _training_jobs[job_id]['progress'] = 100.0
    
    except Exception as e:
        _training_jobs[job_id]['status'] = 'failed'
        _training_jobs[job_id]['message'] = f'Training failed: {str(e)}'
        _training_jobs[job_id]['error'] = str(e)


def _retrain_model_async(job_id: str, request: RetrainingRequest):
    """Background task for retraining model."""
    try:
        _training_jobs[job_id]['status'] = 'running'
        _training_jobs[job_id]['message'] = 'Retraining started...'
        
        from src.retraining import ModelRetrainer
        
        retrainer = ModelRetrainer()
        
        _training_jobs[job_id]['message'] = 'Loading existing model...'
        retrained_classifier, metadata = retrainer.retrain_model(
            model_version=request.model_version,
            new_data_source=None,  # Will use uploads directory
            combine_strategy=request.combine_strategy,
            use_existing_data=request.use_existing_data,
            new_version=request.new_version
        )
        
        _training_jobs[job_id]['status'] = 'completed'
        _training_jobs[job_id]['message'] = 'Retraining completed successfully'
        _training_jobs[job_id]['result'] = {
            'old_version': request.model_version,
            'new_version': metadata.get('version'),
            'test_accuracy': metadata.get('test_accuracy')
        }
        _training_jobs[job_id]['progress'] = 100.0
    
    except Exception as e:
        _training_jobs[job_id]['status'] = 'failed'
        _training_jobs[job_id]['message'] = f'Retraining failed: {str(e)}'
        _training_jobs[job_id]['error'] = str(e)


@router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train a new model.
    
    Args:
        request: Training configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        TrainingResponse with job ID
    """
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    _training_jobs[job_id] = {
        'status': 'pending',
        'message': 'Training job queued',
        'progress': 0.0,
        'created_at': datetime.now().isoformat()
    }
    
    # Start training in background
    background_tasks.add_task(_train_model_async, job_id, request)
    
    return TrainingResponse(
        status="pending",
        message="Training job started",
        job_id=job_id
    )


@router.post("/retrain", response_model=RetrainingResponse)
async def retrain_model(request: RetrainingRequest, background_tasks: BackgroundTasks):
    """
    Retrain an existing model with new data.
    
    Args:
        request: Retraining configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        RetrainingResponse with job ID
    """
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    _training_jobs[job_id] = {
        'status': 'pending',
        'message': 'Retraining job queued',
        'progress': 0.0,
        'created_at': datetime.now().isoformat()
    }
    
    # Start retraining in background
    background_tasks.add_task(_retrain_model_async, job_id, request)
    
    return RetrainingResponse(
        status="pending",
        old_version=request.model_version,
        message="Retraining job started",
        job_id=job_id
    )


@router.get("/training/status/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(job_id: str):
    """
    Get training job status.
    
    Args:
        job_id: Job ID from training/retraining request
        
    Returns:
        TrainingStatusResponse with current status
    """
    if job_id not in _training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job ID not found: {job_id}"
        )
    
    job = _training_jobs[job_id]
    
    return TrainingStatusResponse(
        job_id=job_id,
        status=job['status'],
        progress=job.get('progress', 0.0),
        message=job.get('message'),
        result=job.get('result')
    )


@router.post("/data/upload")
async def upload_training_data(files: list[UploadFile] = File(...)):
    """
    Upload training data (audio files).
    
    Args:
        files: List of audio files to upload
        
    Returns:
        Upload status and file information
    """
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_files = []
    failed_files = []
    
    for file in files:
        try:
            # Validate file
            allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in allowed_extensions:
                failed_files.append({
                    'filename': file.filename,
                    'reason': f'Invalid file format. Allowed: {allowed_extensions}'
                })
                continue
            
            # Save file
            file_path = uploads_dir / file.filename
            content = await file.read()
            
            with open(file_path, 'wb') as f:
                f.write(content)
            
            uploaded_files.append({
                'filename': file.filename,
                'size': len(content),
                'path': str(file_path)
            })
        
        except Exception as e:
            failed_files.append({
                'filename': file.filename,
                'reason': str(e)
            })
    
    return {
        'status': 'completed',
        'uploaded': len(uploaded_files),
        'failed': len(failed_files),
        'uploaded_files': uploaded_files,
        'failed_files': failed_files
    }


@router.get("/training/status")
async def training_status():
    """Get all training jobs status."""
    return {
        'total_jobs': len(_training_jobs),
        'jobs': {
            job_id: {
                'status': job['status'],
                'message': job.get('message'),
                'progress': job.get('progress', 0.0),
                'created_at': job.get('created_at')
            }
            for job_id, job in _training_jobs.items()
        }
    }
