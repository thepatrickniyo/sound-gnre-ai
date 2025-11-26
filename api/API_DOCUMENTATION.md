# API Documentation

## Music Genre Classification API

Base URL: `http://localhost:8000`

API Version: `v1`

---

## Endpoints Overview

### Health & Monitoring
- `GET /` - Root endpoint
- `GET /api/v1/health` - Health check
- `GET /api/v1/model/status` - Model status
- `GET /api/v1/metrics` - Model performance metrics
- `GET /api/v1/stats` - API usage statistics

### Prediction
- `POST /api/v1/predict` - Single prediction
- `POST /api/v1/predict/batch` - Batch prediction
- `GET /api/v1/prediction/status` - Prediction service status

### Training
- `POST /api/v1/train` - Train new model
- `POST /api/v1/retrain` - Retrain existing model
- `GET /api/v1/training/status/{job_id}` - Get training job status
- `GET /api/v1/training/status` - Get all training jobs
- `POST /api/v1/data/upload` - Upload training data

---

## Endpoint Details

### 1. Health Check

**GET** `/api/v1/health`

Returns API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-26T22:00:00",
  "version": "1.0.0"
}
```

---

### 2. Model Status

**GET** `/api/v1/model/status`

Returns current model information.

**Response:**
```json
{
  "model_version": "v1_20251126_211324",
  "model_type": "random_forest",
  "training_date": "2025-11-26T21:13:24",
  "test_accuracy": 0.665,
  "is_loaded": true
}
```

---

### 3. Single Prediction

**POST** `/api/v1/predict`

Predict genre from a single audio file.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Audio file (WAV, MP3, FLAC, M4A, OGG)

**Response:**
```json
{
  "genre": "blues",
  "confidence": 0.85,
  "all_predictions": {
    "blues": 0.85,
    "classical": 0.05,
    "country": 0.03,
    ...
  },
  "processing_time_ms": 1234.56
}
```

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav"
```

---

### 4. Batch Prediction

**POST** `/api/v1/predict/batch`

Predict genres from multiple audio files.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Multiple audio files

**Response:**
```json
{
  "predictions": [
    {
      "genre": "blues",
      "confidence": 0.85,
      "all_predictions": {...},
      "processing_time_ms": 1234.56
    },
    ...
  ],
  "total_files": 5,
  "successful": 4,
  "failed": 1
}
```

---

### 5. Train Model

**POST** `/api/v1/train`

Train a new model.

**Request Body:**
```json
{
  "model_type": "random_forest",
  "test_size": 0.2,
  "tune_hyperparameters": false,
  "version": "v2.0"
}
```

**Response:**
```json
{
  "status": "pending",
  "message": "Training job started",
  "job_id": "uuid-here"
}
```

**Check Status:**
```bash
GET /api/v1/training/status/{job_id}
```

---

### 6. Retrain Model

**POST** `/api/v1/retrain`

Retrain existing model with new data.

**Request Body:**
```json
{
  "model_version": null,
  "combine_strategy": "append",
  "use_existing_data": true,
  "new_version": "v2.0"
}
```

**Response:**
```json
{
  "status": "pending",
  "old_version": "v1_20251126_211324",
  "new_version": null,
  "message": "Retraining job started",
  "job_id": "uuid-here"
}
```

---

### 7. Upload Training Data

**POST** `/api/v1/data/upload`

Upload audio files for training.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Multiple audio files

**Response:**
```json
{
  "status": "completed",
  "uploaded": 5,
  "failed": 0,
  "uploaded_files": [
    {
      "filename": "audio1.wav",
      "size": 123456,
      "path": "data/uploads/audio1.wav"
    }
  ],
  "failed_files": []
}
```

---

### 8. Get Training Status

**GET** `/api/v1/training/status/{job_id}`

Get status of a training job.

**Response:**
```json
{
  "job_id": "uuid-here",
  "status": "completed",
  "progress": 100.0,
  "message": "Training completed successfully",
  "result": {
    "model_version": "v2_20251126_220000",
    "train_accuracy": 0.998,
    "test_accuracy": 0.665,
    "test_f1": 0.659
  }
}
```

**Status values:**
- `pending` - Job queued
- `running` - Job in progress
- `completed` - Job finished successfully
- `failed` - Job failed

---

### 9. Model Metrics

**GET** `/api/v1/metrics`

Get model performance metrics.

**Response:**
```json
{
  "accuracy": 0.665,
  "precision": 0.664,
  "recall": 0.665,
  "f1_score": 0.659,
  "per_class_metrics": {
    "blues": {
      "precision": 0.714,
      "recall": 0.750,
      "f1": 0.732
    },
    ...
  }
}
```

---

### 10. API Statistics

**GET** `/api/v1/stats`

Get API usage statistics.

**Response:**
```json
{
  "total_requests": 150,
  "predictions": 120,
  "training_jobs": 5,
  "errors": 2,
  "uptime_seconds": 3600,
  "uptime_hours": 1.0,
  "start_time": "2025-11-26T21:00:00",
  "requests_per_hour": 150.0
}
```

---

## Error Responses

All endpoints return errors in the following format:

```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "timestamp": "2025-11-26T22:00:00"
}
```

**HTTP Status Codes:**
- `200` - Success
- `400` - Bad Request (invalid input)
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error
- `503` - Service Unavailable (model not loaded)

---

## Interactive Documentation

When the API is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These provide interactive documentation where you can test endpoints directly.

---

## Example Usage

### Python (requests)
```python
import requests

# Single prediction
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/predict',
        files={'file': f}
    )
    result = response.json()
    print(f"Predicted genre: {result['genre']}")
    print(f"Confidence: {result['confidence']}")
```

### cURL
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Model status
curl http://localhost:8000/api/v1/model/status

# Prediction
curl -X POST "http://localhost:8000/api/v1/predict" \
  -F "file=@audio.wav"
```

---

## Notes

- All file uploads support: WAV, MP3, FLAC, M4A, OGG
- Training/retraining jobs run asynchronously
- Use job IDs to track training progress
- Model is loaded on API startup
- API logs all requests to `api.log`

