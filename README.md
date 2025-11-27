<div align="center">
  <h1>Sound Genre AI</h1>
  <p>End-to-end pipeline for classifying music genres, uploading new audio, and retraining models.</p>
</div>

## Overview

This project provides:
- A **FastAPI backend** exposing prediction, data upload, and retraining routes.
- A **Streamlit UI** for uploading audio, monitoring jobs, and visualizing results.
- Reusable modules under `src/` for feature extraction, preprocessing, model training, and retraining.

The default dataset is GTZAN (10 genres × 100 clips) stored in `data/dataset/`, but you can upload your own `.wav/.mp3/.flac/.m4a/.ogg` files via the UI or API (`data/uploads/` is automatically cleaned after each retrain).

## Quick Start

```bash
# clone repo and enter project
git clone <your-fork-url> sound-gnre-ai
cd sound-gnre-ai

# create + activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Run the FastAPI backend

```bash
# with venv activated
uvicorn api.main:app --host 0.0.0.0 --port 8000
# or use helper script
python run_api.py
```

Endpoints are available under `http://localhost:8000/api/v1/...` (see `api/routes/` for details). Health check lives at `/api/v1/health`.

### Run the Streamlit UI

```bash
python run_ui.py
# Streamlit serves at http://localhost:8501
```

Set the sidebar “API Base URL” to match the backend host if you deploy them separately.

### Retraining Workflow

1. Upload labeled audio via the UI or `POST /api/v1/data/upload`.
2. In the UI, start a retraining job (choose append/replace/merge strategy).
3. Background job loads the latest model, extracts features for uploaded audio (if needed), merges datasets, retrains, saves a new version, and clears `data/uploads/`.
4. Monitor job status via the UI or `GET /api/v1/training/status/{job_id}`.

Training history and saved models live under `models/`, with metadata JSON files per version.

## Deployment (backend only)

On Render or any other host:

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Provision a persistent volume for `data/` and `models/` if you need uploads and retrained artifacts to survive restarts.

## Project Structure

```
api/            # FastAPI application (routes, schemas)
src/            # Core ML modules (feature extraction, preprocessing, retraining)
ui/             # Streamlit interface
data/           # Dataset, train/test splits, uploads
models/         # Saved model versions + metadata
run_api.py      # Convenience script for backend
run_ui.py       # Convenience script for UI
notebook/       # Jupyter notebook experiments
```

## License

MIT License. See `LICENSE`.

