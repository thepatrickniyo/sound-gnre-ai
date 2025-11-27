# Sound Genre AI

<div align="center">
  <h1>Sound Genre AI</h1>
  <p>End-to-end pipeline for classifying music genres, uploading new audio, and retraining models.</p>
  
  [![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/thepatrickniyo/sound-gnre-ai)
  [![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
</div>

---

## Video Demo

Watch the project demonstration on YouTube:

[![Video Demo](https://img.shields.io/badge/YouTube-Watch%20Demo-red)](https://youtu.be/eQguT35zi9Q)

**Direct Link:** https://youtu.be/eQguT35zi9Q

---

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)
- [Notebook](#notebook)
- [Model Files](#model-files)
- [API Documentation](#api-documentation)
- [Usage](#usage)
- [Flood Request Simulation](#flood-request-simulation)
- [Contributing](#contributing)
- [License](#license)

---

## Project Description

**Sound Genre AI** is a comprehensive machine learning system designed to classify music genres from audio samples. The project provides a complete end-to-end pipeline including:

- **Data Processing**: Automated feature extraction from audio files using librosa
- **Model Training**: Multiple ML algorithms (Random Forest, SVM, XGBoost) with comprehensive evaluation
- **RESTful API**: FastAPI backend for predictions, data upload, and model retraining
- **Web Interface**: Streamlit UI for interactive predictions, data visualization, and model management
- **Model Retraining**: Incremental learning system for continuous model improvement

The system uses the GTZAN dataset (10 genres × 100 clips) as a default, but supports custom audio uploads in multiple formats (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`).

### Supported Genres

- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

---

## Features

- **High Accuracy**: Trained models achieve competitive performance on music genre classification
- **Retraining Pipeline**: Upload new labeled data and retrain models with version control
- **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, F1-score, and ROC-AUC
- **RESTful API**: FastAPI backend with automatic OpenAPI documentation
- **Interactive UI**: Streamlit-based web interface for predictions and visualizations
- **Real-time Monitoring**: API statistics, model metrics, and uptime tracking
- **Extensible**: Modular architecture for easy customization and extension

---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/thepatrickniyo/sound-gnre-ai.git
cd sound-gnre-ai
```

### Step 2: Create Virtual Environment

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Note:** On macOS, you may need to install system dependencies for librosa:
```bash
brew install ffmpeg
```

### Step 4: Verify Installation

```bash
python -c "import pandas; import numpy; import librosa; import sklearn; print('All packages installed successfully!')"
```

### Step 5: Run Data Preprocessing

```bash
python src/preprocessing.py
```

This will:
- Load the dataset from `data/dataset/`
- Split data into train/test sets (80/20 split)
- Save processed datasets to `data/train/` and `data/test/`

### Step 6: Start the FastAPI Backend

```bash
# Option 1: Using uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Option 2: Using the helper script
python scripts/run_api.py
```

The API will be available at `http://localhost:8000`

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/health

### Step 7: Start the Streamlit UI (Optional)

In a new terminal:

```bash
python scripts/run_ui.py
```

The UI will be available at `http://localhost:8501`

---

## Project Structure

```
sound-gnre-ai/
│
├── api/                    # FastAPI application
│   ├── main.py            # FastAPI app initialization
│   ├── routes/            # API endpoints
│   │   ├── prediction.py  # Prediction endpoints
│   │   ├── training.py    # Training/retraining endpoints
│   │   └── monitoring.py  # Health check and metrics
│   └── models/            # Pydantic schemas
│
├── src/                   # Core ML modules
│   ├── preprocessing.py   # Data acquisition and processing
│   ├── feature_extraction.py  # Audio feature extraction
│   ├── model.py          # Model training and evaluation
│   └── retraining.py     # Model retraining pipeline
│
├── ui/                    # Streamlit interface
│   ├── streamlit_app.py  # Main app
│   └── pages/            # UI pages
│       ├── prediction.py  # Prediction interface
│       ├── visualization.py  # Data visualizations
│       ├── upload_data.py    # Data upload
│       └── retraining.py     # Retraining interface
│
├── notebook/              # Jupyter notebook
│   └── sound_genre_classification.ipynb
│
├── data/                  # Data directory
│   ├── dataset/          # Original GTZAN dataset
│   ├── train/            # Training data
│   ├── test/             # Test data
│   └── uploads/          # User-uploaded audio files
│
├── models/                # Saved models
│   ├── *.pkl             # Trained model files
│   ├── *.json            # Model metadata
│   └── training_history/ # Training history logs
│
├── scripts/               # Helper scripts
│   ├── run_api.py        # API startup script
│   └── run_ui.py         # UI startup script
│
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── IMPLEMENTATION.md     # Detailed implementation guide
```

---

## Notebook

The Jupyter notebook (`notebook/sound_genre_classification.ipynb`) provides a comprehensive walkthrough of the entire machine learning pipeline.

### Notebook Contents

#### 1. **Data Loading and Preprocessing**
- Loading CSV features from `features_30_sec.csv`
- Data validation and cleaning
- Train/test split (80/20) with stratification
- Genre distribution visualization
- Feature distribution analysis

#### 2. **Feature Extraction**
- **Chroma Features**: `chroma_stft_mean`, `chroma_stft_var`
- **RMS Energy**: `rms_mean`, `rms_var`
- **Spectral Features**: 
  - Spectral centroid (mean, var)
  - Spectral bandwidth (mean, var)
  - Spectral rolloff (mean, var)
- **Zero Crossing Rate**: Mean and variance
- **Harmony and Perceptr**: Mean and variance
- **Tempo**: Beat tracking
- **MFCC Features**: 20 coefficients (mean and variance for each)

#### 3. **Model Training**
- **Random Forest Classifier**: Ensemble method with hyperparameter tuning
- **Support Vector Machine (SVM)**: With RBF kernel
- **XGBoost**: Gradient boosting classifier
- Cross-validation for model selection
- Model comparison and selection

#### 4. **Model Evaluation**
- **Classification Metrics**:
  - Accuracy
  - Precision (per-class and macro/micro average)
  - Recall (per-class and macro/micro average)
  - F1-Score (per-class and macro/micro average)
  - Confusion Matrix
  - Classification Report
- **Additional Metrics**:
  - ROC-AUC (One-vs-Rest)
  - Log Loss
- **Visualizations**:
  - Confusion matrix heatmap
  - ROC curves
  - Precision-Recall curves
  - Feature importance plots
  - Per-genre performance breakdown

#### 5. **Model Test / Prediction Functions**
- Single audio file prediction
- Batch prediction
- Confidence scores for all genres
- Feature extraction from raw audio
- Model persistence and loading

### Running the Notebook

```bash
# Start Jupyter Notebook
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Navigate to `notebook/sound_genre_classification.ipynb` and run all cells.

---

## Model Files

The trained models are saved in the `models/` directory in Pickle (`.pkl`) format.

### Model Files Location

```
models/
├── v1_20251126_211324/
│   ├── genre_classifier_random_forest.pkl
│   └── genre_classifier_random_forest_metadata.json
├── v2_20251127_143048/
│   ├── genre_classifier_random_forest.pkl
│   └── genre_classifier_random_forest_metadata.json
├── v2_20251127_144621/
│   ├── genre_classifier_random_forest.pkl
│   └── genre_classifier_random_forest_metadata.json
├── v2_20251127_223054/
│   ├── genre_classifier_random_forest.pkl
│   └── genre_classifier_random_forest_metadata.json
├── data_processor_scaler.pkl
└── data_processor_label_encoder.pkl
```

### Model Metadata

Each model version includes a metadata JSON file with:
- Model version and timestamp
- Training date
- Model type (Random Forest, SVM, etc.)
- Test accuracy, precision, recall, F1-score
- Hyperparameters
- Dataset information

### Loading a Model

```python
import joblib

# Load the model
model = joblib.load('models/v2_20251127_223054/genre_classifier_random_forest.pkl')

# Load the scaler
scaler = joblib.load('models/data_processor_scaler.pkl')

# Load the label encoder
label_encoder = joblib.load('models/data_processor_label_encoder.pkl')
```

---

## API Documentation

### Base URL

```
http://localhost:8000/api/v1
```

### Key Endpoints

#### 1. Health Check
```http
GET /api/v1/health
```

#### 2. Model Status
```http
GET /api/v1/model/status
```

#### 3. Single Prediction
```http
POST /api/v1/predict
Content-Type: multipart/form-data

file: <audio_file>
```

**Response:**
```json
{
  "genre": "rock",
  "confidence": 0.85,
  "all_predictions": {
    "rock": 0.85,
    "pop": 0.10,
    "metal": 0.05
  }
}
```

#### 4. Batch Prediction
```http
POST /api/v1/predict/batch
Content-Type: multipart/form-data

files: <audio_file1>, <audio_file2>, ...
```

#### 5. Upload Training Data
```http
POST /api/v1/data/upload
Content-Type: multipart/form-data

files: <audio_file1>, <audio_file2>, ...
genre: "rock"
```

#### 6. Start Retraining
```http
POST /api/v1/retrain
Content-Type: application/json

{
  "strategy": "append",  # or "replace", "merge"
  "model_type": "random_forest"
}
```

#### 7. Training Status
```http
GET /api/v1/training/status/{job_id}
```

#### 8. Model Metrics
```http
GET /api/v1/metrics
```

#### 9. API Statistics
```http
GET /api/v1/stats
```

For complete API documentation, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Usage

### Using the API

#### Python Example

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

#### cURL Example

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Prediction
curl -X POST "http://localhost:8000/api/v1/predict" \
  -F "file=@audio.wav"
```

### Using the Streamlit UI

1. Start the UI: `python scripts/run_ui.py`
2. Navigate to `http://localhost:8501`
3. Use the sidebar to navigate between pages:
   - **Home**: Dashboard with model status and statistics
   - **Prediction**: Upload audio and get predictions
   - **Visualization**: View data visualizations
   - **Upload Data**: Upload new training data
   - **Retraining**: Trigger model retraining

### Training a New Model

```python
from src.model import GenreClassifier
from src.preprocessing import DataProcessor

# Load and preprocess data
processor = DataProcessor()
X_train, y_train, X_test, y_test = processor.load_and_preprocess()

# Train model
classifier = GenreClassifier(model_type='random_forest')
classifier.train(X_train, y_train)

# Evaluate
metrics = classifier.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.2%}")

# Save model
classifier.save('models/my_model.pkl')
```

---

## Flood Request Simulation

Load testing is performed using **Locust** to evaluate API performance under various load conditions.

### Load Testing Setup

The system supports load testing with different container configurations:
- 1 container
- 2 containers
- 4 containers
- 8 containers

### Metrics Measured

- **Response Time**: p50, p95, p99 percentiles
- **Latency**: Average and maximum latency
- **Throughput**: Requests per second (RPS)
- **Error Rate**: Percentage of failed requests

### Running Load Tests

```bash
# Install Locust
pip install locust

# Run Locust (if locustfile.py exists)
locust -f load_testing/locustfile.py --host=http://localhost:8000
```

### Expected Results

Load testing results demonstrate:
- **Scalability**: Performance improvements with multiple containers
- **Response Times**: Sub-second response times for predictions
- **Throughput**: High request handling capacity
- **Reliability**: Low error rates under load

*Note: Specific load testing results should be documented in the project's load testing reports.*

---

## Technologies Used

### Data Processing
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `librosa` - Audio processing and feature extraction
- `scipy` - Scientific computing

### Machine Learning
- `scikit-learn` - ML algorithms and utilities
- `xgboost` - Gradient boosting
- `joblib` - Model serialization

### API Development
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation

### UI Development
- `streamlit` - Web UI framework
- `plotly` - Interactive visualizations
- `matplotlib` / `seaborn` - Static visualizations

### Testing
- `pytest` - Unit testing
- `locust` - Load testing

---

## Results

### Model Performance

The trained models achieve competitive performance on the GTZAN dataset:

- **Accuracy**: ~66-70% (varies by model type)
- **Precision**: ~66-70% (weighted average)
- **Recall**: ~66-70% (weighted average)
- **F1-Score**: ~66-70% (weighted average)

### Per-Genre Performance

Performance varies by genre, with some genres (e.g., Classical, Jazz) showing higher accuracy than others.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Links

- **Repository**: https://github.com/thepatrickniyo/sound-gnre-ai
- **Video Demo**: https://youtu.be/eQguT35zi9Q
- **Documentation**: See `IMPLEMENTATION.md` for detailed implementation guide

---

## Author

**Patrick Niyo**

- GitHub: [@thepatrickniyo](https://github.com/thepatrickniyo)

---

## Acknowledgments

- GTZAN dataset for music genre classification
- Open source libraries and frameworks used in this project
- Community contributors and feedback

---

<div align="center">
  <p>Made with passion for music and machine learning</p>
  <p>Star this repo if you find it helpful!</p>
</div>
