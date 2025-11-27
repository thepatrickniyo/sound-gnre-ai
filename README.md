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

Load testing is performed using **Locust** to evaluate API performance under various load conditions. The system includes a complete load testing infrastructure that you can run to test your API's performance.

### Prerequisites

Ensure Locust is installed:
```bash
pip install locust
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

### Running Load Tests

#### Quick Start

1. **Start the API** (in one terminal):
```bash
python scripts/run_api.py
# API should be running at http://localhost:8000
```

2. **Run Load Test** (in another terminal):
```bash
# Basic test with default settings (10 users, 2m duration)
python load_testing/run_load_test.py

# Or use the convenience script
python scripts/run_load_test.py
```

#### Custom Test Configuration

```bash
# Test with 50 concurrent users, spawn 5 per second, run for 5 minutes
python load_testing/run_load_test.py --users 50 --spawn-rate 5 --run-time 5m

# Test with 100 users for 10 minutes
python load_testing/run_load_test.py --users 100 --spawn-rate 10 --run-time 10m

# Test with interactive web UI
python load_testing/run_load_test.py --ui

# Test against a different host
python load_testing/run_load_test.py --host http://localhost:8000
```

#### Using Locust Web UI

For interactive testing with real-time charts:
```bash
locust -f load_testing/locustfile.py --host=http://localhost:8000
```

Then open your browser to `http://localhost:8089` to configure and monitor the test.

### Test Scenarios

The load test simulates realistic user behavior with weighted tasks:

- **Health Check** (5x weight) - Most common request
- **Predict Genre** (10x weight) - Main functionality, uploads audio files
- **Model Status** (3x weight) - Status checks
- **Get Metrics** (2x weight) - Performance metrics
- **Get Stats** (1x weight) - API statistics

### Metrics Measured

The load tests collect comprehensive performance metrics:

- **Response Time**: p50, p95, p99 percentiles
- **Average Response Time**: Mean response time per endpoint
- **Min/Max Response Time**: Response time boundaries
- **Throughput**: Requests per second (RPS)
- **Total Requests**: Number of requests sent
- **Failure Rate**: Percentage of failed requests
- **Response Times by Endpoint**: Per-endpoint performance breakdown

### Results Location

Test results are automatically saved to the `load_testing/results/` directory in the project root. This directory contains all load testing artifacts:

```
load_testing/results/
├── Locust Results.pdf                              # Comprehensive PDF report
├── result-page1.png                                # Results visualization (Page 1)
├── resulte-page2.png                               # Results visualization (Page 2)
├── result-page3.png                                # Results visualization (Page 3)
├── load_test_10users_20251127_120000.html          # HTML report
├── load_test_10users_20251127_120000_stats.csv     # Request statistics (CSV)
├── load_test_10users_20251127_120000_stats_history.csv # Historical stats (CSV)
├── load_test_10users_20251127_120000_failures.csv  # Failed requests (CSV)
├── load_test_10users_20251127_120000_exceptions.csv # Exceptions (CSV)
└── load_test_10users_20251127_120000_summary.json  # Test summary (JSON)
```

**Location in Repository:**
- Results directory: `load_testing/results/`
- PDF Report: `load_testing/results/Locust Results.pdf`
- CSV Data: `load_testing/results/*.csv` files
- Visualizations: `load_testing/results/result-page*.png` files

### Viewing Results

Multiple formats are available for analyzing test results:

1. **PDF Report** (`Locust Results.pdf`): Comprehensive document with detailed analysis, charts, and performance metrics
2. **HTML Report** (`.html` files): Interactive web-based report - open in any browser for detailed visual analysis
3. **CSV Files**: 
   - `*_stats.csv` - Request statistics with percentiles
   - `*_stats_history.csv` - Time-series data for trend analysis
   - `*_failures.csv` - Details of any failed requests
   - `*_exceptions.csv` - Exception logs
   - Import into Excel, Google Sheets, or Python pandas for custom analysis
4. **Summary JSON**: Machine-readable test summary for programmatic access
5. **PNG Images**: Visual charts and graphs showing performance metrics

### Load Test Results Visualization

The following images show detailed performance metrics from actual load test runs:

**Results Overview - Page 1:**
![Load Test Results Page 1](https://github.com/thepatrickniyo/sound-gnre-ai/blob/master/load_testing/results/result-page1.png?raw=true)

**Results Overview - Page 2:**
![Load Test Results Page 2](https://github.com/thepatrickniyo/sound-gnre-ai/blob/master/load_testing/results/resulte-page2.png?raw=true)

**Results Overview - Page 3:**
![Load Test Results Page 3](https://github.com/thepatrickniyo/sound-gnre-ai/blob/master/load_testing/results/result-page3.png?raw=true)

These visualizations include:
- Response time distributions
- Requests per second over time
- Endpoint performance comparisons
- Failure rates and error analysis
- Percentile breakdowns (p50, p95, p99)

### Example Results

After running a load test, you'll see output like:

```
Type     Name            # reqs      # fails  |     Avg     Min     Max  Median  |   req/s  failures/s
--------|--------------|-----------|---------|-----------|-------|-------|-------|--------|-----------
GET      Health Check       1250         0(0.00%) |      45      12     156      42  |   10.42    0.00
POST     Predict Genre      2500         0(0.00%) |    1250     890    3450    1200  |   20.83    0.00
GET      Model Status        750         0(0.00%) |      38      15     120      35  |    6.25    0.00
GET      Get Metrics         500         0(0.00%) |      42      18     145      40  |    4.17    0.00
GET      Get Stats           250         0(0.00%) |      35      12     110      32  |    2.08    0.00
--------|--------------|-----------|---------|-----------|-------|-------|-------|--------|-----------
         Aggregated         5250         0(0.00%) |     285     890    3450     280  |   43.75    0.00
```

### Actual Test Results

Here are results from a recent load test run (10 concurrent users, 30 seconds duration):

**Test Configuration:**
- Users: 10 concurrent users
- Spawn Rate: 2 users/second
- Duration: 30 seconds
- Total Requests: 59

**Performance Summary:**

| Endpoint | Requests | Failures | Avg Response (ms) | Min (ms) | Max (ms) | Median (ms) | Requests/s |
|----------|----------|----------|-------------------|----------|----------|-------------|------------|
| Health Check | 16 | 0 (0.00%) | 1,175 | 3 | 4,129 | 640 | 0.64 |
| Predict Genre | 21 | 0 (0.00%) | 3,689 | 1,027 | 5,132 | 3,700 | 0.84 |
| Model Status | 11 | 0 (0.00%) | 1,277 | 11 | 4,130 | 960 | 0.44 |
| Get Metrics | 7 | 0 (0.00%) | 1,190 | 149 | 2,909 | 750 | 0.28 |
| Get Stats | 4 | 0 (0.00%) | 1,237 | 63 | 3,135 | 610 | 0.16 |
| **Aggregated** | **59** | **0 (0.00%)** | **2,095** | **3** | **5,132** | **1,900** | **2.37** |

**Response Time Percentiles (Aggregated):**
- p50 (Median): 1,900 ms
- p95: 5,000 ms
- p99: 5,100 ms
- p99.9: 5,100 ms

**Key Findings:**
- **Zero Failures**: All 59 requests completed successfully (0% failure rate)
- **Throughput**: 2.37 requests per second average
- **Predict Genre Performance**: Average 3.7 seconds per prediction (includes audio processing)
- **Health/Monitoring Endpoints**: Sub-second response times for most requests
- **Reliability**: 100% success rate under load

**View Detailed Results:**
- **PDF Report**: `load_testing/results/Locust Results.pdf` - Comprehensive analysis document
- **HTML Report**: `load_testing/results/load_test_10users_20251127_231541.html` - Interactive web report
- **CSV Statistics**: `load_testing/results/load_test_10users_20251127_231541_stats.csv` - Raw data for analysis
- **Visual Charts**: `load_testing/results/result-page*.png` - Performance visualization images

### Performance Benchmarks

Based on actual load test results, typical performance characteristics:

- **Health Check**: ~1,175ms average (ranges from 3ms to 4,129ms)
- **Predict Genre**: ~3,689ms average (1-5 seconds, depends on audio file size and feature extraction)
- **Model Status**: ~1,277ms average (ranges from 11ms to 4,130ms)
- **Metrics/Stats**: ~1,190-1,237ms average (ranges from 63ms to 3,135ms)

**Note**: Response times can vary based on system load, audio file complexity, and concurrent request volume. The Predict Genre endpoint includes audio feature extraction which is computationally intensive.

### Tips for Load Testing

1. **Start Small**: Begin with 10-20 users to establish baseline
2. **Gradual Ramp-up**: Increase users gradually to find breaking points
3. **Monitor Resources**: Watch CPU, memory, and disk usage during tests
4. **Test Different Scenarios**: Vary user counts, spawn rates, and durations
5. **Compare Results**: Run tests before/after optimizations to measure improvements

### Troubleshooting

**Issue**: "No sample audio file found"
- **Solution**: Ensure `data/dataset/genres_original/` contains audio files, or the test will skip prediction requests

**Issue**: "Connection refused"
- **Solution**: Make sure the API is running on the specified host/port

**Issue**: "Locust not found"
- **Solution**: Install Locust with `pip install locust`

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

### Testing & Load Testing
- `pytest` - Unit testing
- `locust` - Load testing framework

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
