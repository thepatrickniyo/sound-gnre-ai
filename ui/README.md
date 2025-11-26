# Streamlit UI for Music Genre Classification

This directory contains the Streamlit-based user interface for the Music Genre Classification system.

## Structure

```
ui/
├── streamlit_app.py          # Main application (Dashboard/Home page)
├── pages/
│   ├── prediction.py         # Prediction page
│   ├── visualization.py      # Data visualization page
│   ├── upload_data.py        # Data upload page
│   └── retraining.py         # Model retraining page
└── utils/
    ├── api_client.py         # API client for FastAPI communication
    └── audio_utils.py         # Audio file utilities
```

## Features

### 🏠 Dashboard (Home Page)
- API health status
- Model status and version information
- Quick statistics (requests, predictions, errors)
- Model performance metrics
- Per-genre performance breakdown

### 🎤 Prediction Page
- Upload audio files (WAV, MP3, FLAC, M4A, OGG)
- Record audio (basic support)
- Real-time genre prediction
- Confidence scores for all genres
- Interactive visualizations (bar charts, pie charts)

### 📊 Visualization Page
- **Feature 1: Tempo Distribution**
  - Box plots and violin plots showing tempo ranges across genres
  - Statistical analysis of tempo by genre
  
- **Feature 2: MFCC Coefficients**
  - Heatmaps showing average MFCC values by genre
  - Line plots comparing genres
  - Timbral characteristics visualization
  
- **Feature 3: Spectral Centroid**
  - Histograms and density plots
  - Brightness/darkness analysis by genre
  
- **Dataset Overview**
  - Genre distribution charts
  - Feature correlation matrix
  - Dataset statistics

### 📁 Upload Data Page
- Bulk file upload (multiple files at once)
- File validation
- Upload progress tracking
- File information display
- Integration with API upload endpoint

### 🔄 Retraining Page
- Current model information display
- Retraining configuration
- Real-time training status monitoring
- Model comparison (old vs new)
- Training job history

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- `streamlit>=1.28.0`
- `plotly>=5.17.0`
- `requests>=2.31.0`

## Running the UI

### Method 1: Using the run script
```bash
python run_ui.py
```

### Method 2: Direct Streamlit command
```bash
streamlit run ui/streamlit_app.py
```

### Method 3: With custom port
```bash
streamlit run ui/streamlit_app.py --server.port 8501
```

The UI will be available at: `http://localhost:8501`

## Configuration

### API Base URL

The API base URL can be configured in the sidebar of each page. Default: `http://localhost:8000`

To change the default, modify the `api_base_url` parameter in each page file.

## Prerequisites

1. **FastAPI Server**: The API server must be running before using the UI.
   ```bash
   python run_api.py
   # or
   uvicorn api.main:app --reload
   ```

2. **Trained Model**: At least one trained model should be available in the `models/` directory.

3. **Dataset**: For visualizations, the dataset CSV files should be in `data/dataset/`.

## Usage

1. **Start the API server** (if not already running):
   ```bash
   python run_api.py
   ```

2. **Start the Streamlit UI**:
   ```bash
   python run_ui.py
   ```

3. **Open your browser** and navigate to `http://localhost:8501`

4. **Use the sidebar** to navigate between pages

## Page Navigation

Streamlit automatically detects pages in the `pages/` directory and adds them to the sidebar navigation. The pages are:

- **🏠 Dashboard** - Main page (`streamlit_app.py`)
- **🎤 Predict** - Prediction page (`pages/prediction.py`)
- **📊 Visualizations** - Visualization page (`pages/visualization.py`)
- **📁 Upload Data** - Upload page (`pages/upload_data.py`)
- **🔄 Retraining** - Retraining page (`pages/retraining.py`)

## API Integration

The UI communicates with the FastAPI backend through the `APIClient` class in `utils/api_client.py`. All API calls are handled automatically with proper error handling and user feedback.

### Endpoints Used

- `GET /api/v1/health` - Health check
- `GET /api/v1/model/status` - Model status
- `GET /api/v1/metrics` - Model metrics
- `GET /api/v1/stats` - API statistics
- `POST /api/v1/predict` - Single prediction
- `POST /api/v1/predict/batch` - Batch prediction
- `POST /api/v1/data/upload` - Upload training data
- `POST /api/v1/retrain` - Retrain model
- `GET /api/v1/training/status/{job_id}` - Training status

## Troubleshooting

### UI cannot connect to API
- Ensure the API server is running
- Check the API base URL in the sidebar
- Verify the API is accessible at the specified URL

### Model not loaded
- Train a model first using the training script or API
- Check that model files exist in the `models/` directory

### Visualizations not showing
- Ensure dataset CSV files are in `data/dataset/`
- Check that the CSV files contain the expected columns

### File upload fails
- Check file format (WAV, MP3, FLAC, M4A, OGG)
- Verify file size is under 50MB
- Ensure API server is running and accessible

## Development

### Adding New Pages

1. Create a new file in `ui/pages/`
2. Follow the structure of existing pages
3. Use `st.set_page_config()` for page configuration
4. Streamlit will automatically add it to the navigation

### Customizing Styles

Modify the CSS in each page file or create a shared CSS file. The main app includes custom CSS in `streamlit_app.py`.

## Notes

- The UI uses Streamlit's caching (`@st.cache_data`, `@st.cache_resource`) for performance
- API client is cached as a resource to maintain connection
- All pages include error handling and user feedback
- The UI is responsive and works on different screen sizes

