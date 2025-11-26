# Setup and Running Guide

## Step-by-Step Instructions to Run Data Acquisition Module

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

---

## Step 1: Create a Virtual Environment (Recommended)

### For macOS/Linux:
```bash
# Navigate to project directory
cd sound-gnre-ai

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### For Windows:
```bash
# Navigate to project directory
cd C:\path\to\sound-gnre-ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**Note:** After activation, you should see `(venv)` in your terminal prompt.

---

## Step 2: Install Dependencies

```bash
# Make sure you're in the project root directory
# and virtual environment is activated

# Install all required packages
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected output:** All packages should install successfully. This may take a few minutes, especially for librosa which has some system dependencies.

---

## Step 3: Verify Installation

```bash
# Check if packages are installed correctly
python -c "import pandas; import numpy; import librosa; import sklearn; print('All packages installed successfully!')"
```

---

## Step 4: Run the Data Acquisition Module

### Option A: Run as a Python Script

```bash
# From the project root directory
python src/preprocessing.py
```

### Option B: Run as a Python Module

```bash
# From the project root directory
python -m src.preprocessing
```

### Option C: Use in Your Own Script

Create a new Python file (e.g., `run_data_acquisition.py`):

```python
from src.preprocessing import DataAcquisition

# Initialize DataAcquisition
data_acq = DataAcquisition(base_path="data/dataset")

# Process dataset
results = data_acq.process_dataset(
    test_size=0.2,
    random_state=42,
    validate_audio=True,
    audio_sample_size=5  # Sample 5 files per genre for validation
)

# Print summary
print("\n" + "=" * 60)
print("DATA ACQUISITION SUMMARY")
print("=" * 60)
print(f"Total samples: {len(results['raw_data'])}")
print(f"Train samples: {len(results['train_data'])}")
print(f"Test samples: {len(results['test_data'])}")
print(f"Number of features: {len(results['raw_data'].columns) - 1}")
print(f"Number of genres: {results['raw_data']['label'].nunique()}")
print(f"Genres: {sorted(results['raw_data']['label'].unique())}")
print("=" * 60)
```

Then run:
```bash
python run_data_acquisition.py
```

---

## Step 5: Verify Output

After running, check that the following files/directories were created:

```bash
# Check train and test directories
ls -la data/train/
ls -la data/test/

# Verify CSV files were created
ls -lh data/train/train.csv
ls -lh data/test/test.csv
```

You should see:
- `data/train/train.csv` - Training dataset
- `data/test/test.csv` - Test dataset

---

## Expected Output

When you run the module, you should see output similar to:

```
2024-XX-XX XX:XX:XX - __main__ - INFO - ============================================================
2024-XX-XX XX:XX:XX - __main__ - INFO - Starting Data Acquisition Pipeline
2024-XX-XX XX:XX:XX - __main__ - INFO - ============================================================
2024-XX-XX XX:XX:XX - __main__ - INFO - Loading CSV features from: data/dataset/features_30_sec.csv
2024-XX-XX XX:XX:XX - __main__ - INFO - Successfully loaded 1001 rows from CSV
...
2024-XX-XX XX:XX:XX - __main__ - INFO - Splitting data with test_size=0.2, random_state=42, stratify=True
2024-XX-XX XX:XX:XX - __main__ - INFO - Train set: 800 samples (80.0%)
2024-XX-XX XX:XX:XX - __main__ - INFO - Test set: 201 samples (20.0%)
...
2024-XX-XX XX:XX:XX - __main__ - INFO - ============================================================
2024-XX-XX XX:XX:XX - __main__ - INFO - Data Acquisition Pipeline Completed Successfully
2024-XX-XX XX:XX:XX - __main__ - INFO - ============================================================

============================================================
DATA ACQUISITION SUMMARY
============================================================
Total samples: 1001
Train samples: 800
Test samples: 201
Number of features: 58
Number of genres: 10
Genres: ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
============================================================
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'pandas'`
**Solution:** Make sure your virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: `FileNotFoundError: CSV file not found`
**Solution:** Make sure you're running from the project root directory:
```bash
cd /Users/patrickniyo/Documents/workspaces/ALU/sound-gnre-ai
pwd  # Should show the project root
```

### Issue: `librosa` installation fails
**Solution:** On macOS, you may need to install system dependencies:
```bash
# macOS
brew install ffmpeg

# Then reinstall librosa
pip install librosa
```

### Issue: Audio validation takes too long
**Solution:** Reduce the sample size or skip audio validation:
```python
results = data_acq.process_dataset(
    test_size=0.2,
    random_state=42,
    validate_audio=False  # Skip audio validation
)
```

### Issue: Permission errors when creating directories
**Solution:** Check directory permissions:
```bash
ls -la data/
# Make sure you have write permissions
```

---

## Quick Start (All Steps Combined)

```bash
# 1. Navigate to project
cd /Users/patrickniyo/Documents/workspaces/ALU/sound-gnre-ai

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the module
python src/preprocessing.py

# 5. Verify output
ls -la data/train/ data/test/
```

---

## Next Steps

After successfully running the Data Acquisition Module:
1. ✅ Data is loaded and validated
2. ✅ Train/test split is created
3. ✅ Data is saved to `data/train/` and `data/test/`

You can now proceed to:
- **Phase 1, Step 1.2**: Feature Extraction Module
- **Phase 1, Step 1.3**: Data Processing Pipeline
- **Phase 2**: Model Creation & Training

---

## Additional Notes

- The module uses `features_30_sec.csv` by default. To use `features_3_sec.csv`, modify the code or pass it as a parameter.
- Audio validation is optional and can be skipped for faster execution.
- The random_state parameter ensures reproducible train/test splits.
- All logs are printed to console. For production, you may want to configure file logging.

