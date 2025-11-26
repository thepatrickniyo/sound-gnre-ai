# Quick Start Guide

## Copy-Paste Commands (macOS/Linux)

```bash
# Navigate to project directory
cd /sound-gnre-ai

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas; import numpy; import librosa; import sklearn; print('✓ All packages installed successfully!')"

# Run Data Acquisition Module
python src/preprocessing.py

# Verify output files were created
ls -lh data/train/train.csv data/test/test.csv
```

## Copy-Paste Commands (Windows)

```bash
# Navigate to project directory
cd C:\path\to\sound-gnre-ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas; import numpy; import librosa; import sklearn; print('✓ All packages installed successfully!')"

# Run Data Acquisition Module
python src/preprocessing.py

# Verify output files were created
dir data\train\train.csv data\test\test.csv
```

## What to Expect

After running `python src/preprocessing.py`, you should see:

1. **Console Output:**
   - Log messages showing data loading progress
   - Validation results
   - Train/test split information
   - Summary statistics

2. **Created Files:**
   - `data/train/train.csv` - Training dataset (~800 samples)
   - `data/test/test.csv` - Test dataset (~200 samples)

3. **Directory Structure:**
   ```
   data/
   ├── train/
   │   └── train.csv
   └── test/
       └── test.csv
   ```

## Troubleshooting Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| Module not found | `pip install -r requirements.txt` |
| Wrong directory | `cd /Users/patrickniyo/Documents/workspaces/ALU/sound-gnre-ai` |
| Virtual env not active | `source venv/bin/activate` (see `(venv)` in prompt) |
| librosa install fails | `brew install ffmpeg` (macOS) then `pip install librosa` |

