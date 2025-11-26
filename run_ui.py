"""
Run Streamlit UI Application

Usage:
    python run_ui.py
    or
    streamlit run ui/streamlit_app.py
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    # Get the UI directory
    ui_dir = Path(__file__).parent / "ui"
    app_file = ui_dir / "streamlit_app.py"
    
    if not app_file.exists():
        print(f"Error: Streamlit app not found at {app_file}")
        sys.exit(1)
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app_file),
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

