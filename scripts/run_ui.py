"""
Run Streamlit UI Application

Usage:
    python run_ui.py
    or
    streamlit run ui/streamlit_app.py
"""
import subprocess
import sys
import os
from pathlib import Path

if __name__ == "__main__":
    # Get the project root (go up one level from scripts folder)
    project_root = Path(__file__).parent.parent
    ui_dir = project_root / "ui"
    app_file = ui_dir / "streamlit_app.py"
    
    if not app_file.exists():
        print(f"Error: Streamlit app not found at {app_file}")
        sys.exit(1)
    
    # Set PYTHONPATH to include project root so imports work
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{pythonpath}"
    else:
        env["PYTHONPATH"] = str(project_root)
    
    # Change to project root directory so relative paths work
    os.chdir(project_root)
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app_file),
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ], env=env)

