#!/usr/bin/env python3
"""
Script to run the FastAPI application
"""

import sys
import os
from pathlib import Path

# Get project root (go up one level from scripts folder)
project_root = Path(__file__).parent.parent

# Add project root to Python path so imports work
sys.path.insert(0, str(project_root))

# Set PYTHONPATH environment variable for uvicorn
env = os.environ.copy()
pythonpath = env.get("PYTHONPATH", "")
if pythonpath:
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{pythonpath}"
else:
    env["PYTHONPATH"] = str(project_root)

# Change to project root directory so relative paths work
os.chdir(project_root)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

