#!/usr/bin/env python3
"""
Convenience script to run load tests from the scripts directory.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from load_testing.run_load_test import main

if __name__ == "__main__":
    main()

