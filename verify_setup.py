#!/usr/bin/env python3
"""
Setup Verification Script

This script verifies that all dependencies are installed correctly
and that the project structure is set up properly.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python version {version.major}.{version.minor} is too old. Need 3.8+")
        return False
    print(f"✓ Python version {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'librosa': 'librosa',
        'sklearn': 'scikit-learn',
        'dotenv': 'python-dotenv'
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"✓ {package_name} installed")
        except ImportError:
            print(f"❌ {package_name} NOT installed")
            missing.append(package_name)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_project_structure():
    """Check if project structure is correct."""
    required_paths = {
        'src/preprocessing.py': 'Source preprocessing module',
        'data/dataset/features_30_sec.csv': '30-second features CSV',
        'data/dataset/genres_original': 'Audio files directory',
        'requirements.txt': 'Dependencies file'
    }
    
    all_good = True
    for path_str, description in required_paths.items():
        path = Path(path_str)
        if path.exists():
            print(f"✓ {description}: {path_str}")
        else:
            print(f"❌ {description} NOT found: {path_str}")
            all_good = False
    
    return all_good

def check_directories():
    """Check if required directories exist or can be created."""
    required_dirs = ['data/train', 'data/test']
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        if path.exists():
            print(f"✓ Directory ready: {dir_path}")
        else:
            print(f"❌ Cannot create directory: {dir_path}")
            return False
    
    return True

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("SETUP VERIFICATION")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Directories", check_directories)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n[{name}]")
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to run the Data Acquisition Module.")
        print("\nNext step: python src/preprocessing.py")
        return 0
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

