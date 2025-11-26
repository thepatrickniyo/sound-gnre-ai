#!/usr/bin/env python3
"""
Example script demonstrating the Model Retraining Module

This script shows how to:
1. Retrain an existing model with new data
2. View training history
3. Rollback to previous versions
"""

from src.retraining import ModelRetrainer
import pandas as pd

def main():
    """
    Example usage of Model Retraining Module
    """
    print("=" * 60)
    print("MODEL RETRAINING EXAMPLE")
    print("=" * 60)
    
    # Initialize retrainer
    retrainer = ModelRetrainer(models_dir="models", data_dir="data")
    
    # List available model versions
    print("\n[1] Available Model Versions:")
    versions = retrainer.list_model_versions()
    if versions:
        for v in versions:
            acc_str = f"{v['test_accuracy']:.4f}" if v['test_accuracy'] else 'N/A'
            print(f"  - {v['version']}: {v['model_type']} (Accuracy: {acc_str})")
    else:
        print("  No models found. Train a model first using: python src/model.py")
        return
    
    # Get training history
    print("\n[2] Training History:")
    history_df = retrainer.get_training_history()
    if not history_df.empty:
        print(history_df[['retraining_date', 'old_version', 'new_version', 
                         'metrics_comparison']].to_string(index=False))
    else:
        print("  No retraining history found.")
    
    # Example: Retrain with new data
    print("\n[3] Retraining Example:")
    print("  To retrain a model, use:")
    print("  retrained_classifier, metadata = retrainer.retrain_model(")
    print("      model_version=None,  # Use latest")
    print("      new_data_source='data/uploads/new_data.csv',")
    print("      combine_strategy='append'  # or 'replace', 'merge'")
    print("  )")
    
    print("\n" + "=" * 60)
    print("RETRAINING MODULE READY")
    print("=" * 60)
    print("\nKey Features:")
    print("  ✓ Load existing models")
    print("  ✓ Retrain with new data")
    print("  ✓ Combine datasets (append/replace/merge)")
    print("  ✓ Model versioning")
    print("  ✓ Training history tracking")
    print("  ✓ Rollback capability")
    print("\nSee src/retraining.py for full API documentation.")


if __name__ == "__main__":
    main()

