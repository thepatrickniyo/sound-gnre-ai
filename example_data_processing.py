#!/usr/bin/env python3
"""
Example script demonstrating the Data Processing Pipeline

This script shows how to use the DataProcessor class to:
1. Clean data (handle missing values, remove outliers)
2. Encode labels
3. Scale features
4. Prepare data for machine learning
5. Save/load processors
"""

from src.preprocessing import DataAcquisition, DataProcessor
import pandas as pd

def main():
    """
    Example usage of Data Processing Pipeline
    """
    print("=" * 60)
    print("DATA PROCESSING PIPELINE EXAMPLE")
    print("=" * 60)
    
    # Step 1: Load data using DataAcquisition
    print("\n[Step 1] Loading data...")
    data_acq = DataAcquisition(base_path="data/dataset")
    
    # Load CSV features
    df = data_acq.load_csv_features()
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Split into train/test
    train_df, test_df = data_acq.split_data(df, test_size=0.2, random_state=42)
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Step 2: Initialize DataProcessor
    print("\n[Step 2] Initializing DataProcessor...")
    processor = DataProcessor(models_dir="models")
    
    # Step 3: Process training data
    print("\n[Step 3] Processing training data...")
    X_train, y_train, metadata = processor.process_training_data(
        train_df,
        clean_data=True,
        remove_outliers=True
    )
    
    print(f"\nTraining data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Number of features: {metadata['n_features']}")
    print(f"Number of classes: {metadata['n_classes']}")
    print(f"Class names: {metadata['class_names']}")
    
    # Step 4: Process test data
    print("\n[Step 4] Processing test data...")
    X_test, y_test = processor.process_test_data(
        test_df,
        clean_data=True
    )
    
    print(f"\nTest data shape: X={X_test.shape}, y={y_test.shape}")
    
    # Step 5: Save processor (encoders, scalers, metadata)
    print("\n[Step 5] Saving processor...")
    processor.save_processor(prefix="data_processor")
    print("✓ Processor saved successfully")
    
    # Step 6: Demonstrate loading
    print("\n[Step 6] Loading processor...")
    new_processor = DataProcessor(models_dir="models")
    new_processor.load_processor(prefix="data_processor")
    print("✓ Processor loaded successfully")
    
    # Verify loaded processor works
    X_test_loaded, y_test_loaded = new_processor.process_test_data(test_df)
    print(f"✓ Loaded processor works! Test data shape: X={X_test_loaded.shape}, y={y_test_loaded.shape}")
    
    print("\n" + "=" * 60)
    print("DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    print("\nSummary:")
    print(f"- Training samples: {len(X_train)}")
    print(f"- Test samples: {len(X_test)}")
    print(f"- Features: {metadata['n_features']}")
    print(f"- Classes: {metadata['n_classes']}")
    print(f"- Feature names: {metadata['feature_names'][:5]}... (showing first 5)")
    print("\n✓ Data is ready for machine learning!")


if __name__ == "__main__":
    main()

