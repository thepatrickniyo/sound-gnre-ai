"""
Model Architecture and Training Module for Music Genre Classification

This module handles:
- Multiple ML model architectures
- Model training and hyperparameter tuning
- Model persistence and versioning
- Model evaluation
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Try to import XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GenreClassifier:
    """
    Base class for genre classification models with common functionality.
    """
    
    def __init__(self, model_type: str = "random_forest", models_dir: str = "models"):
        """
        Initialize GenreClassifier.
        
        Args:
            model_type: Type of model ('random_forest', 'svm', 'mlp', 'xgboost')
            models_dir: Directory to save models
        """
        self.model_type = model_type
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.model_version = None
        self.training_date = None
        self.metadata = {}
        self.is_trained = False
        
        # Initialize model based on type
        self._initialize_model()
        
        logger.info(f"Initialized {model_type} classifier")
    
    def _initialize_model(self) -> None:
        """Initialize the model based on model_type."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
        elif self.model_type == "svm":
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                verbose=False
            )
        elif self.model_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                verbose=False
            )
        elif self.model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost")
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. "
                           f"Choose from: 'random_forest', 'svm', 'mlp', 'xgboost'")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              use_cross_validation: bool = True,
              cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_cross_validation: Whether to use cross-validation
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model...")
        logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate with cross-validation if requested
        cv_scores = None
        if use_cross_validation:
            logger.info(f"Performing {cv_folds}-fold cross-validation...")
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy')
            logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Predict on training set
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Store metadata
        self.training_date = datetime.now().isoformat()
        self.metadata = {
            'model_type': self.model_type,
            'training_date': self.training_date,
            'train_accuracy': float(train_accuracy),
            'cv_mean': float(cv_scores.mean()) if cv_scores is not None else None,
            'cv_std': float(cv_scores.std()) if cv_scores is not None else None,
            'n_samples': int(len(X_train)),
            'n_features': int(X_train.shape[1]),
            'n_classes': int(len(np.unique(y_train)))
        }
        
        self.is_trained = True
        
        logger.info(f"Model training completed. Train accuracy: {train_accuracy:.4f}")
        
        return {
            'train_accuracy': train_accuracy,
            'cv_scores': cv_scores.tolist() if cv_scores is not None else None,
            'cv_mean': cv_scores.mean() if cv_scores is not None else None,
            'cv_std': cv_scores.std() if cv_scores is not None else None
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            class_names: Names of classes for reporting
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Evaluating model on test set...")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_weighted': float(f1),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        # Update metadata
        self.metadata['test_accuracy'] = float(accuracy)
        self.metadata['test_precision'] = float(precision)
        self.metadata['test_recall'] = float(recall)
        self.metadata['test_f1'] = float(f1)
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test Precision: {precision:.4f}")
        logger.info(f"Test Recall: {recall:.4f}")
        logger.info(f"Test F1-Score: {f1:.4f}")
        
        return metrics
    
    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                            param_grid: Optional[Dict] = None,
                            cv_folds: int = 5,
                            scoring: str = 'accuracy',
                            n_jobs: int = -1) -> Dict[str, Any]:
        """
        Tune hyperparameters using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for grid search (None = use default)
            cv_folds: Number of CV folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary with best parameters and score
        """
        logger.info(f"Tuning hyperparameters for {self.model_type}...")
        
        # Default parameter grids
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        # Create GridSearchCV
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        # Perform grid search
        logger.info(f"Performing grid search with {len(param_grid)} parameter combinations...")
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.metadata['best_params'] = grid_search.best_params_
        self.metadata['best_cv_score'] = float(grid_search.best_score_)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'best_estimator': grid_search.best_estimator_
        }
    
    def _get_default_param_grid(self) -> Dict[str, List]:
        """Get default parameter grid based on model type."""
        if self.model_type == "random_forest":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        elif self.model_type == "svm":
            return {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            }
        elif self.model_type == "mlp":
            return {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        elif self.model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available")
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        else:
            return {}
    
    def save_model(self, version: Optional[str] = None, prefix: str = "genre_classifier") -> str:
        """
        Save trained model to disk.
        
        Args:
            version: Model version (None = auto-generate)
            prefix: Prefix for saved files
            
        Returns:
            Path to saved model
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save.")
        
        # Generate version if not provided
        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v1_{timestamp}"
        
        self.model_version = version
        
        # Create version directory
        version_dir = self.models_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = version_dir / f"{prefix}_{self.model_type}.pkl"
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        metadata_path = version_dir / f"{prefix}_{self.model_type}_metadata.json"
        self.metadata['version'] = version
        self.metadata['model_path'] = str(model_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")
        
        return str(model_path)
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to saved model
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Try to load metadata
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.model_version = self.metadata.get('version')
            self.training_date = self.metadata.get('training_date')
            logger.info(f"Metadata loaded from {metadata_path}")
        
        self.is_trained = True
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance (if available).
        
        Returns:
            Feature importance array or None
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, return absolute coefficients
            return np.abs(self.model.coef_[0])
        else:
            logger.warning(f"Feature importance not available for {self.model_type}")
            return None


def compare_models(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  model_types: Optional[List[str]] = None,
                  class_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare multiple model types and return results.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_types: List of model types to compare (None = all available)
        class_names: Names of classes
        
    Returns:
        DataFrame with comparison results
    """
    if model_types is None:
        model_types = ['random_forest', 'svm', 'mlp']
        if XGBOOST_AVAILABLE:
            model_types.append('xgboost')
    
    results = []
    
    for model_type in model_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training and evaluating {model_type}")
        logger.info(f"{'='*60}")
        
        try:
            # Initialize and train
            classifier = GenreClassifier(model_type=model_type)
            train_metrics = classifier.train(X_train, y_train, use_cross_validation=True)
            
            # Evaluate
            test_metrics = classifier.evaluate(X_test, y_test, class_names=class_names)
            
            # Store results
            results.append({
                'model_type': model_type,
                'train_accuracy': train_metrics['train_accuracy'],
                'cv_mean': train_metrics['cv_mean'],
                'cv_std': train_metrics['cv_std'],
                'test_accuracy': test_metrics['accuracy'],
                'test_precision': test_metrics['precision_weighted'],
                'test_recall': test_metrics['recall_weighted'],
                'test_f1': test_metrics['f1_weighted']
            })
            
            logger.info(f"{model_type} completed successfully")
        
        except Exception as e:
            logger.error(f"Error with {model_type}: {str(e)}")
            results.append({
                'model_type': model_type,
                'error': str(e)
            })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("="*60)
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def train_model_cli():
    """
    Command-line interface for training models.
    """
    import argparse
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.preprocessing import DataAcquisition, DataProcessor
    
    parser = argparse.ArgumentParser(
        description='Train Music Genre Classification Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Random Forest with default settings
  python src/model.py --model-type random_forest
  
  # Train SVM with custom test split
  python src/model.py --model-type svm --test-size 0.3
  
  # Train with hyperparameter tuning
  python src/model.py --model-type random_forest --tune-hyperparameters
  
  # Train and save with custom version
  python src/model.py --model-type mlp --version v2.0 --no-cv
        """
    )
    
    # Model selection
    parser.add_argument(
        '--model-type',
        type=str,
        default='random_forest',
        choices=['random_forest', 'svm', 'mlp', 'xgboost'],
        help='Type of model to train (default: random_forest)'
    )
    
    # Data configuration
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/dataset',
        help='Path to dataset directory (default: data/dataset)'
    )
    parser.add_argument(
        '--csv-file',
        type=str,
        default='features_30_sec.csv',
        choices=['features_30_sec.csv', 'features_3_sec.csv'],
        help='CSV file to use (default: features_30_sec.csv)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of dataset for testing (default: 0.2)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Training configuration
    parser.add_argument(
        '--tune-hyperparameters',
        action='store_true',
        help='Perform hyperparameter tuning using GridSearchCV'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--no-cv',
        action='store_true',
        help='Disable cross-validation during training'
    )
    
    # Data processing options
    parser.add_argument(
        '--no-clean',
        action='store_true',
        help='Skip data cleaning (outlier removal)'
    )
    parser.add_argument(
        '--no-outliers',
        action='store_true',
        help='Skip outlier removal during training'
    )
    
    # Model saving
    parser.add_argument(
        '--version',
        type=str,
        default=None,
        help='Model version (default: auto-generated)'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory to save models (default: models)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='genre_classifier',
        help='Prefix for saved model files (default: genre_classifier)'
    )
    
    # Evaluation options
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation on test set'
    )
    
    # Output options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("MODEL TRAINING SCRIPT")
    logger.info("=" * 60)
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Data Path: {args.data_path}")
    logger.info(f"CSV File: {args.csv_file}")
    logger.info(f"Test Size: {args.test_size}")
    logger.info(f"Random State: {args.random_state}")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load and process data
        logger.info("\n[Step 1/5] Loading and processing data...")
        data_acq = DataAcquisition(base_path=args.data_path)
        csv_path = Path(args.data_path) / args.csv_file
        df = data_acq.load_csv_features(csv_path=csv_path)
        
        train_df, test_df = data_acq.split_data(
            df,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        # Step 2: Process data
        logger.info("\n[Step 2/5] Processing data...")
        processor = DataProcessor(models_dir=args.models_dir)
        
        X_train, y_train, metadata = processor.process_training_data(
            train_df,
            clean_data=not args.no_clean,
            remove_outliers=not args.no_outliers
        )
        
        X_test, y_test = processor.process_test_data(
            test_df,
            clean_data=not args.no_clean
        )
        
        class_names = metadata['class_names']
        logger.info(f"Features: {metadata['n_features']}, Classes: {metadata['n_classes']}")
        
        # Step 3: Initialize and train model
        logger.info(f"\n[Step 3/5] Training {args.model_type} model...")
        classifier = GenreClassifier(
            model_type=args.model_type,
            models_dir=args.models_dir
        )
        
        # Hyperparameter tuning if requested
        if args.tune_hyperparameters:
            logger.info("Performing hyperparameter tuning...")
            tuning_results = classifier.tune_hyperparameters(
                X_train, y_train,
                cv_folds=args.cv_folds
            )
            logger.info(f"Best parameters: {tuning_results['best_params']}")
            logger.info(f"Best CV score: {tuning_results['best_score']:.4f}")
        
        # Train model
        train_metrics = classifier.train(
            X_train, y_train,
            use_cross_validation=not args.no_cv,
            cv_folds=args.cv_folds
        )
        
        logger.info(f"Training completed!")
        logger.info(f"Train Accuracy: {train_metrics['train_accuracy']:.4f}")
        if train_metrics['cv_mean']:
            logger.info(f"CV Accuracy: {train_metrics['cv_mean']:.4f} ± {train_metrics['cv_std']:.4f}")
        
        # Step 4: Evaluate model
        if not args.skip_evaluation:
            logger.info("\n[Step 4/5] Evaluating model on test set...")
            test_metrics = classifier.evaluate(X_test, y_test, class_names=class_names)
            
            logger.info("\n" + "=" * 60)
            logger.info("EVALUATION RESULTS")
            logger.info("=" * 60)
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Test Precision: {test_metrics['precision_weighted']:.4f}")
            logger.info(f"Test Recall: {test_metrics['recall_weighted']:.4f}")
            logger.info(f"Test F1-Score: {test_metrics['f1_weighted']:.4f}")
        else:
            logger.info("\n[Step 4/5] Skipping evaluation (--skip-evaluation)")
            test_metrics = None
        
        # Step 5: Save model
        logger.info("\n[Step 5/5] Saving model...")
        model_path = classifier.save_model(
            version=args.version,
            prefix=args.prefix
        )
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Metadata saved to: {Path(model_path).parent / f'{args.prefix}_{args.model_type}_metadata.json'}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Model Type: {args.model_type}")
        print(f"Model Version: {classifier.model_version}")
        print(f"Train Accuracy: {train_metrics['train_accuracy']:.4f}")
        if train_metrics['cv_mean']:
            print(f"CV Accuracy: {train_metrics['cv_mean']:.4f} ± {train_metrics['cv_std']:.4f}")
        if test_metrics:
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Test F1-Score: {test_metrics['f1_weighted']:.4f}")
        print(f"Model Path: {model_path}")
        print("=" * 60)
        
        return classifier, train_metrics, test_metrics
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


def main():
    """
    Main function - can be used for simple testing or CLI training.
    """
    import sys
    
    # Check if running as CLI (with arguments) or as simple script
    if len(sys.argv) > 1:
        # Run CLI training
        train_model_cli()
    else:
        # Simple example training (backward compatibility)
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.preprocessing import DataAcquisition, DataProcessor
        
        print("Loading and processing data...")
        data_acq = DataAcquisition(base_path="data/dataset")
        df = data_acq.load_csv_features()
        train_df, test_df = data_acq.split_data(df, test_size=0.2, random_state=42)
        
        processor = DataProcessor(models_dir="models")
        X_train, y_train, metadata = processor.process_training_data(train_df)
        X_test, y_test = processor.process_test_data(test_df)
        
        class_names = metadata['class_names']
        
        # Train a single model
        print("\nTraining Random Forest model...")
        classifier = GenreClassifier(model_type="random_forest")
        classifier.train(X_train, y_train)
        
        # Evaluate
        metrics = classifier.evaluate(X_test, y_test, class_names=class_names)
        
        # Save model
        model_path = classifier.save_model()
        print(f"\nModel saved to: {model_path}")
        
        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETED")
        print("="*60)
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test F1-Score: {metrics['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()

