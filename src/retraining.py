"""
Model Retraining Module for Music Genre Classification

This module handles:
- Retraining existing models with new data
- Model versioning and rollback
- Training history tracking
- Data version management
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import shutil
import joblib

from src.model import GenreClassifier
from src.preprocessing import DataProcessor, DataAcquisition
from src.feature_extraction import FeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelRetrainer:
    """
    Class for retraining existing models with new data.
    """
    
    def __init__(self, models_dir: str = "models", data_dir: str = "data"):
        """
        Initialize ModelRetrainer.
        
        Args:
            models_dir: Directory containing models
            data_dir: Directory containing data
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.uploads_dir = self.data_dir / "uploads"
        self.history_dir = self.models_dir / "training_history"
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ModelRetrainer with models_dir: {self.models_dir}")
    
    def load_existing_model(self, model_version: Optional[str] = None) -> Tuple[GenreClassifier, Dict]:
        """
        Load existing model and its metadata.
        
        Args:
            model_version: Version of model to load (None = latest)
            
        Returns:
            Tuple of (classifier, metadata)
        """
        if model_version is None:
            # Find latest version
            version_dirs = sorted([d for d in self.models_dir.iterdir() 
                                 if d.is_dir() and d.name.startswith('v')])
            if not version_dirs:
                raise FileNotFoundError("No model versions found")
            model_version_dir = version_dirs[-1]
            model_version = model_version_dir.name
        else:
            model_version_dir = self.models_dir / model_version
        
        if not model_version_dir.exists():
            raise FileNotFoundError(f"Model version not found: {model_version}")
        
        # Find model file
        model_files = list(model_version_dir.glob("genre_classifier_*.pkl"))
        if not model_files:
            raise FileNotFoundError(f"No model file found in {model_version_dir}")
        
        model_file = model_files[0]
        
        # Extract model type from filename
        model_type = model_file.stem.replace("genre_classifier_", "")
        
        # Load metadata
        metadata_file = model_version_dir / f"genre_classifier_{model_type}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Load model
        classifier = GenreClassifier(model_type=model_type, models_dir=str(self.models_dir))
        classifier.load_model(model_file)
        
        logger.info(f"Loaded model version: {model_version}, type: {model_type}")
        
        return classifier, metadata
    
    def load_new_data(self, data_source: Union[str, Path, pd.DataFrame],
                     data_format: str = "csv") -> pd.DataFrame:
        """
        Load new training data from various sources.
        
        Args:
            data_source: Path to data file, directory, or DataFrame
            data_format: Format of data ('csv', 'audio', 'features')
            
        Returns:
            DataFrame with new data
        """
        logger.info(f"Loading new data from: {data_source}, format: {data_format}")
        
        if isinstance(data_source, pd.DataFrame):
            return data_source.copy()
        
        data_source = Path(data_source)
        
        if data_format == "csv":
            if not data_source.exists():
                raise FileNotFoundError(f"Data file not found: {data_source}")
            df = pd.read_csv(data_source)
            logger.info(f"Loaded {len(df)} samples from CSV")
        
        elif data_format == "audio":
            # Extract features from audio files
            extractor = FeatureExtractor()
            df = extractor.extract_features_from_directory(
                data_source,
                pattern="*.wav",
                recursive=True
            )
            logger.info(f"Extracted features from {len(df)} audio files")
        
        elif data_format == "features":
            # Load from features directory
            data_acq = DataAcquisition(base_path=str(data_source))
            df = data_acq.load_csv_features()
            logger.info(f"Loaded {len(df)} samples from features")
        
        else:
            raise ValueError(f"Unknown data format: {data_format}")
        
        # Validate data
        if 'label' not in df.columns:
            raise ValueError("Data must contain 'label' column")
        
        return df
    
    def combine_datasets(self, existing_df: pd.DataFrame, 
                        new_df: pd.DataFrame,
                        strategy: str = "append") -> pd.DataFrame:
        """
        Combine existing and new datasets.
        
        Args:
            existing_df: Existing training data
            new_df: New training data
            strategy: Combination strategy ('append', 'replace', 'merge')
            
        Returns:
            Combined DataFrame
        """
        logger.info(f"Combining datasets with strategy: {strategy}")
        logger.info(f"Existing: {len(existing_df)} samples, New: {len(new_df)} samples")
        
        if strategy == "append":
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            logger.info(f"Appended datasets: {len(combined_df)} total samples")
        
        elif strategy == "replace":
            combined_df = new_df.copy()
            logger.info(f"Replaced with new data: {len(combined_df)} samples")
        
        elif strategy == "merge":
            # Merge on filename if available, otherwise append
            if 'filename' in existing_df.columns and 'filename' in new_df.columns:
                # Remove duplicates based on filename
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['filename'], keep='last')
                logger.info(f"Merged datasets (removed duplicates): {len(combined_df)} samples")
            else:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                logger.info(f"Merged datasets: {len(combined_df)} samples")
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return combined_df
    
    def retrain_model(self, 
                     model_version: Optional[str] = None,
                     new_data_source: Optional[Union[str, Path, pd.DataFrame]] = None,
                     data_format: str = "csv",
                     combine_strategy: str = "append",
                     use_existing_data: bool = True,
                     test_size: float = 0.2,
                     random_state: int = 42,
                     save_new_version: bool = True,
                     new_version: Optional[str] = None) -> Tuple[GenreClassifier, Dict]:
        """
        Complete retraining pipeline.
        
        Args:
            model_version: Version of model to retrain (None = latest)
            new_data_source: Source of new training data
            data_format: Format of new data ('csv', 'audio', 'features')
            combine_strategy: How to combine datasets ('append', 'replace', 'merge')
            use_existing_data: Whether to use existing training data
            test_size: Proportion for test set
            random_state: Random seed
            save_new_version: Whether to save retrained model
            new_version: Version name for new model (None = auto-generate)
            
        Returns:
            Tuple of (retrained_classifier, metadata)
        """
        logger.info("=" * 60)
        logger.info("STARTING MODEL RETRAINING PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Load existing model
        logger.info("\n[Step 1/6] Loading existing model...")
        existing_classifier, existing_metadata = self.load_existing_model(model_version)
        model_type = existing_classifier.model_type
        
        # Step 2: Load existing training data (if requested)
        existing_train_df = None
        if use_existing_data:
            logger.info("\n[Step 2/6] Loading existing training data...")
            try:
                # Try to load from train directory
                train_file = self.data_dir / "train" / "train.csv"
                if train_file.exists():
                    existing_train_df = pd.read_csv(train_file)
                    logger.info(f"Loaded {len(existing_train_df)} existing training samples")
                else:
                    logger.warning("Existing training data not found, using only new data")
            except Exception as e:
                logger.warning(f"Could not load existing training data: {e}")
        
        # Step 3: Load new data
        explicit_new_df = None
        used_uploaded_data = False
        if new_data_source is None:
            upload_csv_files = sorted(
                self.uploads_dir.glob("*.csv"),
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=True
            )
            if upload_csv_files:
                new_data_source = upload_csv_files[0]
                data_format = "csv"
                logger.info(f"Found uploaded CSV data: {new_data_source}")
                used_uploaded_data = True
            else:
                audio_files = self._find_uploaded_audio_files()
                if audio_files:
                    logger.info(f"Found {len(audio_files)} uploaded audio files. Extracting features...")
                    explicit_new_df = self._build_dataframe_from_audio_uploads(audio_files)
                    used_uploaded_data = True
                else:
                    raise ValueError("No new data provided and no uploaded data found")
        
        logger.info("\n[Step 3/6] Loading new training data...")
        if explicit_new_df is not None:
            new_df = explicit_new_df
        else:
            new_df = self.load_new_data(new_data_source, data_format=data_format)
        
        # Step 4: Combine datasets
        logger.info("\n[Step 4/6] Combining datasets...")
        if existing_train_df is not None and use_existing_data:
            combined_df = self.combine_datasets(existing_train_df, new_df, combine_strategy)
        else:
            combined_df = new_df.copy()
            logger.info("Using only new data (existing data not used)")
        
        # Validate combined data
        if len(combined_df) == 0:
            raise ValueError("Combined dataset is empty")
        
        logger.info(f"Combined dataset: {len(combined_df)} samples")
        logger.info(f"Genre distribution:\n{combined_df['label'].value_counts().sort_index()}")
        
        # Step 5: Process data and retrain
        logger.info("\n[Step 5/6] Processing data and retraining model...")
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            combined_df,
            test_size=test_size,
            random_state=random_state,
            stratify=combined_df['label'] if 'label' in combined_df.columns else None,
            shuffle=True
        )
        
        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        # Process data
        processor = DataProcessor(models_dir=str(self.models_dir))
        X_train, y_train, metadata = processor.process_training_data(train_df)
        X_test, y_test = processor.process_test_data(test_df)
        
        class_names = metadata['class_names']
        
        # Retrain model
        retrained_classifier = GenreClassifier(
            model_type=model_type,
            models_dir=str(self.models_dir)
        )
        
        train_metrics = retrained_classifier.train(
            X_train, y_train,
            use_cross_validation=True,
            cv_folds=5
        )
        
        # Evaluate retrained model
        test_metrics = retrained_classifier.evaluate(X_test, y_test, class_names=class_names)
        
        # Step 6: Save new version
        if save_new_version:
            logger.info("\n[Step 6/6] Saving retrained model...")
            
            # Generate version if not provided
            if new_version is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Increment version number
                if model_version:
                    try:
                        base_version = model_version.split('_')[0]  # e.g., 'v1'
                        version_num = int(base_version.replace('v', ''))
                        new_version = f"v{version_num + 1}_{timestamp}"
                    except:
                        new_version = f"v2_{timestamp}"
                else:
                    new_version = f"v2_{timestamp}"
            
            model_path = retrained_classifier.save_model(version=new_version)
            
            # Save training history
            self._save_training_history(
                old_version=model_version,
                new_version=new_version,
                existing_metadata=existing_metadata,
                new_metadata=retrained_classifier.metadata,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                data_info={
                    'existing_samples': len(existing_train_df) if existing_train_df is not None else 0,
                    'new_samples': len(new_df),
                    'combined_samples': len(combined_df),
                    'train_samples': len(train_df),
                    'test_samples': len(test_df)
                }
            )
            
            logger.info(f"Retrained model saved as version: {new_version}")
        else:
            logger.info("\n[Step 6/6] Skipping model save (save_new_version=False)")
        
        logger.info("=" * 60)
        logger.info("RETRAINING PIPELINE COMPLETED")
        logger.info("=" * 60)
        
        # Clean uploads directory if data was consumed from there
        if used_uploaded_data:
            self._cleanup_uploads_dir()
        
        return retrained_classifier, retrained_classifier.metadata
    
    def _find_uploaded_audio_files(self) -> List[Path]:
        """
        Find audio files inside the uploads directory.
        """
        if not self.uploads_dir.exists():
            return []
        
        allowed_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')
        audio_files: List[Path] = []
        
        for ext in allowed_extensions:
            audio_files.extend(self.uploads_dir.rglob(f"*{ext}"))
        
        audio_files = [path for path in audio_files if path.is_file()]
        audio_files.sort()
        
        return audio_files
    
    def _infer_label_from_audio_path(self, audio_path: Path) -> str:
        """
        Infer the genre label from an audio file path.
        """
        uploads_root = self.uploads_dir.resolve()
        audio_path = audio_path.resolve()
        
        label_candidate: Optional[str] = None
        try:
            relative_parts = audio_path.relative_to(uploads_root).parts
            if len(relative_parts) > 1:
                label_candidate = relative_parts[0]
        except ValueError:
            # File not under uploads directory
            pass
        
        if not label_candidate:
            stem = audio_path.stem
            label_candidate = stem.split('.')[0] if '.' in stem else stem
        
        label_candidate = label_candidate.strip().lower()
        if not label_candidate:
            raise ValueError(f"Unable to infer label from filename: {audio_path.name}")
        
        return label_candidate
    
    def _build_dataframe_from_audio_uploads(self, audio_files: List[Path]) -> pd.DataFrame:
        """
        Convert uploaded audio files into a labeled feature DataFrame.
        """
        extractor = FeatureExtractor()
        features_df = extractor.extract_features_batch([str(path) for path in audio_files])
        
        if features_df.empty:
            raise ValueError("Failed to extract features from uploaded audio files")
        
        label_map = {
            path.name: self._infer_label_from_audio_path(path)
            for path in audio_files
        }
        
        features_df['label'] = features_df['filename'].map(label_map)
        if features_df['label'].isna().any():
            missing = features_df.loc[features_df['label'].isna(), 'filename'].tolist()
            raise ValueError(f"Could not determine labels for files: {missing}")
        
        return features_df
    
    def _cleanup_uploads_dir(self) -> None:
        """
        Remove all files from the uploads directory after they are consumed.
        """
        logger.info("Cleaning up uploads directory now that data has been consumed")
        
        if not self.uploads_dir.exists():
            return
        
        for path in self.uploads_dir.iterdir():
            try:
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            except Exception as exc:
                logger.warning(f"Failed to delete {path}: {exc}")
        
        logger.info("Uploads directory cleaned")
    
    def _save_training_history(self, old_version: Optional[str], new_version: str,
                              existing_metadata: Dict, new_metadata: Dict,
                              train_metrics: Dict, test_metrics: Dict,
                              data_info: Dict) -> None:
        """
        Save training history record.
        
        Args:
            old_version: Previous model version
            new_version: New model version
            existing_metadata: Metadata from old model
            new_metadata: Metadata from new model
            train_metrics: Training metrics
            test_metrics: Test metrics
            data_info: Information about data used
        """
        history_record = {
            'retraining_date': datetime.now().isoformat(),
            'old_version': old_version,
            'new_version': new_version,
            'model_type': new_metadata.get('model_type'),
            'data_info': data_info,
            'metrics_comparison': {
                'old': {
                    'train_accuracy': existing_metadata.get('train_accuracy'),
                    'test_accuracy': existing_metadata.get('test_accuracy'),
                    'test_f1': existing_metadata.get('test_f1')
                },
                'new': {
                    'train_accuracy': train_metrics.get('train_accuracy'),
                    'test_accuracy': test_metrics.get('accuracy'),
                    'test_f1': test_metrics.get('f1_weighted')
                }
            },
            'improvement': {
                'accuracy_change': test_metrics.get('accuracy', 0) - existing_metadata.get('test_accuracy', 0),
                'f1_change': test_metrics.get('f1_weighted', 0) - existing_metadata.get('test_f1', 0)
            }
        }
        
        history_file = self.history_dir / f"retraining_{new_version}.json"
        with open(history_file, 'w') as f:
            json.dump(history_record, f, indent=2)
        
        logger.info(f"Training history saved to {history_file}")
    
    def get_training_history(self) -> pd.DataFrame:
        """
        Get all training history records.
        
        Returns:
            DataFrame with training history
        """
        history_files = list(self.history_dir.glob("retraining_*.json"))
        
        if not history_files:
            logger.warning("No training history found")
            return pd.DataFrame()
        
        records = []
        for history_file in history_files:
            with open(history_file, 'r') as f:
                records.append(json.load(f))
        
        df = pd.DataFrame(records)
        return df
    
    def rollback_model(self, target_version: str, 
                      keep_current: bool = True) -> bool:
        """
        Rollback to a previous model version.
        
        Args:
            target_version: Version to rollback to
            keep_current: Whether to keep current version
            
        Returns:
            True if rollback successful
        """
        logger.info(f"Rolling back to version: {target_version}")
        
        target_dir = self.models_dir / target_version
        if not target_dir.exists():
            raise FileNotFoundError(f"Target version not found: {target_version}")
        
        # Find current version
        current_version = None
        version_dirs = sorted([d for d in self.models_dir.iterdir() 
                              if d.is_dir() and d.name.startswith('v')])
        if version_dirs:
            current_version = version_dirs[-1].name
        
        if current_version == target_version:
            logger.warning(f"Already at version {target_version}")
            return False
        
        # Create rollback version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rollback_version = f"{target_version}_rollback_{timestamp}"
        rollback_dir = self.models_dir / rollback_version
        
        # Copy target version files
        shutil.copytree(target_dir, rollback_dir)
        
        # Update metadata
        metadata_files = list(rollback_dir.glob("*_metadata.json"))
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            metadata['version'] = rollback_version
            metadata['rollback_from'] = current_version
            metadata['rollback_date'] = datetime.now().isoformat()
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Rolled back to version {target_version} (saved as {rollback_version})")
        
        return True
    
    def list_model_versions(self) -> List[Dict]:
        """
        List all available model versions.
        
        Returns:
            List of version information dictionaries
        """
        version_dirs = sorted([d for d in self.models_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('v')])
        
        versions = []
        for version_dir in version_dirs:
            metadata_files = list(version_dir.glob("*_metadata.json"))
            if metadata_files:
                with open(metadata_files[0], 'r') as f:
                    metadata = json.load(f)
                versions.append({
                    'version': metadata.get('version', version_dir.name),
                    'model_type': metadata.get('model_type'),
                    'training_date': metadata.get('training_date'),
                    'test_accuracy': metadata.get('test_accuracy'),
                    'path': str(version_dir)
                })
            else:
                versions.append({
                    'version': version_dir.name,
                    'model_type': 'unknown',
                    'training_date': None,
                    'test_accuracy': None,
                    'path': str(version_dir)
                })
        
        return versions


def main():
    """
    Main function for testing the retraining module.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    print("=" * 60)
    print("MODEL RETRAINING MODULE TEST")
    print("=" * 60)
    
    # Initialize retrainer
    retrainer = ModelRetrainer()
    
    # List available versions
    print("\nAvailable model versions:")
    versions = retrainer.list_model_versions()
    for v in versions:
        print(f"  {v['version']}: {v['model_type']} (accuracy: {v['test_accuracy']})")
    
    if not versions:
        print("No models found. Train a model first.")
        return
    
    # Example: Retrain with new data (if available)
    print("\n" + "=" * 60)
    print("Retraining example (commented out - uncomment to test)")
    print("=" * 60)
    
    # Uncomment to test retraining:
    # retrained_classifier, metadata = retrainer.retrain_model(
    #     model_version=None,  # Use latest
    #     new_data_source="data/uploads/new_data.csv",
    #     combine_strategy="append"
    # )
    # print(f"Retrained model version: {metadata['version']}")
    # print(f"New test accuracy: {metadata['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()

