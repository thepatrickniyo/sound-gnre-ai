"""
Data Acquisition Module for Music Genre Classification

This module handles:
- Loading CSV feature files
- Loading audio files from directories
- Data validation
- Train/test splitting with stratification
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import librosa
import joblib
import json
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataAcquisition:
    """
    Data Acquisition class for loading and validating music genre dataset.
    """
    
    def __init__(self, base_path: str = "data/dataset"):
        """
        Initialize DataAcquisition.
        
        Args:
            base_path: Base path to the dataset directory
        """
        self.base_path = Path(base_path)
        self.features_30_sec_path = self.base_path / "features_30_sec.csv"
        self.features_3_sec_path = self.base_path / "features_3_sec.csv"
        self.audio_dir = self.base_path / "genres_original"
        self.train_dir = Path("data/train")
        self.test_dir = Path("data/test")
        
        # Create train and test directories if they don't exist
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DataAcquisition with base_path: {self.base_path}")
    
    def load_csv_features(self, csv_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load features from CSV file.
        
        Args:
            csv_path: Path to CSV file. If None, uses features_30_sec.csv by default
            
        Returns:
            DataFrame with features
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV file is empty or invalid
        """
        if csv_path is None:
            csv_path = self.features_30_sec_path
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        logger.info(f"Loading CSV features from: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            
            if df.empty:
                raise ValueError(f"CSV file is empty: {csv_path}")
            
            logger.info(f"Successfully loaded {len(df)} rows from CSV")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_path}: {str(e)}")
            raise
    
    def get_audio_files_by_genre(self) -> Dict[str, List[Path]]:
        """
        Load audio files organized by genre.
        
        Returns:
            Dictionary mapping genre names to lists of audio file paths
            
        Raises:
            FileNotFoundError: If audio directory doesn't exist
        """
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")
        
        logger.info(f"Loading audio files from: {self.audio_dir}")
        
        audio_files_by_genre = {}
        
        # Get all genre directories
        genre_dirs = [d for d in self.audio_dir.iterdir() if d.is_dir()]
        
        if not genre_dirs:
            logger.warning(f"No genre directories found in {self.audio_dir}")
            return audio_files_by_genre
        
        for genre_dir in genre_dirs:
            genre_name = genre_dir.name
            audio_files = list(genre_dir.glob("*.wav"))
            
            if audio_files:
                audio_files_by_genre[genre_name] = sorted(audio_files)
                logger.info(f"Found {len(audio_files)} audio files for genre: {genre_name}")
            else:
                logger.warning(f"No WAV files found in genre directory: {genre_name}")
        
        total_files = sum(len(files) for files in audio_files_by_genre.values())
        logger.info(f"Total audio files loaded: {total_files}")
        
        return audio_files_by_genre
    
    def validate_audio_file(self, audio_path: Path) -> bool:
        """
        Validate an audio file by attempting to load it with librosa.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            # Try to load the audio file
            y, sr = librosa.load(str(audio_path), duration=1.0)  # Load only 1 second for validation
            
            if len(y) == 0:
                logger.warning(f"Audio file is empty: {audio_path}")
                return False
            
            return True
        
        except Exception as e:
            logger.warning(f"Invalid audio file {audio_path}: {str(e)}")
            return False
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate DataFrame for missing values and data consistency.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'missing_percentage': {},
            'dtypes': df.dtypes.to_dict(),
            'is_valid': True
        }
        
        # Check for missing values
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        
        validation_results['missing_values'] = missing_values[missing_values > 0].to_dict()
        validation_results['missing_percentage'] = missing_percentage[missing_percentage > 0].to_dict()
        
        if len(validation_results['missing_values']) > 0:
            logger.warning(f"Found missing values in {len(validation_results['missing_values'])} columns")
            validation_results['is_valid'] = False
        
        # Check for label column
        if 'label' not in df.columns:
            logger.error("Label column 'label' not found in DataFrame")
            validation_results['is_valid'] = False
        
        # Check for feature consistency
        expected_feature_columns = [
            'chroma_stft_mean', 'chroma_stft_var',
            'rms_mean', 'rms_var',
            'spectral_centroid_mean', 'spectral_centroid_var',
            'tempo'
        ]
        
        missing_features = [col for col in expected_feature_columns if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing expected feature columns: {missing_features}")
        
        logger.info(f"Data validation completed. Valid: {validation_results['is_valid']}")
        
        return validation_results
    
    def validate_audio_files(self, audio_files_by_genre: Dict[str, List[Path]], 
                            sample_size: Optional[int] = None) -> Dict[str, Dict]:
        """
        Validate audio files (can sample for large datasets).
        
        Args:
            audio_files_by_genre: Dictionary of genre to audio file paths
            sample_size: Number of files to sample per genre for validation (None = all)
            
        Returns:
            Dictionary with validation results per genre
        """
        validation_results = {}
        
        for genre, audio_files in audio_files_by_genre.items():
            logger.info(f"Validating audio files for genre: {genre}")
            
            # Sample files if specified
            files_to_validate = audio_files
            if sample_size and len(audio_files) > sample_size:
                files_to_validate = np.random.choice(audio_files, sample_size, replace=False)
                logger.info(f"Sampling {sample_size} files out of {len(audio_files)} for validation")
            
            valid_count = 0
            invalid_files = []
            
            for audio_file in files_to_validate:
                if self.validate_audio_file(audio_file):
                    valid_count += 1
                else:
                    invalid_files.append(str(audio_file))
            
            validation_results[genre] = {
                'total_files': len(audio_files),
                'validated_files': len(files_to_validate),
                'valid_count': valid_count,
                'invalid_count': len(invalid_files),
                'invalid_files': invalid_files,
                'validity_rate': (valid_count / len(files_to_validate)) * 100 if files_to_validate else 0
            }
            
            logger.info(f"Genre {genre}: {valid_count}/{len(files_to_validate)} files valid")
        
        return validation_results
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                   random_state: int = 42, stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train and test sets with stratification.
        
        Args:
            df: DataFrame to split
            test_size: Proportion of dataset to include in test split (0.0 to 1.0)
            random_state: Random seed for reproducibility
            stratify: Whether to use stratified splitting based on labels
            
        Returns:
            Tuple of (train_df, test_df)
            
        Raises:
            ValueError: If label column is missing or test_size is invalid
        """
        if 'label' not in df.columns:
            raise ValueError("DataFrame must contain 'label' column for splitting")
        
        if not (0 < test_size < 1):
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        
        logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}, stratify={stratify}")
        
        # Get labels for stratification
        labels = df['label'] if stratify else None
        
        # Split the data
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
            shuffle=True
        )
        
        logger.info(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        # Log label distribution
        if stratify:
            train_labels = train_df['label'].value_counts().sort_index()
            test_labels = test_df['label'].value_counts().sort_index()
            
            logger.info("Train set label distribution:")
            for label, count in train_labels.items():
                logger.info(f"  {label}: {count} samples")
            
            logger.info("Test set label distribution:")
            for label, count in test_labels.items():
                logger.info(f"  {label}: {count} samples")
        
        return train_df, test_df
    
    def save_split_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                       train_filename: str = "train.csv", 
                       test_filename: str = "test.csv") -> None:
        """
        Save train and test datasets to CSV files.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            train_filename: Filename for training data
            test_filename: Filename for test data
        """
        train_path = self.train_dir / train_filename
        test_path = self.test_dir / test_filename
        
        logger.info(f"Saving train data to: {train_path}")
        train_df.to_csv(train_path, index=False)
        
        logger.info(f"Saving test data to: {test_path}")
        test_df.to_csv(test_path, index=False)
        
        logger.info("Data split saved successfully")
    
    def process_dataset(self, csv_path: Optional[Path] = None, 
                       test_size: float = 0.2, random_state: int = 42,
                       validate_audio: bool = False, audio_sample_size: Optional[int] = 10) -> Dict:
        """
        Complete data acquisition and processing pipeline.
        
        Args:
            csv_path: Path to CSV file (None = use default)
            test_size: Proportion for test set
            random_state: Random seed
            validate_audio: Whether to validate audio files
            audio_sample_size: Number of audio files to sample for validation
            
        Returns:
            Dictionary with processing results and dataframes
        """
        logger.info("=" * 60)
        logger.info("Starting Data Acquisition Pipeline")
        logger.info("=" * 60)
        
        results = {
            'raw_data': None,
            'train_data': None,
            'test_data': None,
            'validation_results': {},
            'audio_validation_results': {}
        }
        
        # Step 1: Load CSV features
        try:
            df = self.load_csv_features(csv_path)
            results['raw_data'] = df
        except Exception as e:
            logger.error(f"Failed to load CSV features: {str(e)}")
            raise
        
        # Step 2: Validate DataFrame
        try:
            validation_results = self.validate_dataframe(df)
            results['validation_results'] = validation_results
            
            if not validation_results['is_valid']:
                logger.warning("Data validation found issues. Proceeding with caution.")
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise
        
        # Step 3: Validate audio files (optional)
        if validate_audio:
            try:
                audio_files_by_genre = self.get_audio_files_by_genre()
                audio_validation_results = self.validate_audio_files(
                    audio_files_by_genre, 
                    sample_size=audio_sample_size
                )
                results['audio_validation_results'] = audio_validation_results
            except Exception as e:
                logger.warning(f"Audio validation failed: {str(e)}")
        
        # Step 4: Split data
        try:
            train_df, test_df = self.split_data(df, test_size=test_size, random_state=random_state)
            results['train_data'] = train_df
            results['test_data'] = test_df
        except Exception as e:
            logger.error(f"Data splitting failed: {str(e)}")
            raise
        
        # Step 5: Save split data
        try:
            self.save_split_data(train_df, test_df)
        except Exception as e:
            logger.error(f"Failed to save split data: {str(e)}")
            raise
        
        logger.info("=" * 60)
        logger.info("Data Acquisition Pipeline Completed Successfully")
        logger.info("=" * 60)
        
        return results


class DataProcessor:
    """
    Data Processing Pipeline class for cleaning, transforming, and preparing data for ML.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize DataProcessor.
        
        Args:
            models_dir: Directory to save encoders and scalers
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        self.feature_columns = None
        self.is_fitted = False
        
        logger.info(f"Initialized DataProcessor with models_dir: {self.models_dir}")
    
    def clean_data(self, df: pd.DataFrame, 
                   handle_missing: str = 'drop',
                   remove_outliers: bool = True,
                   outlier_method: str = 'iqr',
                   outlier_threshold: float = 3.0) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and outliers.
        
        Args:
            df: DataFrame to clean
            handle_missing: Strategy for missing values ('drop', 'mean', 'median', 'mode')
            remove_outliers: Whether to remove outliers
            outlier_method: Method for outlier detection ('iqr', 'zscore')
            outlier_threshold: Threshold for outlier detection
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        original_size = len(df)
        df_cleaned = df.copy()
        
        # Handle missing values
        missing_before = df_cleaned.isnull().sum().sum()
        if missing_before > 0:
            logger.info(f"Found {missing_before} missing values")
            
            if handle_missing == 'drop':
                df_cleaned = df_cleaned.dropna()
                logger.info(f"Dropped rows with missing values: {len(df) - len(df_cleaned)} rows")
            elif handle_missing == 'mean':
                numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
                df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
                logger.info("Filled missing values with mean")
            elif handle_missing == 'median':
                numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
                df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
                logger.info("Filled missing values with median")
            elif handle_missing == 'mode':
                for col in df_cleaned.columns:
                    df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
                logger.info("Filled missing values with mode")
        
        # Remove outliers
        if remove_outliers and 'label' in df_cleaned.columns:
            # Get feature columns (exclude metadata columns)
            exclude_cols = ['filename', 'label', 'length']
            feature_cols = [col for col in df_cleaned.columns if col not in exclude_cols]
            
            if feature_cols:
                if outlier_method == 'iqr':
                    # IQR method
                    Q1 = df_cleaned[feature_cols].quantile(0.25)
                    Q3 = df_cleaned[feature_cols].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - outlier_threshold * IQR
                    upper_bound = Q3 + outlier_threshold * IQR
                    
                    # Remove outliers
                    mask = ((df_cleaned[feature_cols] >= lower_bound) & 
                           (df_cleaned[feature_cols] <= upper_bound)).all(axis=1)
                    outliers_removed = len(df_cleaned) - mask.sum()
                    df_cleaned = df_cleaned[mask]
                    
                    logger.info(f"Removed {outliers_removed} outliers using IQR method")
                
                elif outlier_method == 'zscore':
                    # Z-score method
                    if not SCIPY_AVAILABLE:
                        raise ImportError("scipy is required for zscore outlier detection. Install with: pip install scipy")
                    z_scores = np.abs(stats.zscore(df_cleaned[feature_cols]))
                    mask = (z_scores < outlier_threshold).all(axis=1)
                    outliers_removed = len(df_cleaned) - mask.sum()
                    df_cleaned = df_cleaned[mask]
                    
                    logger.info(f"Removed {outliers_removed} outliers using Z-score method")
        
        final_size = len(df_cleaned)
        logger.info(f"Data cleaning completed: {original_size} -> {final_size} samples")
        
        return df_cleaned
    
    def encode_labels(self, labels: pd.Series, fit: bool = True) -> np.ndarray:
        """
        Encode genre labels to integers.
        
        Args:
            labels: Series of genre labels (strings)
            fit: Whether to fit the encoder (True for training, False for test)
            
        Returns:
            Encoded labels as numpy array
        """
        if fit:
            encoded = self.label_encoder.fit_transform(labels)
            logger.info(f"Fitted label encoder with {len(self.label_encoder.classes_)} classes")
            label_mapping = dict(zip(self.label_encoder.classes_, 
                                    self.label_encoder.transform(self.label_encoder.classes_)))
            logger.info(f"Label mapping: {label_mapping}")
        else:
            encoded = self.label_encoder.transform(labels)
            logger.info(f"Transformed {len(labels)} labels using fitted encoder")
        
        return encoded
    
    def get_feature_columns(self, df: pd.DataFrame, 
                           exclude_columns: Optional[List[str]] = None) -> List[str]:
        """
        Get list of feature columns (excluding metadata).
        
        Args:
            df: DataFrame
            exclude_columns: Columns to exclude from features
            
        Returns:
            List of feature column names
        """
        if exclude_columns is None:
            exclude_columns = ['filename', 'label', 'length']
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        return feature_columns
    
    def prepare_features_and_labels(self, df: pd.DataFrame,
                                   exclude_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and labels from DataFrame.
        
        Args:
            df: DataFrame with features and labels
            exclude_columns: Columns to exclude from features
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        if exclude_columns is None:
            exclude_columns = ['filename', 'label', 'length']
        
        if 'label' not in df.columns:
            raise ValueError("DataFrame must contain 'label' column")
        
        # Get feature columns
        self.feature_columns = self.get_feature_columns(df, exclude_columns)
        
        # Separate features and labels
        features_df = df[self.feature_columns].copy()
        labels_series = df['label'].copy()
        
        logger.info(f"Prepared {len(self.feature_columns)} features and {len(labels_series)} labels")
        
        return features_df, labels_series
    
    def fit_scaler(self, features_df: pd.DataFrame) -> StandardScaler:
        """
        Fit StandardScaler on feature data.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Fitted StandardScaler
        """
        logger.info("Fitting StandardScaler on features...")
        self.scaler.fit(features_df)
        self.is_fitted = True
        logger.info("Scaler fitted successfully")
        return self.scaler
    
    def transform_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scaler.
        
        Args:
            features_df: DataFrame with features to transform
            
        Returns:
            DataFrame with scaled features
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        logger.info("Transforming features...")
        scaled_features = self.scaler.transform(features_df)
        scaled_df = pd.DataFrame(scaled_features, columns=features_df.columns, index=features_df.index)
        logger.info("Features transformed successfully")
        return scaled_df
    
    def fit_transform_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler and transform features in one step.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            DataFrame with scaled features
        """
        self.fit_scaler(features_df)
        return self.transform_features(features_df)
    
    def process_training_data(self, train_df: pd.DataFrame,
                             clean_data: bool = True,
                             remove_outliers: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Complete processing pipeline for training data.
        
        Args:
            train_df: Training DataFrame
            clean_data: Whether to clean data
            remove_outliers: Whether to remove outliers
            
        Returns:
            Tuple of (X_train, y_train, metadata_dict)
        """
        logger.info("=" * 60)
        logger.info("Processing Training Data")
        logger.info("=" * 60)
        
        # Clean data
        if clean_data:
            train_df = self.clean_data(train_df, remove_outliers=remove_outliers)
        
        # Prepare features and labels
        X_train_df, y_train_series = self.prepare_features_and_labels(train_df)
        
        # Encode labels
        y_train = self.encode_labels(y_train_series, fit=True)
        
        # Scale features
        X_train_scaled = self.fit_transform_features(X_train_df)
        X_train = X_train_scaled.values
        
        # Metadata
        metadata = {
            'n_samples': len(X_train),
            'n_features': len(self.feature_columns),
            'n_classes': len(self.label_encoder.classes_),
            'class_names': self.label_encoder.classes_.tolist(),
            'feature_names': self.feature_columns
        }
        
        logger.info(f"Training data processed: {metadata['n_samples']} samples, "
                   f"{metadata['n_features']} features, {metadata['n_classes']} classes")
        logger.info("=" * 60)
        
        return X_train, y_train, metadata
    
    def process_test_data(self, test_df: pd.DataFrame,
                         clean_data: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete processing pipeline for test data (using fitted transformers).
        
        Args:
            test_df: Test DataFrame
            clean_data: Whether to clean data
            
        Returns:
            Tuple of (X_test, y_test)
        """
        if not self.is_fitted:
            raise ValueError("Processor not fitted. Call process_training_data() first.")
        
        logger.info("=" * 60)
        logger.info("Processing Test Data")
        logger.info("=" * 60)
        
        # Clean data (without removing outliers to preserve test set integrity)
        if clean_data:
            test_df = self.clean_data(test_df, remove_outliers=False)
        
        # Prepare features and labels
        X_test_df, y_test_series = self.prepare_features_and_labels(test_df)
        
        # Encode labels (using fitted encoder)
        y_test = self.encode_labels(y_test_series, fit=False)
        
        # Transform features (using fitted scaler)
        X_test_scaled = self.transform_features(X_test_df)
        X_test = X_test_scaled.values
        
        logger.info(f"Test data processed: {len(X_test)} samples")
        logger.info("=" * 60)
        
        return X_test, y_test
    
    def save_processor(self, prefix: str = "data_processor") -> None:
        """
        Save label encoder, scaler, and metadata.
        
        Args:
            prefix: Prefix for saved files
        """
        if not self.is_fitted:
            raise ValueError("Processor not fitted. Cannot save.")
        
        # Save label encoder
        encoder_path = self.models_dir / f"{prefix}_label_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_path)
        logger.info(f"Label encoder saved to {encoder_path}")
        
        # Save scaler
        scaler_path = self.models_dir / f"{prefix}_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'class_names': self.label_encoder.classes_.tolist(),
            'n_classes': len(self.label_encoder.classes_),
            'n_features': len(self.feature_columns) if self.feature_columns else 0
        }
        
        metadata_path = self.models_dir / f"{prefix}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_processor(self, prefix: str = "data_processor") -> None:
        """
        Load label encoder, scaler, and metadata.
        
        Args:
            prefix: Prefix for saved files
        """
        # Load label encoder
        encoder_path = self.models_dir / f"{prefix}_label_encoder.pkl"
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)
            logger.info(f"Label encoder loaded from {encoder_path}")
        else:
            raise FileNotFoundError(f"Label encoder not found: {encoder_path}")
        
        # Load scaler
        scaler_path = self.models_dir / f"{prefix}_scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            self.is_fitted = True
            logger.info(f"Scaler loaded from {scaler_path}")
        else:
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        # Load metadata
        metadata_path = self.models_dir / f"{prefix}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.feature_columns = metadata['feature_columns']
            logger.info(f"Metadata loaded from {metadata_path}")
        else:
            logger.warning(f"Metadata not found: {metadata_path}")


def main():
    """
    Main function for testing the Data Acquisition module.
    """
    # Initialize DataAcquisition
    data_acq = DataAcquisition(base_path="data/dataset")
    
    # Process dataset
    results = data_acq.process_dataset(
        test_size=0.2,
        random_state=42,
        validate_audio=True,
        audio_sample_size=5  # Sample 5 files per genre for validation
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATA ACQUISITION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(results['raw_data'])}")
    print(f"Train samples: {len(results['train_data'])}")
    print(f"Test samples: {len(results['test_data'])}")
    print(f"Number of features: {len(results['raw_data'].columns) - 1}")  # Exclude label
    print(f"Number of genres: {results['raw_data']['label'].nunique()}")
    print(f"Genres: {sorted(results['raw_data']['label'].unique())}")
    print("=" * 60)


if __name__ == "__main__":
    main()

