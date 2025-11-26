"""
Feature Extraction Module for Music Genre Classification

This module handles:
- Extracting audio features from raw audio files
- Batch processing of multiple audio files
- Feature normalization and scaling
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import librosa
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Feature extraction class for extracting audio features from music files.
    """
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        """
        Initialize FeatureExtractor.
        
        Args:
            sample_rate: Target sample rate for audio loading
            hop_length: Number of samples between successive frames
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.scaler = None
        
        logger.info(f"Initialized FeatureExtractor with sample_rate={sample_rate}, hop_length={hop_length}")
    
    def extract_features(self, audio_path: Union[str, Path], 
                        duration: Optional[float] = None) -> Dict[str, float]:
        """
        Extract all features from a single audio file.
        
        Args:
            audio_path: Path to audio file
            duration: Maximum duration to load (None = load entire file)
            
        Returns:
            Dictionary containing all extracted features
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio file cannot be loaded
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio file
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate, duration=duration)
            
            if len(y) == 0:
                raise ValueError(f"Audio file is empty: {audio_path}")
            
            logger.debug(f"Loaded audio: {audio_path.name}, length: {len(y)/sr:.2f}s, sr: {sr}")
            
            # Extract all features
            features = {}
            
            # Chroma features
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
            features['chroma_stft_mean'] = float(np.mean(chroma_stft))
            features['chroma_stft_var'] = float(np.var(chroma_stft))
            
            # RMS energy
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
            features['rms_mean'] = float(np.mean(rms))
            features['rms_var'] = float(np.var(rms))
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_var'] = float(np.var(spectral_centroids))
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['spectral_bandwidth_var'] = float(np.var(spectral_bandwidth))
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
            features['rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['rolloff_var'] = float(np.var(spectral_rolloff))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
            features['zero_crossing_rate_mean'] = float(np.mean(zcr))
            features['zero_crossing_rate_var'] = float(np.var(zcr))
            
            # Harmony and Perceptr (using harmonic and percussive components)
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmony = librosa.feature.rms(y=y_harmonic, hop_length=self.hop_length)[0]
            perceptr = librosa.feature.rms(y=y_percussive, hop_length=self.hop_length)[0]
            features['harmony_mean'] = float(np.mean(harmony))
            features['harmony_var'] = float(np.var(harmony))
            features['perceptr_mean'] = float(np.mean(perceptr))
            features['perceptr_var'] = float(np.var(perceptr))
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
            # Handle tempo as scalar (it may be an array)
            tempo_value = tempo[0] if isinstance(tempo, np.ndarray) and len(tempo) > 0 else float(tempo)
            features['tempo'] = float(tempo_value)
            
            # MFCC features (1-20 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=self.hop_length)
            for i in range(1, 21):  # MFCC 1-20
                features[f'mfcc{i}_mean'] = float(np.mean(mfccs[i-1]))
                features[f'mfcc{i}_var'] = float(np.var(mfccs[i-1]))
            
            # Audio length (number of samples)
            features['length'] = len(y)
            
            logger.debug(f"Extracted {len(features)} features from {audio_path.name}")
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {str(e)}")
            raise ValueError(f"Failed to extract features from {audio_path}: {str(e)}")
    
    def extract_features_batch(self, audio_paths: List[Union[str, Path]], 
                               duration: Optional[float] = None,
                               show_progress: bool = True) -> pd.DataFrame:
        """
        Extract features from multiple audio files in batch.
        
        Args:
            audio_paths: List of paths to audio files
            duration: Maximum duration to load per file (None = load entire file)
            show_progress: Whether to show progress logging
            
        Returns:
            DataFrame with extracted features, one row per audio file
        """
        if not audio_paths:
            logger.warning("No audio files provided for batch processing")
            return pd.DataFrame()
        
        logger.info(f"Starting batch feature extraction for {len(audio_paths)} files")
        
        features_list = []
        failed_files = []
        
        for idx, audio_path in enumerate(audio_paths, 1):
            try:
                features = self.extract_features(audio_path, duration=duration)
                features['filename'] = Path(audio_path).name
                features_list.append(features)
                
                if show_progress and idx % 10 == 0:
                    logger.info(f"Processed {idx}/{len(audio_paths)} files")
            
            except Exception as e:
                logger.warning(f"Failed to extract features from {audio_path}: {str(e)}")
                failed_files.append(str(audio_path))
                continue
        
        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} files out of {len(audio_paths)}")
        
        if not features_list:
            logger.error("No features extracted from any files")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        # Reorder columns: filename first, then length, then other features
        cols = ['filename', 'length'] + [c for c in df.columns if c not in ['filename', 'length']]
        df = df[cols]
        
        logger.info(f"Successfully extracted features from {len(features_list)} files")
        
        return df
    
    def extract_features_from_upload(self, audio_data: bytes, 
                                     filename: str = "uploaded_audio.wav",
                                     duration: Optional[float] = None) -> Dict[str, float]:
        """
        Extract features from uploaded audio data (bytes).
        
        Args:
            audio_data: Audio file data as bytes
            filename: Original filename (for logging)
            duration: Maximum duration to load (None = load entire file)
            
        Returns:
            Dictionary containing all extracted features
        """
        import tempfile
        
        # Save uploaded data to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name
        
        try:
            # Extract features
            features = self.extract_features(tmp_path, duration=duration)
            features['filename'] = filename
            
            return features
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def fit_scaler(self, features_df: pd.DataFrame, 
                   exclude_columns: Optional[List[str]] = None) -> StandardScaler:
        """
        Fit a StandardScaler on feature data.
        
        Args:
            features_df: DataFrame with features
            exclude_columns: Columns to exclude from scaling (e.g., 'filename', 'label')
            
        Returns:
            Fitted StandardScaler
        """
        if exclude_columns is None:
            exclude_columns = ['filename', 'label', 'length']
        
        # Get feature columns (exclude non-feature columns)
        feature_columns = [col for col in features_df.columns 
                          if col not in exclude_columns]
        
        if not feature_columns:
            raise ValueError("No feature columns found for scaling")
        
        logger.info(f"Fitting scaler on {len(feature_columns)} features")
        
        # Fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(features_df[feature_columns])
        
        logger.info("Scaler fitted successfully")
        
        return self.scaler
    
    def transform_features(self, features_df: pd.DataFrame,
                           exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Transform features using fitted scaler.
        
        Args:
            features_df: DataFrame with features to transform
            exclude_columns: Columns to exclude from scaling
            
        Returns:
            DataFrame with scaled features
            
        Raises:
            ValueError: If scaler is not fitted
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        if exclude_columns is None:
            exclude_columns = ['filename', 'label', 'length']
        
        # Get feature columns
        feature_columns = [col for col in features_df.columns 
                          if col not in exclude_columns]
        
        # Create a copy to avoid modifying original
        scaled_df = features_df.copy()
        
        # Transform features
        scaled_features = self.scaler.transform(features_df[feature_columns])
        scaled_df[feature_columns] = scaled_features
        
        logger.info(f"Transformed {len(feature_columns)} features")
        
        return scaled_df
    
    def fit_transform_features(self, features_df: pd.DataFrame,
                              exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit scaler and transform features in one step.
        
        Args:
            features_df: DataFrame with features
            exclude_columns: Columns to exclude from scaling
            
        Returns:
            DataFrame with scaled features
        """
        self.fit_scaler(features_df, exclude_columns)
        return self.transform_features(features_df, exclude_columns)
    
    def save_scaler(self, filepath: Union[str, Path]) -> None:
        """
        Save fitted scaler to disk.
        
        Args:
            filepath: Path to save scaler
        """
        if self.scaler is None:
            raise ValueError("No scaler fitted. Cannot save.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, filepath)
        logger.info(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: Union[str, Path]) -> StandardScaler:
        """
        Load scaler from disk.
        
        Args:
            filepath: Path to scaler file
            
        Returns:
            Loaded StandardScaler
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
        
        self.scaler = joblib.load(filepath)
        logger.info(f"Scaler loaded from {filepath}")
        
        return self.scaler
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names (excluding metadata columns).
        
        Returns:
            List of feature names
        """
        # This represents the standard feature set
        feature_names = [
            'chroma_stft_mean', 'chroma_stft_var',
            'rms_mean', 'rms_var',
            'spectral_centroid_mean', 'spectral_centroid_var',
            'spectral_bandwidth_mean', 'spectral_bandwidth_var',
            'rolloff_mean', 'rolloff_var',
            'zero_crossing_rate_mean', 'zero_crossing_rate_var',
            'harmony_mean', 'harmony_var',
            'perceptr_mean', 'perceptr_var',
            'tempo'
        ]
        
        # Add MFCC features
        for i in range(1, 21):
            feature_names.extend([f'mfcc{i}_mean', f'mfcc{i}_var'])
        
        return feature_names


def extract_features_from_directory(directory: Union[str, Path],
                                    pattern: str = "*.wav",
                                    duration: Optional[float] = None,
                                    recursive: bool = True) -> pd.DataFrame:
    """
    Extract features from all audio files in a directory.
    
    Args:
        directory: Directory containing audio files
        pattern: File pattern to match (e.g., "*.wav", "*.mp3")
        duration: Maximum duration to load per file
        recursive: Whether to search recursively
        
    Returns:
        DataFrame with extracted features
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Find all audio files
    if recursive:
        audio_files = list(directory.rglob(pattern))
    else:
        audio_files = list(directory.glob(pattern))
    
    if not audio_files:
        logger.warning(f"No audio files found in {directory} matching pattern {pattern}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(audio_files)} audio files in {directory}")
    
    # Extract features
    extractor = FeatureExtractor()
    return extractor.extract_features_batch(audio_files, duration=duration)


def main():
    """
    Main function for testing the Feature Extraction module.
    """
    import sys
    
    # Example usage
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        # Use a sample audio file from the dataset
        audio_path = "data/dataset/genres_original/blues/blues.00000.wav"
    
    if not Path(audio_path).exists():
        print(f"Audio file not found: {audio_path}")
        print("Usage: python feature_extraction.py <audio_file_path>")
        return
    
    # Initialize extractor
    extractor = FeatureExtractor()
    
    # Extract features from single file
    print(f"\nExtracting features from: {audio_path}")
    features = extractor.extract_features(audio_path)
    
    print("\n" + "=" * 60)
    print("EXTRACTED FEATURES")
    print("=" * 60)
    for key, value in features.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print(f"Total features extracted: {len(features)}")
    print("=" * 60)
    
    # Test batch extraction
    print("\n\nTesting batch extraction...")
    audio_dir = Path("data/dataset/genres_original/blues")
    if audio_dir.exists():
        audio_files = list(audio_dir.glob("*.wav"))[:5]  # Test with 5 files
        print(f"Extracting features from {len(audio_files)} files...")
        df = extractor.extract_features_batch(audio_files, duration=3.0)  # 3 seconds
        
        print(f"\nBatch extraction completed!")
        print(f"Shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())


if __name__ == "__main__":
    main()

