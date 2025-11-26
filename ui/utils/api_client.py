"""
API client utility for communicating with FastAPI backend
"""
import requests
from typing import Optional, Dict, Any, List
import streamlit as st


class APIClient:
    """Client for interacting with the FastAPI backend."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the FastAPI server
        """
        self.base_url = base_url.rstrip('/')
        self.api_base = f"{self.base_url}/api/v1"
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request to API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests
            
        Returns:
            Response JSON or None if error
        """
        url = f"{self.api_base}{endpoint}"
        
        try:
            response = requests.request(method, url, **kwargs, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    st.error(f"Details: {error_data.get('detail', 'Unknown error')}")
                except:
                    st.error(f"Status: {e.response.status_code}")
            return None
    
    def health_check(self) -> Optional[Dict[str, Any]]:
        """Check API health."""
        return self._make_request("GET", "/health")
    
    def get_model_status(self) -> Optional[Dict[str, Any]]:
        """Get model status."""
        return self._make_request("GET", "/model/status")
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get model performance metrics."""
        return self._make_request("GET", "/metrics")
    
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get API usage statistics."""
        return self._make_request("GET", "/stats")
    
    def predict(self, audio_file: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """
        Make prediction from audio file.
        
        Args:
            audio_file: Audio file bytes
            filename: Original filename
            
        Returns:
            Prediction result
        """
        files = {"file": (filename, audio_file, "audio/wav")}
        return self._make_request("POST", "/predict", files=files)
    
    def predict_batch(self, audio_files: List[tuple]) -> Optional[Dict[str, Any]]:
        """
        Make batch predictions.
        
        Args:
            audio_files: List of (filename, bytes) tuples
            
        Returns:
            Batch prediction results
        """
        files = [("files", (name, data, "audio/wav")) for name, data in audio_files]
        return self._make_request("POST", "/predict/batch", files=files)
    
    def upload_data(self, audio_files: List[tuple]) -> Optional[Dict[str, Any]]:
        """
        Upload training data.
        
        Args:
            audio_files: List of (filename, bytes) tuples
            
        Returns:
            Upload result
        """
        files = [("files", (name, data, "audio/wav")) for name, data in audio_files]
        return self._make_request("POST", "/data/upload", files=files)
    
    def train_model(
        self, 
        model_type: str = "random_forest",
        test_size: float = 0.2,
        tune_hyperparameters: bool = False,
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Start training a new model.
        
        Args:
            model_type: Type of model to train
            test_size: Test set proportion
            tune_hyperparameters: Whether to tune hyperparameters
            version: Model version name
            
        Returns:
            Training job response
        """
        data = {
            "model_type": model_type,
            "test_size": test_size,
            "tune_hyperparameters": tune_hyperparameters
        }
        if version:
            data["version"] = version
        
        return self._make_request("POST", "/train", json=data)
    
    def retrain_model(
        self,
        model_version: Optional[str] = None,
        combine_strategy: str = "append",
        use_existing_data: bool = True,
        new_version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Start retraining an existing model.
        
        Args:
            model_version: Version to retrain (None = latest)
            combine_strategy: How to combine datasets
            use_existing_data: Whether to use existing training data
            new_version: Version name for new model
            
        Returns:
            Retraining job response
        """
        data = {
            "combine_strategy": combine_strategy,
            "use_existing_data": use_existing_data
        }
        if model_version:
            data["model_version"] = model_version
        if new_version:
            data["new_version"] = new_version
        
        return self._make_request("POST", "/retrain", json=data)
    
    def get_training_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get training job status.
        
        Args:
            job_id: Training job ID
            
        Returns:
            Training status
        """
        return self._make_request("GET", f"/training/status/{job_id}")
    
    def get_all_training_status(self) -> Optional[Dict[str, Any]]:
        """Get all training jobs status."""
        return self._make_request("GET", "/training/status")


# Singleton instance
@st.cache_resource
def get_api_client(base_url: str = "http://localhost:8000") -> APIClient:
    """
    Get cached API client instance.
    
    Args:
        base_url: Base URL of the FastAPI server
        
    Returns:
        APIClient instance
    """
    return APIClient(base_url)

