"""
Locust load testing script for Sound Genre AI API
"""
import os
import random
from pathlib import Path
from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser


class SoundGenreAPIUser(FastHttpUser):
    """
    Simulates a user making requests to the Sound Genre AI API.
    """
    wait_time = between(1, 3)  # Wait between 1 and 3 seconds between requests
    
    def on_start(self):
        """Called when a simulated user starts."""
        # Load a sample audio file for testing
        self.sample_audio_path = self._find_sample_audio()
        if not self.sample_audio_path:
            print("Warning: No sample audio file found. Some tests may fail.")
    
    def _find_sample_audio(self):
        """Find a sample audio file from the dataset."""
        dataset_path = Path("data/dataset/genres_original")
        if not dataset_path.exists():
            # Try alternative paths
            dataset_path = Path("../data/dataset/genres_original")
        
        if dataset_path.exists():
            # Find any .wav file in any genre folder
            for genre_dir in dataset_path.iterdir():
                if genre_dir.is_dir():
                    wav_files = list(genre_dir.glob("*.wav"))
                    if wav_files:
                        return str(wav_files[0])
        return None
    
    @task(5)
    def health_check(self):
        """Health check endpoint - most common request."""
        with self.client.get("/api/v1/health", catch_response=True, name="Health Check") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Expected 200, got {response.status_code}")
    
    @task(3)
    def model_status(self):
        """Get model status."""
        with self.client.get("/api/v1/model/status", catch_response=True, name="Model Status") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Expected 200, got {response.status_code}")
    
    @task(10)
    def predict_genre(self):
        """Predict genre from audio file - main functionality."""
        if not self.sample_audio_path or not os.path.exists(self.sample_audio_path):
            return
        
        try:
            with open(self.sample_audio_path, 'rb') as f:
                files = {'file': (os.path.basename(self.sample_audio_path), f, 'audio/wav')}
                with self.client.post(
                    "/api/v1/predict",
                    files=files,
                    catch_response=True,
                    name="Predict Genre"
                ) as response:
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            if 'genre' in data and 'confidence' in data:
                                response.success()
                            else:
                                response.failure("Invalid response format")
                        except Exception:
                            response.failure("Failed to parse JSON response")
                    else:
                        response.failure(f"Expected 200, got {response.status_code}")
        except Exception as e:
            # If exception occurs before response context, log it
            self.environment.events.request.fire(
                request_type="POST",
                name="Predict Genre",
                response_time=0,
                response_length=0,
                exception=e
            )
    
    @task(2)
    def get_metrics(self):
        """Get model metrics."""
        with self.client.get("/api/v1/metrics", catch_response=True, name="Get Metrics") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Expected 200, got {response.status_code}")
    
    @task(1)
    def get_stats(self):
        """Get API statistics."""
        with self.client.get("/api/v1/stats", catch_response=True, name="Get Stats") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Expected 200, got {response.status_code}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts."""
    print("=" * 60)
    print("Starting Load Test for Sound Genre AI API")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops."""
    print("=" * 60)
    print("Load Test Completed")
    print("=" * 60)

