"""
Audio utility functions for Streamlit UI
"""
import io
import wave
import numpy as np
from typing import Optional, Tuple


def validate_audio_file(file_bytes: bytes, filename: str) -> Tuple[bool, Optional[str]]:
    """
    Validate audio file format.
    
    Args:
        file_bytes: File bytes
        filename: Original filename
        
    Returns:
        (is_valid, error_message)
    """
    allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if not file_ext or f'.{file_ext}' not in allowed_extensions:
        return False, f"Invalid file format. Allowed: {', '.join(allowed_extensions)}"
    
    # Check file size (max 50MB)
    max_size = 50 * 1024 * 1024  # 50MB
    if len(file_bytes) > max_size:
        return False, f"File too large. Maximum size: 50MB"
    
    return True, None


def get_audio_info(file_bytes: bytes, filename: str) -> dict:
    """
    Get basic audio file information.
    
    Args:
        file_bytes: Audio file bytes
        filename: Original filename
        
    Returns:
        Dictionary with audio info
    """
    info = {
        'filename': filename,
        'size_bytes': len(file_bytes),
        'size_mb': round(len(file_bytes) / (1024 * 1024), 2)
    }
    
    # Try to get WAV file info
    if filename.lower().endswith('.wav'):
        try:
            with wave.open(io.BytesIO(file_bytes)) as wav_file:
                info['channels'] = wav_file.getnchannels()
                info['sample_width'] = wav_file.getsampwidth()
                info['framerate'] = wav_file.getframerate()
                info['frames'] = wav_file.getnframes()
                info['duration_seconds'] = round(wav_file.getnframes() / wav_file.getframerate(), 2)
        except Exception:
            pass
    
    return info


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

