"""
Caching utilities for the Board Game NLP project.
"""

import pickle
import hashlib
import os
from typing import Any, Optional
from pathlib import Path


class CacheManager:
    """Manages caching for expensive operations."""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        # Create a hash of the key for filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                # If cache file is corrupted, remove it
                cache_path.unlink(missing_ok=True)
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value."""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception:
            # If we can't write to cache, just continue without caching
            pass
    
    def clear(self) -> None:
        """Clear all cache files."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink(missing_ok=True)
    
    def get_or_set(self, key: str, func, *args, **kwargs) -> Any:
        """Get cached value or compute and cache it."""
        cached_value = self.get(key)
        if cached_value is not None:
            return cached_value
        
        # Compute the value
        value = func(*args, **kwargs)
        self.set(key, value)
        return value


class SentimentCache:
    """Specialized cache for sentiment analysis results."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
    
    def get_sentiment(self, text: str) -> Optional[tuple]:
        """Get cached sentiment result."""
        key = f"sentiment_{hashlib.md5(text.encode()).hexdigest()}"
        return self.cache.get(key)
    
    def set_sentiment(self, text: str, sentiment: str, confidence: float) -> None:
        """Cache sentiment result."""
        key = f"sentiment_{hashlib.md5(text.encode()).hexdigest()}"
        self.cache.set(key, (sentiment, confidence))
    
    def get_absa(self, text: str, aspects: list) -> Optional[dict]:
        """Get cached ABSA result."""
        aspects_key = "_".join(sorted(aspects))
        key = f"absa_{hashlib.md5(f'{text}_{aspects_key}'.encode()).hexdigest()}"
        return self.cache.get(key)
    
    def set_absa(self, text: str, aspects: list, result: dict) -> None:
        """Cache ABSA result."""
        aspects_key = "_".join(sorted(aspects))
        key = f"absa_{hashlib.md5(f'{text}_{aspects_key}'.encode()).hexdigest()}"
        self.cache.set(key, result)
