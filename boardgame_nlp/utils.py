"""
Utility functions for the Board Game NLP project.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config.get('level', 'INFO')),
        format=config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(config.get('file', 'boardgame_analysis.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file: {e}")


def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Clean text by removing URLs, brackets, and extra whitespace."""
    import re
    
    if not isinstance(text, str):
        return ""
    
    if not text.strip():
        return ""
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|img\S+", '', text)
    # Remove content in brackets
    text = re.sub(r"\[.*?\]", '', text)
    # Remove extra whitespace
    text = re.sub(r"\s+", ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^\w\s.,!?;:-]", '', text)
    return text.strip()


def is_english(text: str) -> bool:
    """Check if text is in English using langdetect."""
    if not isinstance(text, str) or not text.strip():
        return False
    
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        return detect(text) == 'en'
    except Exception:
        # Fallback: check for common English words
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        text_lower = text.lower()
        return any(word in text_lower for word in english_words)


class ConfigManager:
    """Configuration manager for the project."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config.get('logging', {}))
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_data_path(self, filename: str) -> str:
        """Get full path for data file."""
        base_path = self.get('data.base_path', './data')
        return os.path.join(base_path, filename)
    
    def get_model_path(self) -> str:
        """Get model directory path."""
        return self.get('model.model_path', './Model')
    
    def get_results_path(self) -> str:
        """Get results directory path."""
        return self.get('data.results_path', './Results')
