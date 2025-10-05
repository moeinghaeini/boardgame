"""
Board Game NLP Analysis Package

A comprehensive NLP toolkit for analyzing board game reviews and comments
using sentiment analysis and aspect-based sentiment analysis (ABSA).
"""

__version__ = "1.0.0"
__author__ = "Board Game NLP Team"

from .data_collector import DataCollector
from .sentiment_analyzer import SentimentAnalyzer
from .absa_analyzer import ABSAAnalyzer
from .model_trainer import ModelTrainer
from .quality_metrics import QualityMetrics
from .cache import CacheManager, SentimentCache

__all__ = [
    "DataCollector",
    "SentimentAnalyzer", 
    "ABSAAnalyzer",
    "ModelTrainer",
    "QualityMetrics",
    "CacheManager",
    "SentimentCache"
]
