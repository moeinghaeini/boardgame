"""
Sentiment analysis module for board game comments.
"""

import torch
import pandas as pd
from typing import Tuple, Dict, Any
from transformers import AlbertTokenizer, AlbertForSequenceClassification

from .utils import ConfigManager
from .cache import CacheManager, SentimentCache


class SentimentAnalyzer:
    """Performs sentiment analysis on board game comments."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = config_manager.logger
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize caching if enabled
        self.cache_enabled = self.config.get('analysis.enable_caching', True)
        if self.cache_enabled:
            self.cache_manager = CacheManager()
            self.sentiment_cache = SentimentCache(self.cache_manager)
        else:
            self.sentiment_cache = None
        
    def load_model(self, model_path: str = None) -> None:
        """Load the pre-trained ALBERT model and tokenizer."""
        if model_path is None:
            model_path = self.config.get_model_path()
        
        try:
            self.logger.info(f"Loading model from {model_path}")
            self.model = AlbertForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AlbertTokenizer.from_pretrained(model_path)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment of a single text.
        
        Returns:
            Tuple of (sentiment_label, confidence_score)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Check cache first
        if self.sentiment_cache:
            cached_result = self.sentiment_cache.get_sentiment(text)
            if cached_result is not None:
                return cached_result
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=self.config.get('model.max_length', 512),
                return_tensors='pt'
            )
            
            # Move inputs to device
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).item()
                confidence = torch.softmax(logits, dim=1).max().item()
            
            # Map to labels
            labels = {0: 'negative', 1: 'positive'}
            sentiment = labels[predicted_class]
            
            # Cache the result
            if self.sentiment_cache:
                self.sentiment_cache.set_sentiment(text, sentiment, confidence)
            
            return sentiment, confidence
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return 'unknown', 0.0
    
    def analyze_batch(self, texts: list) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts with optimized processing.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            DataFrame with sentiment analysis results
        """
        if not texts:
            return pd.DataFrame()
        
        results = []
        batch_size = self.config.get('model.batch_size', 16)
        
        self.logger.info(f"Analyzing sentiment for {len(texts)} texts in batches of {batch_size}")
        
        # Process in batches for better performance
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.get('model.max_length', 512),
                    return_tensors='pt'
                )
                
                # Move to device
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=1)
                    confidences = torch.softmax(logits, dim=1).max(dim=1)[0]
                
                # Process results
                labels = {0: 'negative', 1: 'positive'}
                for j, (text, pred, conf) in enumerate(zip(batch_texts, predictions, confidences)):
                    results.append({
                        'text': text,
                        'sentiment': labels[pred.item()],
                        'confidence': conf.item()
                    })
                
                if (i + batch_size) % (batch_size * 10) == 0:
                    self.logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
                    
            except Exception as e:
                self.logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Fallback to individual processing for this batch
                for text in batch_texts:
                    sentiment, confidence = self.analyze_sentiment(text)
                    results.append({
                        'text': text,
                        'sentiment': sentiment,
                        'confidence': confidence
                    })
        
        return pd.DataFrame(results)
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'cleaned_comment') -> pd.DataFrame:
        """
        Analyze sentiment for a DataFrame.
        
        Args:
            df: DataFrame containing text data
            text_column: Name of the column containing text
            
        Returns:
            DataFrame with added sentiment columns
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        self.logger.info(f"Analyzing sentiment for {len(df)} comments")
        
        # Apply sentiment analysis
        sentiment_results = df[text_column].apply(
            lambda x: pd.Series(self.analyze_sentiment(x))
        )
        
        # Add results to dataframe
        df_result = df.copy()
        df_result['sentiment'] = sentiment_results[0]
        df_result['sentiment_confidence'] = sentiment_results[1]
        
        # Log results summary
        sentiment_counts = df_result['sentiment'].value_counts()
        self.logger.info(f"Sentiment analysis complete:")
        self.logger.info(f"  Positive: {sentiment_counts.get('positive', 0)}")
        self.logger.info(f"  Negative: {sentiment_counts.get('negative', 0)}")
        self.logger.info(f"  Unknown: {sentiment_counts.get('unknown', 0)}")
        
        return df_result
    
    def save_results(self, df: pd.DataFrame, output_path: str) -> None:
        """Save analysis results to CSV file."""
        try:
            df.to_csv(output_path, index=False)
            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise
