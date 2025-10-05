"""
Aspect-Based Sentiment Analysis (ABSA) module for board game comments.
"""

import pandas as pd
from typing import List, Dict, Any, Tuple
from .sentiment_analyzer import SentimentAnalyzer
from .utils import ConfigManager


class ABSAAnalyzer:
    """Performs aspect-based sentiment analysis on board game comments."""
    
    def __init__(self, config_manager: ConfigManager, sentiment_analyzer: SentimentAnalyzer):
        self.config = config_manager
        self.logger = config_manager.logger
        self.sentiment_analyzer = sentiment_analyzer
        self.aspects = self.config.get('aspects', {})
        
    def extract_aspects(self, comment: str) -> List[str]:
        """
        Extract relevant aspects from a comment based on keyword matching.
        
        Args:
            comment: Text comment to analyze
            
        Returns:
            List of detected aspects
        """
        if not isinstance(comment, str) or not comment.strip():
            return []
        
        detected_aspects = []
        comment_lower = comment.lower()
        
        # Enhanced keyword matching with context awareness
        for aspect, keywords in self.aspects.items():
            aspect_found = False
            
            # Check for exact keyword matches
            for keyword in keywords:
                if keyword in comment_lower:
                    # Additional context validation for certain aspects
                    if aspect == 'bash_the_leader':
                        # Look for context around "leader" mentions
                        if 'leader' in comment_lower and any(word in comment_lower for word in ['target', 'attack', 'beat', 'stop']):
                            aspect_found = True
                    elif aspect == 'downtime':
                        # Look for context around waiting/idle mentions
                        if any(word in comment_lower for word in ['wait', 'idle', 'turn']) and any(word in comment_lower for word in ['long', 'boring', 'slow']):
                            aspect_found = True
                    else:
                        aspect_found = True
                    break
            
            if aspect_found:
                detected_aspects.append(aspect)
        
        return detected_aspects
    
    def analyze_aspect_sentiment(self, comment: str, aspects: List[str]) -> Dict[str, str]:
        """
        Analyze sentiment for specific aspects in a comment.
        
        Args:
            comment: Text comment to analyze
            aspects: List of aspects to analyze
            
        Returns:
            Dictionary mapping aspects to sentiment labels
        """
        aspect_sentiments = {}
        
        for aspect in aspects:
            try:
                sentiment, _ = self.sentiment_analyzer.analyze_sentiment(comment)
                aspect_sentiments[aspect] = sentiment
            except Exception as e:
                self.logger.warning(f"Error analyzing aspect '{aspect}': {e}")
                aspect_sentiments[aspect] = 'unknown'
        
        return aspect_sentiments
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'cleaned_comment') -> pd.DataFrame:
        """
        Perform ABSA on a DataFrame of comments.
        
        Args:
            df: DataFrame containing text data
            text_column: Name of the column containing text
            
        Returns:
            DataFrame with added ABSA columns
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        self.logger.info(f"Performing ABSA on {len(df)} comments")
        
        # Extract aspects for each comment
        df['detected_aspects'] = df[text_column].apply(self.extract_aspects)
        
        # Filter comments that have detected aspects
        df_with_aspects = df[df['detected_aspects'].map(len) > 0].copy()
        
        self.logger.info(f"Found {len(df_with_aspects)} comments with detectable aspects")
        
        if len(df_with_aspects) == 0:
            self.logger.warning("No comments with detectable aspects found")
            return df
        
        # Analyze aspect sentiments
        def analyze_row(row):
            return self.analyze_aspect_sentiment(row[text_column], row['detected_aspects'])
        
        df_with_aspects['aspect_sentiments'] = df_with_aspects.apply(analyze_row, axis=1)
        
        # Log aspect statistics
        self._log_aspect_statistics(df_with_aspects)
        
        return df_with_aspects
    
    def _log_aspect_statistics(self, df: pd.DataFrame) -> None:
        """Log statistics about detected aspects."""
        # Count aspect occurrences
        aspect_counts = {}
        for aspects in df['detected_aspects']:
            for aspect in aspects:
                aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
        
        self.logger.info("Aspect detection statistics:")
        for aspect, count in sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"  {aspect}: {count} comments")
        
        # Count sentiment distribution for each aspect
        aspect_sentiment_counts = {}
        for _, row in df.iterrows():
            for aspect, sentiment in row['aspect_sentiments'].items():
                if aspect not in aspect_sentiment_counts:
                    aspect_sentiment_counts[aspect] = {'positive': 0, 'negative': 0, 'unknown': 0}
                aspect_sentiment_counts[aspect][sentiment] += 1
        
        self.logger.info("Aspect sentiment distribution:")
        for aspect, sentiments in aspect_sentiment_counts.items():
            total = sum(sentiments.values())
            pos_pct = (sentiments['positive'] / total) * 100 if total > 0 else 0
            neg_pct = (sentiments['negative'] / total) * 100 if total > 0 else 0
            self.logger.info(f"  {aspect}: {pos_pct:.1f}% positive, {neg_pct:.1f}% negative")
    
    def get_aspect_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a summary of aspect sentiment analysis.
        
        Args:
            df: DataFrame with ABSA results
            
        Returns:
            DataFrame with aspect sentiment summary
        """
        aspect_data = []
        
        for aspect in self.aspects.keys():
            # Count comments mentioning this aspect
            aspect_comments = df[df['detected_aspects'].apply(lambda x: aspect in x)]
            
            if len(aspect_comments) == 0:
                continue
            
            # Count sentiment distribution
            sentiment_counts = {'positive': 0, 'negative': 0, 'unknown': 0}
            for _, row in aspect_comments.iterrows():
                if aspect in row['aspect_sentiments']:
                    sentiment = row['aspect_sentiments'][aspect]
                    sentiment_counts[sentiment] += 1
            
            total = sum(sentiment_counts.values())
            if total > 0:
                aspect_data.append({
                    'aspect': aspect,
                    'total_comments': total,
                    'positive_count': sentiment_counts['positive'],
                    'negative_count': sentiment_counts['negative'],
                    'unknown_count': sentiment_counts['unknown'],
                    'positive_percentage': (sentiment_counts['positive'] / total) * 100,
                    'negative_percentage': (sentiment_counts['negative'] / total) * 100
                })
        
        return pd.DataFrame(aspect_data)
    
    def save_results(self, df: pd.DataFrame, output_path: str) -> None:
        """Save ABSA results to CSV file."""
        try:
            df.to_csv(output_path, index=False)
            self.logger.info(f"ABSA results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving ABSA results: {e}")
            raise
