"""
Data quality metrics and reporting for the Board Game NLP project.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import Counter
import re


class QualityMetrics:
    """Calculate and report data quality metrics."""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = config_manager.logger
    
    def calculate_text_quality(self, df: pd.DataFrame, text_column: str = 'cleaned_comment') -> Dict[str, Any]:
        """Calculate text quality metrics."""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        texts = df[text_column].dropna()
        
        metrics = {
            'total_texts': len(texts),
            'avg_length': texts.str.len().mean(),
            'median_length': texts.str.len().median(),
            'min_length': texts.str.len().min(),
            'max_length': texts.str.len().max(),
            'empty_texts': (texts.str.len() == 0).sum(),
            'very_short_texts': (texts.str.len() < 10).sum(),
            'very_long_texts': (texts.str.len() > 1000).sum(),
        }
        
        # Calculate readability metrics
        metrics.update(self._calculate_readability_metrics(texts))
        
        # Calculate language quality
        metrics.update(self._calculate_language_quality(texts))
        
        return metrics
    
    def _calculate_readability_metrics(self, texts: pd.Series) -> Dict[str, float]:
        """Calculate basic readability metrics."""
        metrics = {}
        
        # Average sentence length
        sentence_lengths = []
        for text in texts:
            sentences = re.split(r'[.!?]+', text)
            sentence_lengths.extend([len(s.split()) for s in sentences if s.strip()])
        
        if sentence_lengths:
            metrics['avg_sentence_length'] = np.mean(sentence_lengths)
            metrics['avg_sentences_per_text'] = len(sentence_lengths) / len(texts)
        else:
            metrics['avg_sentence_length'] = 0
            metrics['avg_sentences_per_text'] = 0
        
        return metrics
    
    def _calculate_language_quality(self, texts: pd.Series) -> Dict[str, Any]:
        """Calculate language quality metrics."""
        metrics = {}
        
        # Character distribution
        all_text = ' '.join(texts.astype(str))
        char_counts = Counter(all_text.lower())
        
        # Calculate character diversity
        total_chars = len(all_text)
        unique_chars = len(char_counts)
        metrics['char_diversity'] = unique_chars / total_chars if total_chars > 0 else 0
        
        # Word frequency analysis
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_counts = Counter(words)
        metrics['unique_words'] = len(word_counts)
        metrics['total_words'] = len(words)
        metrics['avg_words_per_text'] = len(words) / len(texts) if len(texts) > 0 else 0
        
        # Most common words
        metrics['top_10_words'] = dict(word_counts.most_common(10))
        
        return metrics
    
    def calculate_sentiment_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate sentiment analysis quality metrics."""
        if 'sentiment' not in df.columns:
            return {}
        
        sentiment_counts = df['sentiment'].value_counts()
        total = len(df)
        
        metrics = {
            'total_analyzed': total,
            'sentiment_distribution': sentiment_counts.to_dict(),
            'sentiment_balance': {
                'positive_pct': (sentiment_counts.get('positive', 0) / total) * 100,
                'negative_pct': (sentiment_counts.get('negative', 0) / total) * 100,
                'unknown_pct': (sentiment_counts.get('unknown', 0) / total) * 100,
            }
        }
        
        # Confidence analysis
        if 'sentiment_confidence' in df.columns:
            confidences = df['sentiment_confidence'].dropna()
            metrics['confidence_stats'] = {
                'avg_confidence': confidences.mean(),
                'median_confidence': confidences.median(),
                'low_confidence_count': (confidences < 0.7).sum(),
                'high_confidence_count': (confidences > 0.9).sum(),
            }
        
        return metrics
    
    def calculate_absa_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate ABSA quality metrics."""
        if 'detected_aspects' not in df.columns:
            return {}
        
        metrics = {
            'total_with_aspects': len(df),
            'aspects_detected': {},
            'sentiment_by_aspect': {},
        }
        
        # Count aspect occurrences
        all_aspects = []
        for aspects in df['detected_aspects']:
            if isinstance(aspects, list):
                all_aspects.extend(aspects)
        
        aspect_counts = Counter(all_aspects)
        metrics['aspects_detected'] = dict(aspect_counts)
        
        # Analyze sentiment by aspect
        if 'aspect_sentiments' in df.columns:
            aspect_sentiments = {}
            for _, row in df.iterrows():
                if isinstance(row['aspect_sentiments'], dict):
                    for aspect, sentiment in row['aspect_sentiments'].items():
                        if aspect not in aspect_sentiments:
                            aspect_sentiments[aspect] = {'positive': 0, 'negative': 0, 'unknown': 0}
                        aspect_sentiments[aspect][sentiment] += 1
            
            metrics['sentiment_by_aspect'] = aspect_sentiments
        
        return metrics
    
    def generate_quality_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive quality report."""
        report = []
        report.append("=" * 60)
        report.append("BOARD GAME NLP DATA QUALITY REPORT")
        report.append("=" * 60)
        
        # Text quality
        text_metrics = self.calculate_text_quality(df)
        report.append("\n📝 TEXT QUALITY METRICS")
        report.append("-" * 30)
        report.append(f"Total texts: {text_metrics['total_texts']:,}")
        report.append(f"Average length: {text_metrics['avg_length']:.1f} characters")
        report.append(f"Median length: {text_metrics['median_length']:.1f} characters")
        report.append(f"Length range: {text_metrics['min_length']} - {text_metrics['max_length']} characters")
        report.append(f"Empty texts: {text_metrics['empty_texts']}")
        report.append(f"Very short texts (<10 chars): {text_metrics['very_short_texts']}")
        report.append(f"Very long texts (>1000 chars): {text_metrics['very_long_texts']}")
        
        if 'avg_sentence_length' in text_metrics:
            report.append(f"Average sentence length: {text_metrics['avg_sentence_length']:.1f} words")
            report.append(f"Average sentences per text: {text_metrics['avg_sentences_per_text']:.1f}")
        
        # Language quality
        if 'char_diversity' in text_metrics:
            report.append(f"\nCharacter diversity: {text_metrics['char_diversity']:.3f}")
            report.append(f"Unique words: {text_metrics['unique_words']:,}")
            report.append(f"Total words: {text_metrics['total_words']:,}")
            report.append(f"Average words per text: {text_metrics['avg_words_per_text']:.1f}")
        
        # Sentiment quality
        sentiment_metrics = self.calculate_sentiment_quality(df)
        if sentiment_metrics:
            report.append("\n😊 SENTIMENT ANALYSIS QUALITY")
            report.append("-" * 30)
            report.append(f"Total analyzed: {sentiment_metrics['total_analyzed']:,}")
            
            balance = sentiment_metrics['sentiment_balance']
            report.append(f"Positive: {balance['positive_pct']:.1f}%")
            report.append(f"Negative: {balance['negative_pct']:.1f}%")
            report.append(f"Unknown: {balance['unknown_pct']:.1f}%")
            
            if 'confidence_stats' in sentiment_metrics:
                conf = sentiment_metrics['confidence_stats']
                report.append(f"Average confidence: {conf['avg_confidence']:.3f}")
                report.append(f"Low confidence (<0.7): {conf['low_confidence_count']}")
                report.append(f"High confidence (>0.9): {conf['high_confidence_count']}")
        
        # ABSA quality
        absa_metrics = self.calculate_absa_quality(df)
        if absa_metrics:
            report.append("\n🎯 ASPECT-BASED SENTIMENT ANALYSIS QUALITY")
            report.append("-" * 30)
            report.append(f"Texts with aspects: {absa_metrics['total_with_aspects']:,}")
            
            if absa_metrics['aspects_detected']:
                report.append("\nAspect detection counts:")
                for aspect, count in sorted(absa_metrics['aspects_detected'].items(), 
                                          key=lambda x: x[1], reverse=True):
                    report.append(f"  {aspect}: {count}")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)
    
    def save_quality_report(self, df: pd.DataFrame, output_path: str) -> None:
        """Save quality report to file."""
        report = self.generate_quality_report(df)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Quality report saved to {output_path}")
    
    def get_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate an overall data quality score (0-100)."""
        score = 100.0
        
        # Penalize for empty texts
        if 'cleaned_comment' in df.columns:
            empty_ratio = (df['cleaned_comment'].str.len() == 0).sum() / len(df)
            score -= empty_ratio * 20
        
        # Penalize for very short texts
        if 'cleaned_comment' in df.columns:
            short_ratio = (df['cleaned_comment'].str.len() < 10).sum() / len(df)
            score -= short_ratio * 10
        
        # Penalize for low confidence sentiment
        if 'sentiment_confidence' in df.columns:
            low_conf_ratio = (df['sentiment_confidence'] < 0.7).sum() / len(df)
            score -= low_conf_ratio * 15
        
        # Bonus for good aspect detection
        if 'detected_aspects' in df.columns:
            aspect_ratio = (df['detected_aspects'].str.len() > 0).sum() / len(df)
            score += aspect_ratio * 5
        
        return max(0, min(100, score))
