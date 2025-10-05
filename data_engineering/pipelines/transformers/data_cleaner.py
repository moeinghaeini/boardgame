"""
Data Cleaning and Transformation Pipeline

Handles data cleaning, preprocessing, and transformation for board game comments
with advanced text processing and quality validation.
"""

import pandas as pd
import numpy as np
import re
import string
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import langdetect
from langdetect import DetectorFactory

# Set seed for consistent language detection
DetectorFactory.seed = 0

@dataclass
class CleaningConfig:
    """Configuration for data cleaning."""
    min_comment_length: int = 10
    max_comment_length: int = 5000
    remove_stopwords: bool = True
    lemmatize: bool = True
    remove_punctuation: bool = False
    language_threshold: float = 0.8
    supported_languages: List[str] = None
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ['en']

class DataCleaner:
    """Cleans and transforms board game comment data."""
    
    def __init__(self, config: CleaningConfig = None):
        self.config = config or CleaningConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK components
        self._setup_nltk()
        
        # Initialize text processing tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def _setup_nltk(self):
        """Setup NLTK resources."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def clean_data(self, input_path: str, output_path: str) -> str:
        """
        Clean and transform data from input to output file.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to output CSV file
            
        Returns:
            Path to cleaned data file
        """
        self.logger.info(f"Starting data cleaning from {input_path}")
        
        # Load data
        df = pd.read_csv(input_path)
        self.logger.info(f"Loaded {len(df)} comments")
        
        # Clean data
        df_cleaned = self._clean_dataframe(df)
        
        # Save cleaned data
        df_cleaned.to_csv(output_path, index=False)
        
        self.logger.info(f"Data cleaning completed. Saved {len(df_cleaned)} comments to {output_path}")
        
        return output_path
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and transform DataFrame."""
        original_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['boardgame_id', 'cleaned_comment'])
        self.logger.info(f"Removed {original_count - len(df)} duplicate comments")
        
        # Clean comment text
        df['cleaned_comment'] = df['cleaned_comment'].apply(self._clean_text)
        
        # Filter by comment length
        df = df[df['cleaned_comment'].str.len() >= self.config.min_comment_length]
        df = df[df['cleaned_comment'].str.len() <= self.config.max_comment_length]
        
        # Language detection and filtering
        df = self._filter_by_language(df)
        
        # Remove empty comments
        df = df[df['cleaned_comment'].str.strip() != '']
        
        # Clean usernames
        df['username'] = df['username'].fillna('anonymous')
        df['username'] = df['username'].str.strip()
        
        # Clean ratings
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['rating'] = df['rating'].fillna(0)
        
        # Add metadata
        df['cleaned_at'] = datetime.now().isoformat()
        df['comment_length'] = df['cleaned_comment'].str.len()
        
        self.logger.info(f"Final dataset: {len(df)} comments (removed {original_count - len(df)})")
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """Clean individual text comment."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        
        # Remove BGG-specific formatting
        text = re.sub(r'\[IMG\].*?\[/IMG\]', '', text, flags=re.DOTALL)
        text = re.sub(r'\[URL\].*?\[/URL\]', '', text, flags=re.DOTALL)
        text = re.sub(r'\[QUOTE\].*?\[/QUOTE\]', '', text, flags=re.DOTALL)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep punctuation if configured
        if self.config.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        else:
            # Keep only common punctuation
            text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize and process
        if self.config.remove_stopwords or self.config.lemmatize:
            tokens = word_tokenize(text)
            
            if self.config.remove_stopwords:
                tokens = [token for token in tokens if token not in self.stop_words]
            
            if self.config.lemmatize:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            text = ' '.join(tokens)
        
        return text.strip()
    
    def _filter_by_language(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter comments by language detection."""
        if not self.config.supported_languages:
            return df
        
        self.logger.info("Detecting languages...")
        
        def detect_language(text: str) -> str:
            try:
                if len(text.strip()) < 10:
                    return 'unknown'
                return langdetect.detect(text)
            except:
                return 'unknown'
        
        df['detected_language'] = df['cleaned_comment'].apply(detect_language)
        
        # Filter by supported languages
        original_count = len(df)
        df = df[df['detected_language'].isin(self.config.supported_languages)]
        
        self.logger.info(f"Language filtering: {len(df)}/{original_count} comments in supported languages")
        
        return df
    
    def validate_cleaned_data(self, data_path: str) -> Dict:
        """Validate cleaned data quality."""
        try:
            df = pd.read_csv(data_path)
            
            validation_result = {
                'total_comments': len(df),
                'avg_comment_length': df['cleaned_comment'].str.len().mean(),
                'min_comment_length': df['cleaned_comment'].str.len().min(),
                'max_comment_length': df['cleaned_comment'].str.len().max(),
                'empty_comments': df['cleaned_comment'].str.strip().eq('').sum(),
                'unique_games': df['boardgame_id'].nunique(),
                'language_distribution': df['detected_language'].value_counts().to_dict(),
                'has_metadata': 'cleaned_at' in df.columns and 'comment_length' in df.columns,
                'is_valid': True
            }
            
            # Quality checks
            if validation_result['empty_comments'] > 0:
                validation_result['warnings'] = [f"{validation_result['empty_comments']} empty comments found"]
            
            if validation_result['avg_comment_length'] < 20:
                validation_result['warnings'] = validation_result.get('warnings', [])
                validation_result['warnings'].append("Average comment length is very short")
            
            return validation_result
            
        except Exception as e:
            return {
                'is_valid': False,
                'errors': [f'Failed to validate cleaned data: {e}']
            }
    
    def get_cleaning_report(self, original_path: str, cleaned_path: str) -> Dict:
        """Generate comprehensive cleaning report."""
        try:
            original_df = pd.read_csv(original_path)
            cleaned_df = pd.read_csv(cleaned_path)
            
            report = {
                'original_count': len(original_df),
                'cleaned_count': len(cleaned_df),
                'removed_count': len(original_df) - len(cleaned_df),
                'removal_rate': (len(original_df) - len(cleaned_df)) / len(original_df),
                'avg_length_before': original_df['cleaned_comment'].str.len().mean(),
                'avg_length_after': cleaned_df['cleaned_comment'].str.len().mean(),
                'language_distribution': cleaned_df['detected_language'].value_counts().to_dict(),
                'quality_metrics': self.validate_cleaned_data(cleaned_path)
            }
            
            return report
            
        except Exception as e:
            return {
                'error': f'Failed to generate cleaning report: {e}'
            }
