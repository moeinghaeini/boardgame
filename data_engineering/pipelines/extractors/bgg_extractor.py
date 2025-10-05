"""
BGG Data Extractor

Extracts board game data from BoardGameGeek API with robust error handling,
rate limiting, and data validation.
"""

import os
import time
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ExtractionConfig:
    """Configuration for data extraction."""
    base_url: str = "https://api.geekdo.com/xmlapi2/thing"
    rate_limit_delay: float = 1.5
    max_retries: int = 3
    timeout: int = 30
    top_games_count: int = 10
    max_comments_per_game: int = 1000
    comments_per_page: int = 100

class BGGDataExtractor:
    """Extracts board game data from BGG API."""
    
    def __init__(self, config: ExtractionConfig = None):
        self.config = config or ExtractionConfig()
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BoardGame-NLP-Analysis/1.0'
        })
        
    def extract_data(self, top_games_count: int = None, max_comments_per_game: int = None) -> str:
        """
        Extract complete dataset from BGG API.
        
        Args:
            top_games_count: Number of top games to extract
            max_comments_per_game: Maximum comments per game
            
        Returns:
            Path to extracted data file
        """
        top_games_count = top_games_count or self.config.top_games_count
        max_comments_per_game = max_comments_per_game or self.config.max_comments_per_game
        
        self.logger.info(f"Starting data extraction for {top_games_count} games")
        
        # Get top games
        top_games = self._get_top_games(top_games_count)
        self.logger.info(f"Found {len(top_games)} top games")
        
        # Extract comments for each game
        all_comments = []
        for i, game in enumerate(top_games, 1):
            self.logger.info(f"Extracting comments for game {i}/{len(top_games)}: {game['name']}")
            
            try:
                game_comments = self._extract_game_comments(
                    game['id'], 
                    max_comments_per_game
                )
                all_comments.extend(game_comments)
                self.logger.info(f"Extracted {len(game_comments)} comments for {game['name']}")
                
            except Exception as e:
                self.logger.error(f"Failed to extract comments for {game['name']}: {e}")
                continue
                
            # Rate limiting
            time.sleep(self.config.rate_limit_delay)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_comments)
        output_path = f"/tmp/bgg_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Extraction completed. Total comments: {len(all_comments)}")
        self.logger.info(f"Data saved to: {output_path}")
        
        return output_path
    
    def _get_top_games(self, count: int) -> List[Dict]:
        """Get top ranked games from BGG."""
        url = f"{self.config.base_url}?type=boardgame&stats=1&rank=1"
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(url, timeout=self.config.timeout)
                response.raise_for_status()
                
                root = ET.fromstring(response.content)
                games = []
                
                for item in root.findall('item'):
                    if len(games) >= count:
                        break
                        
                    game_data = {
                        'id': item.get('id'),
                        'name': item.find('name').get('value') if item.find('name') is not None else 'Unknown',
                        'rank': item.find('statistics/ratings/ranks/rank').get('value') if item.find('statistics/ratings/ranks/rank') is not None else '0'
                    }
                    games.append(game_data)
                
                return games
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
    
    def _extract_game_comments(self, game_id: str, max_comments: int) -> List[Dict]:
        """Extract comments for a specific game."""
        comments = []
        page = 1
        
        while len(comments) < max_comments:
            url = f"{self.config.base_url}?type=boardgame&id={game_id}&comments=1&page={page}"
            
            try:
                response = self.session.get(url, timeout=self.config.timeout)
                response.raise_for_status()
                
                root = ET.fromstring(response.content)
                page_comments = self._parse_comments(root, game_id)
                
                if not page_comments:
                    break
                    
                comments.extend(page_comments)
                page += 1
                
                # Rate limiting
                time.sleep(self.config.rate_limit_delay)
                
            except Exception as e:
                self.logger.error(f"Failed to extract page {page} for game {game_id}: {e}")
                break
        
        return comments[:max_comments]
    
    def _parse_comments(self, root: ET.Element, game_id: str) -> List[Dict]:
        """Parse comments from XML response."""
        comments = []
        
        for comment in root.findall('item/comment'):
            try:
                comment_data = {
                    'boardgame_id': game_id,
                    'username': comment.get('username', ''),
                    'rating': comment.get('rating', ''),
                    'value': comment.get('value', ''),
                    'cleaned_comment': self._clean_comment_text(comment.text or '')
                }
                
                # Only include comments with actual text
                if comment_data['cleaned_comment'].strip():
                    comments.append(comment_data)
                    
            except Exception as e:
                self.logger.warning(f"Failed to parse comment: {e}")
                continue
        
        return comments
    
    def _clean_comment_text(self, text: str) -> str:
        """Clean and preprocess comment text."""
        if not text:
            return ""
        
        # Basic text cleaning
        text = text.strip()
        text = ' '.join(text.split())  # Remove extra whitespace
        
        # Remove BGG-specific formatting
        text = text.replace('[IMG]', '').replace('[/IMG]', '')
        text = text.replace('[URL]', '').replace('[/URL]', '')
        
        return text
    
    def validate_extraction(self, data_path: str) -> Dict:
        """Validate extracted data quality."""
        try:
            df = pd.read_csv(data_path)
            
            validation_result = {
                'total_comments': len(df),
                'unique_games': df['boardgame_id'].nunique(),
                'empty_comments': df['cleaned_comment'].isna().sum(),
                'avg_comment_length': df['cleaned_comment'].str.len().mean(),
                'has_required_columns': all(col in df.columns for col in 
                    ['boardgame_id', 'username', 'rating', 'value', 'cleaned_comment']),
                'is_valid': True
            }
            
            # Check for minimum data quality
            if validation_result['total_comments'] < 100:
                validation_result['is_valid'] = False
                validation_result['errors'] = ['Insufficient data extracted']
            
            if validation_result['empty_comments'] > validation_result['total_comments'] * 0.5:
                validation_result['is_valid'] = False
                validation_result['errors'] = ['Too many empty comments']
            
            return validation_result
            
        except Exception as e:
            return {
                'is_valid': False,
                'errors': [f'Failed to validate data: {e}']
            }
