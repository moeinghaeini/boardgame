"""
Data collection module for Board Game Geek API.
"""

import os
import time
import math
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from pathlib import Path

from .utils import ConfigManager, clean_text, is_english


class DataCollector:
    """Collects board game data from Board Game Geek API."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = config_manager.logger
        self.base_url = self.config.get('api.base_url')
        self.rate_limit_delay = self.config.get('api.rate_limit_delay', 1.5)
        self.max_retries = self.config.get('api.max_retries', 3)
        self.timeout = self.config.get('api.timeout', 30)
        
    def extract_zip(self, zip_file_path: str, extract_dir: str) -> None:
        """Extract zip file to directory."""
        import zipfile
        
        try:
            Path(extract_dir).mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            self.logger.info(f"Extracted files to {extract_dir}")
        except Exception as e:
            self.logger.error(f"Error extracting zip file: {e}")
            raise
    
    def download_page_comments(self, url: str, game_id: int) -> List[Dict]:
        """Download comments from a single page."""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                root = ET.fromstring(response.content)
                
                page_comments = []
                for comment in root.iter('comment'):
                    comment_data = comment.attrib
                    comment_data['boardgame_id'] = game_id
                    page_comments.append(comment_data)
                
                if page_comments:
                    return page_comments
                else:
                    self.logger.warning(f"No comments received for page, attempt {attempt + 1}")
                    time.sleep(1)
                    
            except requests.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        return []
    
    def download_game_comments(self, game_id: int) -> pd.DataFrame:
        """Download all comments for a specific game."""
        if not isinstance(game_id, int) or game_id <= 0:
            raise ValueError(f"Invalid game_id: {game_id}")
        
        comments = []
        base_url = f'{self.base_url}?type=boardgame&id={game_id}&comments=1'
        
        self.logger.info(f'Downloading comments for game with id {game_id}')
        
        try:
            # Get total number of comments
            response = requests.get(base_url, timeout=self.timeout)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            
            comments_elem = root[0].find('comments')
            if comments_elem is None:
                self.logger.warning(f"No comments found for game {game_id}")
                return pd.DataFrame()
            
            total_comments = int(comments_elem.attrib.get('totalitems', 0))
            if total_comments == 0:
                self.logger.warning(f"No comments available for game {game_id}")
                return pd.DataFrame()
            
            comments_per_page = self.config.get('analysis.comments_per_page', 100)
            total_pages = math.ceil(total_comments / comments_per_page)
            
            self.logger.info(f"Found {total_comments} comments across {total_pages} pages")
            time.sleep(self.rate_limit_delay)
            
            # Download comments from all pages
            for page in range(1, total_pages + 1):
                url = f'{base_url}&page={page}'
                page_comments = self.download_page_comments(url, game_id)
                comments.extend(page_comments)
                
                self.logger.info(f"Downloaded {len(comments)}/{total_comments} comments")
                time.sleep(self.rate_limit_delay)
                
        except requests.RequestException as e:
            self.logger.error(f"Network error downloading comments for game {game_id}: {e}")
            raise
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error for game {game_id}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error downloading comments for game {game_id}: {e}")
            raise
        
        return pd.DataFrame(comments)
    
    def process_comments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean comments data."""
        if df.empty:
            self.logger.warning("Empty DataFrame provided for processing")
            return df
        
        self.logger.info("Processing and cleaning comments data")
        
        # Validate required columns
        required_columns = ['value']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Drop rows with missing comments or very short comments
        initial_count = len(df)
        df = df.dropna(subset=['value'])
        df = df[df['value'].notna()]
        
        min_length = self.config.get('analysis.min_comment_length', 10)
        df = df[df['value'].str.len() > min_length]
        
        self.logger.info(f"Filtered from {initial_count} to {len(df)} comments after length filtering")
        
        # Clean comments
        df['cleaned_comment'] = df['value'].apply(clean_text)
        
        # Remove empty cleaned comments
        df = df[df['cleaned_comment'].str.len() > 0]
        
        # Convert rating to numeric if present
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Filter for English comments
        self.logger.info("Detecting English comments...")
        df['is_english'] = df['cleaned_comment'].apply(is_english)
        df_english = df[df['is_english']].copy()
        df_english = df_english.drop(columns=['is_english'])
        
        self.logger.info(f"Filtered to {len(df_english)} English comments from {len(df)} total")
        
        # Validate final dataset
        if df_english.empty:
            self.logger.warning("No English comments found after processing")
        else:
            self.logger.info(f"Final dataset contains {len(df_english)} comments")
            self.logger.info(f"Average comment length: {df_english['cleaned_comment'].str.len().mean():.1f} characters")
        
        return df_english
    
    def collect_data(self, rankings_file: Optional[str] = None) -> pd.DataFrame:
        """Main method to collect board game data."""
        if rankings_file is None:
            rankings_file = self.config.get_data_path('boardgames_ranks.csv')
        
        comments_file = self.config.get_data_path('boardgames_comments.csv')
        english_comments_file = self.config.get_data_path('english_boardgames_comments.csv')
        
        # Check if comments already exist
        if os.path.isfile(english_comments_file):
            self.logger.info(f"Loading existing English comments from {english_comments_file}")
            return pd.read_csv(english_comments_file)
        
        # Load top games from rankings
        if not os.path.isfile(rankings_file):
            raise FileNotFoundError(f"Rankings file not found: {rankings_file}")
        
        df_bgs = pd.read_csv(rankings_file)
        top_games_count = self.config.get('analysis.top_games_count', 10)
        df_bgs_top = df_bgs.head(top_games_count)
        
        self.logger.info(f"Processing top {top_games_count} games")
        
        # Download comments for each game
        all_comments = []
        for _, game in df_bgs_top.iterrows():
            game_id = game['id']
            try:
                game_comments = self.download_game_comments(game_id)
                all_comments.append(game_comments)
            except Exception as e:
                self.logger.error(f"Failed to download comments for game {game_id}: {e}")
                continue
        
        if not all_comments:
            raise RuntimeError("No comments were successfully downloaded")
        
        # Combine all comments
        df_comments = pd.concat(all_comments, ignore_index=True)
        
        # Save raw comments
        df_comments.to_csv(comments_file, index=False)
        self.logger.info(f"Raw comments saved to {comments_file}")
        
        # Process and save English comments
        df_english = self.process_comments(df_comments)
        df_english.to_csv(english_comments_file, index=False)
        self.logger.info(f"English comments saved to {english_comments_file}")
        
        return df_english
