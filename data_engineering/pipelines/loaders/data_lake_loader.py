"""
Data Lake Loader

Handles loading data to different layers of the data lake architecture
with proper partitioning, versioning, and metadata management.
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import json
from dataclasses import dataclass, asdict

@dataclass
class DataLakeConfig:
    """Configuration for data lake operations."""
    s3_bucket: str = "boardgame-nlp-data-lake"
    s3_prefix: str = "boardgame-nlp"
    partition_by: str = "date"  # date, game_id, or none
    file_format: str = "parquet"  # parquet, csv, json
    compression: str = "snappy"
    enable_versioning: bool = True
    metadata_table: str = "data_lake_metadata"

class DataLakeLoader:
    """Loads data to different layers of the data lake."""
    
    def __init__(self, config: DataLakeConfig = None):
        self.config = config or DataLakeConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')
        
    def load_to_raw_layer(self, local_path: str, s3_bucket: str = None, s3_key: str = None) -> str:
        """
        Load raw data to the raw layer of the data lake.
        
        Args:
            local_path: Local file path
            s3_bucket: S3 bucket name
            s3_key: S3 object key
            
        Returns:
            S3 path of uploaded data
        """
        s3_bucket = s3_bucket or self.config.s3_bucket
        s3_key = s3_key or self._generate_s3_key("raw", local_path)
        
        self.logger.info(f"Loading raw data to S3: s3://{s3_bucket}/{s3_key}")
        
        try:
            # Upload file to S3
            self.s3_client.upload_file(local_path, s3_bucket, s3_key)
            
            # Store metadata
            metadata = self._create_metadata(local_path, "raw", s3_bucket, s3_key)
            self._store_metadata(metadata)
            
            s3_path = f"s3://{s3_bucket}/{s3_key}"
            self.logger.info(f"Successfully loaded raw data to {s3_path}")
            
            return s3_path
            
        except Exception as e:
            self.logger.error(f"Failed to load raw data: {e}")
            raise
    
    def load_to_processed_layer(self, local_path: str, processing_type: str, 
                               s3_bucket: str = None, s3_key: str = None) -> str:
        """
        Load processed data to the processed layer.
        
        Args:
            local_path: Local file path
            processing_type: Type of processing (cleaned, transformed, etc.)
            s3_bucket: S3 bucket name
            s3_key: S3 object key
            
        Returns:
            S3 path of uploaded data
        """
        s3_bucket = s3_bucket or self.config.s3_bucket
        s3_key = s3_key or self._generate_s3_key("processed", local_path, processing_type)
        
        self.logger.info(f"Loading processed data to S3: s3://{s3_bucket}/{s3_key}")
        
        try:
            # Convert to appropriate format
            if self.config.file_format == "parquet":
                df = pd.read_csv(local_path)
                parquet_path = local_path.replace('.csv', '.parquet')
                df.to_parquet(parquet_path, compression=self.config.compression)
                upload_path = parquet_path
            else:
                upload_path = local_path
            
            # Upload to S3
            self.s3_client.upload_file(upload_path, s3_bucket, s3_key)
            
            # Store metadata
            metadata = self._create_metadata(local_path, "processed", s3_bucket, s3_key, processing_type)
            self._store_metadata(metadata)
            
            s3_path = f"s3://{s3_bucket}/{s3_key}"
            self.logger.info(f"Successfully loaded processed data to {s3_path}")
            
            return s3_path
            
        except Exception as e:
            self.logger.error(f"Failed to load processed data: {e}")
            raise
    
    def load_to_curated_layer(self, local_path: str, dataset_name: str,
                            s3_bucket: str = None, s3_key: str = None) -> str:
        """
        Load curated data to the curated layer.
        
        Args:
            local_path: Local file path
            dataset_name: Name of the curated dataset
            s3_bucket: S3 bucket name
            s3_key: S3 object key
            
        Returns:
            S3 path of uploaded data
        """
        s3_bucket = s3_bucket or self.config.s3_bucket
        s3_key = s3_key or self._generate_s3_key("curated", local_path, dataset_name)
        
        self.logger.info(f"Loading curated data to S3: s3://{s3_bucket}/{s3_key}")
        
        try:
            # Load and validate data
            df = pd.read_csv(local_path)
            
            # Apply data quality checks
            quality_report = self._validate_curated_data(df)
            
            # Convert to parquet for better performance
            if self.config.file_format == "parquet":
                parquet_path = local_path.replace('.csv', '.parquet')
                df.to_parquet(parquet_path, compression=self.config.compression)
                upload_path = parquet_path
            else:
                upload_path = local_path
            
            # Upload to S3
            self.s3_client.upload_file(upload_path, s3_bucket, s3_key)
            
            # Store metadata with quality report
            metadata = self._create_metadata(local_path, "curated", s3_bucket, s3_key, dataset_name)
            metadata['quality_report'] = quality_report
            self._store_metadata(metadata)
            
            s3_path = f"s3://{s3_bucket}/{s3_key}"
            self.logger.info(f"Successfully loaded curated data to {s3_path}")
            
            return s3_path
            
        except Exception as e:
            self.logger.error(f"Failed to load curated data: {e}")
            raise
    
    def load_ml_models(self, model_path: str, model_name: str, model_version: str,
                      s3_bucket: str = None, s3_key: str = None) -> str:
        """
        Load ML models to the model registry.
        
        Args:
            model_path: Local model directory path
            model_name: Name of the model
            model_version: Version of the model
            s3_bucket: S3 bucket name
            s3_key: S3 object key
            
        Returns:
            S3 path of uploaded model
        """
        s3_bucket = s3_bucket or self.config.s3_bucket
        s3_key = s3_key or f"ml_models/{model_name}/{model_version}/"
        
        self.logger.info(f"Loading ML model to S3: s3://{s3_bucket}/{s3_key}")
        
        try:
            # Upload model files
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file_path, model_path)
                    s3_file_key = f"{s3_key}{relative_path}"
                    
                    self.s3_client.upload_file(local_file_path, s3_bucket, s3_file_key)
            
            # Store model metadata
            model_metadata = {
                'model_name': model_name,
                'model_version': model_version,
                's3_path': f"s3://{s3_bucket}/{s3_key}",
                'uploaded_at': datetime.now().isoformat(),
                'file_count': sum(len(files) for _, _, files in os.walk(model_path))
            }
            
            self._store_metadata(model_metadata, metadata_type="model")
            
            s3_path = f"s3://{s3_bucket}/{s3_key}"
            self.logger.info(f"Successfully loaded model to {s3_path}")
            
            return s3_path
            
        except Exception as e:
            self.logger.error(f"Failed to load ML model: {e}")
            raise
    
    def _generate_s3_key(self, layer: str, local_path: str, suffix: str = None) -> str:
        """Generate S3 key based on layer and timestamp."""
        timestamp = datetime.now().strftime('%Y/%m/%d/%H%M%S')
        filename = Path(local_path).name
        
        if suffix:
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{suffix}{ext}"
        
        return f"{self.config.s3_prefix}/{layer}/{timestamp}/{filename}"
    
    def _create_metadata(self, local_path: str, layer: str, s3_bucket: str, 
                        s3_key: str, processing_type: str = None) -> Dict:
        """Create metadata for data lake entry."""
        file_stats = os.stat(local_path)
        
        metadata = {
            'layer': layer,
            'local_path': local_path,
            's3_bucket': s3_bucket,
            's3_key': s3_key,
            's3_path': f"s3://{s3_bucket}/{s3_key}",
            'file_size': file_stats.st_size,
            'uploaded_at': datetime.now().isoformat(),
            'processing_type': processing_type
        }
        
        return metadata
    
    def _store_metadata(self, metadata: Dict, metadata_type: str = "data") -> None:
        """Store metadata in S3."""
        try:
            metadata_key = f"{self.config.s3_prefix}/metadata/{metadata_type}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2),
                ContentType='application/json'
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store metadata: {e}")
    
    def _validate_curated_data(self, df: pd.DataFrame) -> Dict:
        """Validate curated data quality."""
        return {
            'row_count': len(df),
            'column_count': len(df.columns),
            'null_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def list_data_lake_contents(self, layer: str = None) -> List[Dict]:
        """List contents of the data lake."""
        try:
            prefix = f"{self.config.s3_prefix}/"
            if layer:
                prefix += f"{layer}/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.s3_bucket,
                Prefix=prefix
            )
            
            contents = []
            for obj in response.get('Contents', []):
                contents.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'storage_class': obj.get('StorageClass', 'STANDARD')
                })
            
            return contents
            
        except Exception as e:
            self.logger.error(f"Failed to list data lake contents: {e}")
            return []
