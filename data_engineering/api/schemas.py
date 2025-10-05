"""
Pydantic schemas for Board Game NLP API

Defines data models for request/response validation and serialization.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum

class AnalysisType(str, Enum):
    """Types of analysis available."""
    SENTIMENT = "sentiment"
    ABSA = "absa"
    BOTH = "both"

class SentimentLabel(str, Enum):
    """Sentiment labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class AspectType(str, Enum):
    """Available aspects for ABSA."""
    LUCK = "luck"
    BOOKKEEPING = "bookkeeping"
    DOWNTIME = "downtime"
    INTERACTION = "interaction"
    BASH_THE_LEADER = "bash_the_leader"
    COMPLEXITY = "complexity"

class JobStatus(str, Enum):
    """Job status values."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Request schemas
class CommentRequest(BaseModel):
    """Request schema for single comment analysis."""
    comment: str = Field(..., min_length=1, max_length=5000, description="Comment text to analyze")
    game_id: Optional[str] = Field(None, description="Board game ID")
    user_id: Optional[str] = Field(None, description="User ID")

class SentimentAnalysisRequest(BaseModel):
    """Request schema for sentiment analysis."""
    comments: List[str] = Field(..., min_items=1, max_items=1000, description="List of comments to analyze")
    batch_size: int = Field(16, ge=1, le=128, description="Batch size for processing")
    return_confidence: bool = Field(True, description="Whether to return confidence scores")

class ABSAAnalysisRequest(BaseModel):
    """Request schema for aspect-based sentiment analysis."""
    comments: List[str] = Field(..., min_items=1, max_items=1000, description="List of comments to analyze")
    aspects: Optional[List[AspectType]] = Field(None, description="Specific aspects to analyze")
    batch_size: int = Field(16, ge=1, le=128, description="Batch size for processing")
    return_confidence: bool = Field(True, description="Whether to return confidence scores")

class BatchInferenceRequest(BaseModel):
    """Request schema for batch inference."""
    data_source: str = Field(..., description="Data source path or identifier")
    analysis_type: AnalysisType = Field(AnalysisType.SENTIMENT, description="Type of analysis to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

# Response schemas
class CommentResponse(BaseModel):
    """Response schema for comment data."""
    id: str = Field(..., description="Comment ID")
    boardgame_id: str = Field(..., description="Board game ID")
    username: str = Field(..., description="Username")
    rating: Optional[float] = Field(None, description="User rating")
    comment: str = Field(..., description="Comment text")
    cleaned_comment: str = Field(..., description="Cleaned comment text")
    created_at: datetime = Field(..., description="Creation timestamp")

class SentimentResult(BaseModel):
    """Sentiment analysis result."""
    comment: str = Field(..., description="Original comment")
    sentiment: SentimentLabel = Field(..., description="Predicted sentiment")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")

class SentimentAnalysisResponse(BaseModel):
    """Response schema for sentiment analysis."""
    results: List[SentimentResult] = Field(..., description="Analysis results")
    total_comments: int = Field(..., description="Total number of comments processed")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_version: str = Field(..., description="Model version used")

class ABSAResult(BaseModel):
    """ABSA analysis result."""
    comment: str = Field(..., description="Original comment")
    detected_aspects: List[str] = Field(..., description="Detected aspects")
    aspect_sentiments: Dict[str, str] = Field(..., description="Sentiment for each aspect")
    aspect_confidences: Dict[str, float] = Field(..., description="Confidence for each aspect")

class ABSAAnalysisResponse(BaseModel):
    """Response schema for ABSA analysis."""
    results: List[ABSAResult] = Field(..., description="Analysis results")
    total_comments: int = Field(..., description="Total number of comments processed")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_version: str = Field(..., description="Model version used")

class JobStatusResponse(BaseModel):
    """Response schema for job status."""
    job_id: str = Field(..., description="Job ID")
    status: JobStatus = Field(..., description="Current job status")
    progress: float = Field(..., ge=0.0, le=100.0, description="Progress percentage")
    started_at: datetime = Field(..., description="Job start time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    result_path: Optional[str] = Field(None, description="Path to results if completed")

# Monitoring schemas
class DataQualityMetric(BaseModel):
    """Data quality metric."""
    metric_name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Metric value")
    threshold: Optional[float] = Field(None, description="Quality threshold")
    status: str = Field(..., description="Quality status (good/warning/error)")

class DataQualityReport(BaseModel):
    """Data quality report."""
    report_id: str = Field(..., description="Report ID")
    generated_at: datetime = Field(..., description="Report generation time")
    data_source: str = Field(..., description="Data source identifier")
    total_records: int = Field(..., description="Total number of records")
    quality_score: float = Field(..., ge=0.0, le=100.0, description="Overall quality score")
    metrics: List[DataQualityMetric] = Field(..., description="Quality metrics")
    issues: List[str] = Field(default_factory=list, description="Identified issues")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")

class ModelMetrics(BaseModel):
    """Model performance metrics."""
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Model accuracy")
    precision: float = Field(..., ge=0.0, le=1.0, description="Model precision")
    recall: float = Field(..., ge=0.0, le=1.0, description="Model recall")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1 score")
    last_updated: datetime = Field(..., description="Last update time")
    training_samples: int = Field(..., description="Number of training samples")
    validation_samples: int = Field(..., description="Number of validation samples")

class Alert(BaseModel):
    """Monitoring alert."""
    alert_id: str = Field(..., description="Alert ID")
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(..., description="Alert severity (low/medium/high/critical)")
    message: str = Field(..., description="Alert message")
    created_at: datetime = Field(..., description="Alert creation time")
    resolved_at: Optional[datetime] = Field(None, description="Alert resolution time")
    status: str = Field(..., description="Alert status (active/resolved)")

class HealthCheck(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Check timestamp")
    services: Dict[str, bool] = Field(..., description="Service health status")
    error: Optional[str] = Field(None, description="Error message if unhealthy")

# Game and data schemas
class GameInfo(BaseModel):
    """Board game information."""
    game_id: str = Field(..., description="Game ID")
    name: str = Field(..., description="Game name")
    rank: Optional[int] = Field(None, description="BGG rank")
    rating: Optional[float] = Field(None, description="Average rating")
    comment_count: int = Field(..., description="Number of comments")

class DataSource(BaseModel):
    """Data source information."""
    source_id: str = Field(..., description="Source identifier")
    source_type: str = Field(..., description="Type of data source")
    last_updated: datetime = Field(..., description="Last update time")
    record_count: int = Field(..., description="Number of records")
    data_quality_score: float = Field(..., ge=0.0, le=100.0, description="Data quality score")

# Validation methods
class SentimentAnalysisRequest(BaseModel):
    """Request schema for sentiment analysis with validation."""
    comments: List[str] = Field(..., min_items=1, max_items=1000)
    batch_size: int = Field(16, ge=1, le=128)
    return_confidence: bool = Field(True)
    
    @validator('comments')
    def validate_comments(cls, v):
        """Validate comment list."""
        if not v:
            raise ValueError('Comments list cannot be empty')
        
        for comment in v:
            if len(comment.strip()) < 1:
                raise ValueError('Comments cannot be empty')
            if len(comment) > 5000:
                raise ValueError('Comments cannot exceed 5000 characters')
        
        return v

class ABSAAnalysisRequest(BaseModel):
    """Request schema for ABSA with validation."""
    comments: List[str] = Field(..., min_items=1, max_items=1000)
    aspects: Optional[List[AspectType]] = None
    batch_size: int = Field(16, ge=1, le=128)
    return_confidence: bool = Field(True)
    
    @validator('comments')
    def validate_comments(cls, v):
        """Validate comment list."""
        if not v:
            raise ValueError('Comments list cannot be empty')
        
        for comment in v:
            if len(comment.strip()) < 1:
                raise ValueError('Comments cannot be empty')
            if len(comment) > 5000:
                raise ValueError('Comments cannot exceed 5000 characters')
        
        return v
    
    @validator('aspects')
    def validate_aspects(cls, v):
        """Validate aspects list."""
        if v is not None and len(v) == 0:
            raise ValueError('Aspects list cannot be empty if provided')
        return v
