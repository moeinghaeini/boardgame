"""
FastAPI Application for Board Game NLP Data Engineering Solution

Provides REST API endpoints for data access, model inference, and monitoring.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from api.schemas import (
    CommentRequest, CommentResponse, SentimentAnalysisRequest,
    SentimentAnalysisResponse, ABSAAnalysisRequest, ABSAAnalysisResponse,
    DataQualityReport, ModelMetrics, HealthCheck
)
from api.services.data_service import DataService
from api.services.ml_service import MLService
from api.services.monitoring_service import MonitoringService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services
data_service = None
ml_service = None
monitoring_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global data_service, ml_service, monitoring_service
    
    # Startup
    logger.info("Starting Board Game NLP API...")
    
    try:
        data_service = DataService()
        ml_service = MLService()
        monitoring_service = MonitoringService()
        
        # Initialize services
        await data_service.initialize()
        await ml_service.initialize()
        await monitoring_service.initialize()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Board Game NLP API...")
    if data_service:
        await data_service.cleanup()
    if ml_service:
        await ml_service.cleanup()
    if monitoring_service:
        await monitoring_service.cleanup()

# Create FastAPI application
app = FastAPI(
    title="Board Game NLP API",
    description="Data Engineering Solution for Board Game Sentiment Analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection
def get_data_service() -> DataService:
    if data_service is None:
        raise HTTPException(status_code=503, detail="Data service not available")
    return data_service

def get_ml_service() -> MLService:
    if ml_service is None:
        raise HTTPException(status_code=503, detail="ML service not available")
    return ml_service

def get_monitoring_service() -> MonitoringService:
    if monitoring_service is None:
        raise HTTPException(status_code=503, detail="Monitoring service not available")
    return monitoring_service

# Health check endpoints
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    try:
        # Check service health
        data_health = await data_service.health_check() if data_service else False
        ml_health = await ml_service.health_check() if ml_service else False
        monitoring_health = await monitoring_service.health_check() if monitoring_service else False
        
        overall_health = all([data_health, ml_health, monitoring_health])
        
        return HealthCheck(
            status="healthy" if overall_health else "unhealthy",
            timestamp=datetime.now().isoformat(),
            services={
                "data_service": data_health,
                "ml_service": ml_health,
                "monitoring_service": monitoring_health
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )

@app.get("/health/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    try:
        if not all([data_service, ml_service, monitoring_service]):
            raise HTTPException(status_code=503, detail="Services not ready")
        
        # Check if services are ready
        data_ready = await data_service.is_ready() if data_service else False
        ml_ready = await ml_service.is_ready() if ml_service else False
        
        if not (data_ready and ml_ready):
            raise HTTPException(status_code=503, detail="Services not ready")
        
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

# Data access endpoints
@app.get("/api/v1/data/comments", response_model=List[CommentResponse])
async def get_comments(
    game_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    data_service: DataService = Depends(get_data_service)
):
    """Get board game comments with optional filtering."""
    try:
        comments = await data_service.get_comments(
            game_id=game_id,
            limit=limit,
            offset=offset
        )
        return comments
    except Exception as e:
        logger.error(f"Failed to get comments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/data/games")
async def get_games(
    limit: int = 50,
    data_service: DataService = Depends(get_data_service)
):
    """Get list of board games."""
    try:
        games = await data_service.get_games(limit=limit)
        return games
    except Exception as e:
        logger.error(f"Failed to get games: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ML inference endpoints
@app.post("/api/v1/ml/sentiment", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(
    request: SentimentAnalysisRequest,
    ml_service: MLService = Depends(get_ml_service)
):
    """Analyze sentiment of board game comments."""
    try:
        result = await ml_service.analyze_sentiment(
            comments=request.comments,
            batch_size=request.batch_size
        )
        return result
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ml/absa", response_model=ABSAAnalysisResponse)
async def analyze_absa(
    request: ABSAAnalysisRequest,
    ml_service: MLService = Depends(get_ml_service)
):
    """Perform aspect-based sentiment analysis."""
    try:
        result = await ml_service.analyze_absa(
            comments=request.comments,
            aspects=request.aspects,
            batch_size=request.batch_size
        )
        return result
    except Exception as e:
        logger.error(f"ABSA analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ml/batch-inference")
async def batch_inference(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    ml_service: MLService = Depends(get_ml_service)
):
    """Start batch inference job."""
    try:
        job_id = await ml_service.start_batch_inference(
            data_source=request.get("data_source"),
            analysis_type=request.get("analysis_type", "sentiment"),
            parameters=request.get("parameters", {})
        )
        
        # Start background task for monitoring
        background_tasks.add_task(ml_service.monitor_batch_job, job_id)
        
        return {"job_id": job_id, "status": "started"}
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ml/batch-inference/{job_id}")
async def get_batch_inference_status(
    job_id: str,
    ml_service: MLService = Depends(get_ml_service)
):
    """Get batch inference job status."""
    try:
        status = await ml_service.get_batch_job_status(job_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get batch job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring endpoints
@app.get("/api/v1/monitoring/data-quality", response_model=DataQualityReport)
async def get_data_quality_report(
    monitoring_service: MonitoringService = Depends(get_monitoring_service)
):
    """Get data quality report."""
    try:
        report = await monitoring_service.get_data_quality_report()
        return report
    except Exception as e:
        logger.error(f"Failed to get data quality report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/monitoring/model-metrics", response_model=ModelMetrics)
async def get_model_metrics(
    model_name: Optional[str] = None,
    monitoring_service: MonitoringService = Depends(get_monitoring_service)
):
    """Get model performance metrics."""
    try:
        metrics = await monitoring_service.get_model_metrics(model_name)
        return metrics
    except Exception as e:
        logger.error(f"Failed to get model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/monitoring/alerts")
async def get_active_alerts(
    monitoring_service: MonitoringService = Depends(get_monitoring_service)
):
    """Get active monitoring alerts."""
    try:
        alerts = await monitoring_service.get_active_alerts()
        return alerts
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data management endpoints
@app.post("/api/v1/data/refresh")
async def refresh_data(
    background_tasks: BackgroundTasks,
    data_service: DataService = Depends(get_data_service)
):
    """Trigger data refresh from BGG API."""
    try:
        job_id = await data_service.start_data_refresh()
        background_tasks.add_task(data_service.monitor_refresh_job, job_id)
        
        return {"job_id": job_id, "status": "started"}
    except Exception as e:
        logger.error(f"Data refresh failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/data/refresh/{job_id}")
async def get_refresh_status(
    job_id: str,
    data_service: DataService = Depends(get_data_service)
):
    """Get data refresh job status."""
    try:
        status = await data_service.get_refresh_job_status(job_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get refresh status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
