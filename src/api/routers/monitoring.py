"""
Monitoring and metrics API endpoints for data ingestion.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from ...data.ingestion.service import DataIngestionService
from ...data.ingestion.client_factory import get_default_client
from ...data.ingestion.transformers import MarketDataTransformer
from ...data.ingestion.metrics import InMemoryMetricsCollector
from ...data.storage.sinks import InMemoryDataSink
from ...data.models.market_data import MarketDataType, DataGranularity

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])

# Global ingestion service instance (in production, this would be properly injected)
_ingestion_service: Optional[DataIngestionService] = None


def get_ingestion_service() -> DataIngestionService:
    """Get or create the ingestion service instance."""
    global _ingestion_service
    
    if _ingestion_service is None:
        # Initialize ingestion service with dependencies
        data_source = get_default_client()
        data_transformer = MarketDataTransformer()
        data_sink = InMemoryDataSink()  # In production, use InfluxDB or other storage
        metrics_collector = InMemoryMetricsCollector()
        
        _ingestion_service = DataIngestionService(
            data_source=data_source,
            data_transformer=data_transformer,
            data_sink=data_sink,
            metrics_collector=metrics_collector
        )
    
    return _ingestion_service


@router.get("/health", summary="System Health Check")
async def health_check(service: DataIngestionService = Depends(get_ingestion_service)) -> Dict[str, Any]:
    """
    Perform comprehensive health check on the ingestion system.
    
    Returns:
        Dict containing health status of all components
    """
    try:
        health = await service.health_check()
        
        # Add API-specific health information
        health["api"] = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0"
        }
        
        return health
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/status", summary="Ingestion Status")
async def get_ingestion_status(service: DataIngestionService = Depends(get_ingestion_service)) -> Dict[str, Any]:
    """
    Get current status of the data ingestion pipeline.
    
    Returns:
        Dict containing ingestion status and statistics
    """
    try:
        status = await service.get_ingestion_status()
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/metrics", summary="System Metrics")
async def get_metrics(service: DataIngestionService = Depends(get_ingestion_service)) -> Dict[str, Any]:
    """
    Get system metrics and performance statistics.
    
    Returns:
        Dict containing all collected metrics
    """
    try:
        metrics = service.metrics_collector.get_metrics_summary()
        
        # Add timestamp and metadata
        metrics["metadata"] = {
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "service_uptime": "Not implemented",  # Would track service start time
            "version": "1.0.0"
        }
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/jobs", summary="Active Jobs")
async def get_active_jobs(service: DataIngestionService = Depends(get_ingestion_service)) -> List[Dict[str, Any]]:
    """
    Get list of currently active ingestion jobs.
    
    Returns:
        List of active ingestion jobs with their details
    """
    try:
        jobs = await service.scheduler.get_active_jobs()
        return jobs
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get jobs: {str(e)}")


@router.post("/jobs/start", summary="Start Continuous Ingestion")
async def start_continuous_ingestion(
    symbols: List[str],
    interval: int = 300,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    service: DataIngestionService = Depends(get_ingestion_service)
) -> Dict[str, Any]:
    """
    Start continuous data ingestion for specified symbols.
    
    Args:
        symbols: List of stock symbols to monitor
        interval: Ingestion interval in seconds (default: 300)
        
    Returns:
        Dict containing job information
    """
    try:
        if not symbols:
            raise HTTPException(status_code=400, detail="At least one symbol is required")
        
        if interval < 30:
            raise HTTPException(status_code=400, detail="Minimum interval is 30 seconds")
        
        batch_job_id = await service.start_continuous_ingestion(symbols, interval)
        
        return {
            "success": True,
            "batch_job_id": batch_job_id,
            "symbols": symbols,
            "interval": interval,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "message": f"Started continuous ingestion for {len(symbols)} symbols"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start ingestion: {str(e)}")


@router.post("/jobs/stop/{job_id}", summary="Stop Continuous Ingestion")
async def stop_continuous_ingestion(
    job_id: str,
    service: DataIngestionService = Depends(get_ingestion_service)
) -> Dict[str, Any]:
    """
    Stop continuous data ingestion for a specific job.
    
    Args:
        job_id: ID of the job to stop
        
    Returns:
        Dict containing operation result
    """
    try:
        success = await service.stop_continuous_ingestion(job_id)
        
        if success:
            return {
                "success": True,
                "job_id": job_id,
                "stopped_at": datetime.now(timezone.utc).isoformat(),
                "message": "Continuous ingestion stopped successfully"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop ingestion: {str(e)}")


@router.post("/ingest/symbol", summary="Ingest Single Symbol")
async def ingest_symbol_data(
    symbol: str,
    data_type: str = "quote",
    granularity: Optional[str] = None,
    service: DataIngestionService = Depends(get_ingestion_service)
) -> Dict[str, Any]:
    """
    Manually trigger data ingestion for a single symbol.
    
    Args:
        symbol: Stock symbol to ingest
        data_type: Type of data (quote, daily, intraday)
        granularity: Data granularity (for intraday data)
        
    Returns:
        Dict containing ingestion result
    """
    try:
        # Validate parameters
        valid_data_types = ["quote", "daily", "intraday"]
        if data_type not in valid_data_types:
            raise HTTPException(status_code=400, detail=f"Invalid data_type. Must be one of: {valid_data_types}")
        
        # Convert string to enum
        data_type_enum = MarketDataType(data_type)
        granularity_enum = None
        if granularity:
            try:
                granularity_enum = DataGranularity(granularity)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid granularity: {granularity}")
        
        # Perform ingestion
        result = await service.ingest_symbol_data(symbol, data_type_enum, granularity_enum)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/ingest/batch", summary="Ingest Multiple Symbols")
async def ingest_batch_data(
    symbols: List[str],
    data_type: str = "quote",
    service: DataIngestionService = Depends(get_ingestion_service)
) -> Dict[str, Any]:
    """
    Manually trigger batch data ingestion for multiple symbols.
    
    Args:
        symbols: List of stock symbols to ingest
        data_type: Type of data (quote, daily, intraday)
        
    Returns:
        Dict containing batch ingestion results
    """
    try:
        if not symbols:
            raise HTTPException(status_code=400, detail="At least one symbol is required")
        
        if len(symbols) > 50:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Maximum 50 symbols per batch")
        
        # Validate data type
        valid_data_types = ["quote", "daily", "intraday"]
        if data_type not in valid_data_types:
            raise HTTPException(status_code=400, detail=f"Invalid data_type. Must be one of: {valid_data_types}")
        
        # Convert string to enum
        data_type_enum = MarketDataType(data_type)
        
        # Perform batch ingestion
        result = await service.ingest_batch_data(symbols, data_type_enum)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch ingestion failed: {str(e)}")


@router.get("/dashboard", summary="Monitoring Dashboard Data")
async def get_dashboard_data(service: DataIngestionService = Depends(get_ingestion_service)) -> Dict[str, Any]:
    """
    Get comprehensive dashboard data for monitoring UI.
    
    Returns:
        Dict containing all dashboard data
    """
    try:
        # Get all monitoring data in parallel
        import asyncio
        
        health_task = asyncio.create_task(service.health_check())
        status_task = asyncio.create_task(service.get_ingestion_status())
        jobs_task = asyncio.create_task(service.scheduler.get_active_jobs())
        
        health, status, jobs = await asyncio.gather(health_task, status_task, jobs_task)
        metrics = service.metrics_collector.get_metrics_summary()
        
        # Compile dashboard data
        dashboard = {
            "overview": {
                "status": health.get("overall", "unknown"),
                "active_jobs": len(jobs),
                "total_requests": metrics.get("counters", {}).get("ingestion_requests_total", 0),
                "success_rate": _calculate_success_rate(metrics),
                "last_updated": datetime.now(timezone.utc).isoformat()
            },
            "health": health,
            "metrics": metrics,
            "jobs": jobs,
            "performance": {
                "avg_request_time": _get_avg_metric(metrics, "ingestion_request_duration_seconds"),
                "avg_transformation_time": _get_avg_metric(metrics, "ingestion_transformation_duration_seconds"),
                "circuit_breaker_state": status.get("circuit_breaker_state", "unknown")
            },
            "errors": {
                "validation_errors": metrics.get("counters", {}).get("ingestion_validation_errors_total", 0),
                "transformation_errors": metrics.get("counters", {}).get("ingestion_transformation_errors_total", 0),
                "failed_requests": metrics.get("counters", {}).get("ingestion_requests_failed_total", 0)
            }
        }
        
        return dashboard
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")


@router.delete("/metrics/reset", summary="Reset Metrics")
async def reset_metrics(service: DataIngestionService = Depends(get_ingestion_service)) -> Dict[str, Any]:
    """
    Reset all collected metrics (useful for development/testing).
    
    Returns:
        Dict containing operation result
    """
    try:
        if hasattr(service.metrics_collector, 'reset_metrics'):
            service.metrics_collector.reset_metrics()
        
        return {
            "success": True,
            "message": "Metrics reset successfully",
            "reset_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset metrics: {str(e)}")


def _calculate_success_rate(metrics: Dict[str, Any]) -> float:
    """Calculate success rate from metrics."""
    counters = metrics.get("counters", {})
    total = counters.get("ingestion_requests_total", 0)
    success = counters.get("ingestion_requests_success_total", 0)
    
    if total == 0:
        return 0.0
    
    return round((success / total) * 100, 2)


def _get_avg_metric(metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
    """Get average value for a timing metric."""
    timings = metrics.get("timings", {})
    
    for key, value in timings.items():
        if metric_name in key:
            return round(value.get("avg", 0), 4)
    
    return None