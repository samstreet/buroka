"""
Storage pipeline monitoring and management API endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta

from ...data.storage.service import DataStorageService, DataRetentionManager
from ...data.storage.sinks import InfluxDBDataSink, InMemoryDataSink, CompositeSink
from ...data.ingestion.client_factory import get_default_client
from ...data.ingestion.transformers import MarketDataTransformer
from ...data.ingestion.metrics import InMemoryMetricsCollector

router = APIRouter(prefix="/api/v1/storage", tags=["storage"])

# Global storage service instance
_storage_service: Optional[DataStorageService] = None
_retention_manager: Optional[DataRetentionManager] = None


def get_storage_service() -> DataStorageService:
    """Get or create the storage service instance."""
    global _storage_service, _retention_manager
    
    if _storage_service is None:
        # Create sinks
        try:
            # Try to create InfluxDB sink
            influx_sink = InfluxDBDataSink(
                url="http://influxdb:8086",
                token="dev-token",
                org="market-analysis",
                bucket="market-data",
                batch_size=500,
                flush_interval=15
            )
            primary_sink = influx_sink
        except:
            # Fallback to in-memory sink for development
            primary_sink = InMemoryDataSink(max_records=50000)
        
        # Create backup sink
        backup_sink = InMemoryDataSink(max_records=10000)
        
        # Create storage service
        _storage_service = DataStorageService(
            primary_sink=primary_sink,
            backup_sink=backup_sink,
            batch_size=1000,
            batch_timeout=30,
            max_retries=3,
            retry_delay=1.0,
            deduplication_window=300
        )
        
        # Create retention manager
        _retention_manager = DataRetentionManager(_storage_service)
    
    return _storage_service


def get_retention_manager() -> DataRetentionManager:
    """Get retention manager instance."""
    # Ensure storage service is created first
    get_storage_service()
    return _retention_manager


@router.get("/health", summary="Storage Health Check")
async def storage_health_check(
    service: DataStorageService = Depends(get_storage_service)
) -> Dict[str, Any]:
    """
    Perform comprehensive health check on the storage pipeline.
    
    Returns:
        Dict containing health status of all storage components
    """
    try:
        health = await service.health_check()
        return health
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage health check failed: {str(e)}")


@router.get("/stats", summary="Storage Statistics")
async def get_storage_stats(
    service: DataStorageService = Depends(get_storage_service)
) -> Dict[str, Any]:
    """
    Get comprehensive storage performance statistics.
    
    Returns:
        Dict containing storage performance metrics
    """
    try:
        stats = await service.get_storage_stats()
        
        # Add timestamp and additional metadata
        stats["metadata"] = {
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "service_uptime": "Not implemented",  # Would track service start time
            "version": "1.0.0"
        }
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get storage stats: {str(e)}")


@router.post("/store/single", summary="Store Single Record")
async def store_single_record(
    data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    service: DataStorageService = Depends(get_storage_service)
) -> Dict[str, Any]:
    """
    Store a single data record with deduplication and batching.
    
    Args:
        data: The data record to store
        metadata: Optional metadata to attach to the record
        
    Returns:
        Dict containing operation result
    """
    try:
        success = await service.store_data(data, metadata)
        
        return {
            "success": success,
            "message": "Data stored successfully" if success else "Failed to store data",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store data: {str(e)}")


@router.post("/store/batch", summary="Store Data Batch")
async def store_data_batch(
    data_batch: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    service: DataStorageService = Depends(get_storage_service)
) -> Dict[str, Any]:
    """
    Store a batch of data records with deduplication.
    
    Args:
        data_batch: List of data records to store
        metadata: Optional metadata to attach to the batch
        
    Returns:
        Dict containing batch operation result
    """
    try:
        if not data_batch:
            raise HTTPException(status_code=400, detail="Data batch cannot be empty")
        
        if len(data_batch) > 10000:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Batch size too large (max 10,000 records)")
        
        success = await service.store_batch(data_batch, metadata)
        
        return {
            "success": success,
            "batch_size": len(data_batch),
            "message": f"Batch of {len(data_batch)} records processed successfully" if success else "Failed to store batch",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store batch: {str(e)}")


@router.post("/flush", summary="Force Flush Batches")
async def flush_batches(
    service: DataStorageService = Depends(get_storage_service)
) -> Dict[str, Any]:
    """
    Force flush all pending batches to storage.
    
    Returns:
        Dict containing flush operation result
    """
    try:
        success = await service.flush_all_batches()
        
        return {
            "success": success,
            "message": "All batches flushed successfully" if success else "Failed to flush batches",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to flush batches: {str(e)}")


@router.get("/retention/policies", summary="Get Retention Policies")
async def get_retention_policies(
    retention_manager: DataRetentionManager = Depends(get_retention_manager)
) -> Dict[str, Any]:
    """
    Get all configured retention policies.
    
    Returns:
        Dict containing retention policies and their settings
    """
    try:
        policies = {}
        for name, period in retention_manager.retention_policies.items():
            policies[name] = {
                "retention_period": str(period),
                "retention_days": period.days
            }
        
        return {
            "policies": policies,
            "total_policies": len(policies),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get retention policies: {str(e)}")


@router.post("/retention/policies", summary="Add Retention Policy")
async def add_retention_policy(
    policy_name: str,
    retention_days: int,
    retention_manager: DataRetentionManager = Depends(get_retention_manager)
) -> Dict[str, Any]:
    """
    Add a new retention policy.
    
    Args:
        policy_name: Name of the retention policy
        retention_days: Number of days to retain data
        
    Returns:
        Dict containing operation result
    """
    try:
        if retention_days <= 0:
            raise HTTPException(status_code=400, detail="Retention days must be positive")
        
        if retention_days > 365 * 10:  # 10 years max
            raise HTTPException(status_code=400, detail="Retention days cannot exceed 10 years")
        
        retention_period = timedelta(days=retention_days)
        retention_manager.add_retention_policy(policy_name, retention_period)
        
        return {
            "success": True,
            "policy_name": policy_name,
            "retention_days": retention_days,
            "message": f"Retention policy '{policy_name}' added successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add retention policy: {str(e)}")


@router.delete("/retention/policies/{policy_name}", summary="Remove Retention Policy")
async def remove_retention_policy(
    policy_name: str,
    retention_manager: DataRetentionManager = Depends(get_retention_manager)
) -> Dict[str, Any]:
    """
    Remove a retention policy.
    
    Args:
        policy_name: Name of the retention policy to remove
        
    Returns:
        Dict containing operation result
    """
    try:
        success = retention_manager.remove_retention_policy(policy_name)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Retention policy '{policy_name}' not found")
        
        return {
            "success": True,
            "policy_name": policy_name,
            "message": f"Retention policy '{policy_name}' removed successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove retention policy: {str(e)}")


@router.post("/retention/cleanup", summary="Apply Retention Policies")
async def apply_retention_cleanup(
    background_tasks: BackgroundTasks,
    retention_manager: DataRetentionManager = Depends(get_retention_manager)
) -> Dict[str, Any]:
    """
    Apply all retention policies and clean up old data.
    
    Returns:
        Dict containing cleanup operation result
    """
    try:
        # Run cleanup in background
        background_tasks.add_task(_apply_retention_cleanup_task, retention_manager)
        
        return {
            "success": True,
            "message": "Retention cleanup started in background",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start retention cleanup: {str(e)}")


async def _apply_retention_cleanup_task(retention_manager: DataRetentionManager):
    """Background task to apply retention cleanup."""
    try:
        cleanup_stats = await retention_manager.apply_retention_policies()
        total_cleaned = sum(count for count in cleanup_stats.values() if count > 0)
        
        # Log results
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Retention cleanup completed: {total_cleaned} records cleaned")
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Retention cleanup failed: {e}")


@router.get("/retention/cleanup/status", summary="Get Cleanup Status")
async def get_cleanup_status(
    retention_manager: DataRetentionManager = Depends(get_retention_manager)
) -> Dict[str, Any]:
    """
    Get status of the last retention cleanup operation.
    
    Returns:
        Dict containing cleanup status information
    """
    try:
        # This would normally track cleanup job status
        # For now, return basic information
        return {
            "status": "not_implemented",
            "message": "Cleanup status tracking not yet implemented",
            "last_cleanup": "unknown",
            "next_cleanup": "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cleanup status: {str(e)}")


@router.get("/performance/metrics", summary="Storage Performance Metrics")
async def get_performance_metrics(
    service: DataStorageService = Depends(get_storage_service)
) -> Dict[str, Any]:
    """
    Get detailed storage performance metrics.
    
    Returns:
        Dict containing performance metrics and analysis
    """
    try:
        stats = await service.get_storage_stats()
        
        # Calculate derived metrics
        total_operations = stats.get("total_writes", 0)
        successful_operations = stats.get("successful_writes", 0)
        failed_operations = stats.get("failed_writes", 0)
        
        performance_metrics = {
            "throughput": {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "failed_operations": failed_operations,
                "success_rate": stats.get("success_rate", 0.0),
                "failure_rate": (failed_operations / max(total_operations, 1)) * 100
            },
            "efficiency": {
                "duplicates_filtered": stats.get("duplicates_filtered", 0),
                "batch_writes": stats.get("batch_writes", 0),
                "retry_attempts": stats.get("retry_attempts", 0),
                "backup_writes": stats.get("backup_writes", 0)
            },
            "resource_usage": {
                "batch_buffer_size": stats.get("batch_buffer_size", 0),
                "dedup_cache_size": stats.get("dedup_cache_size", 0),
                "memory_efficiency": "good" if stats.get("dedup_cache_size", 0) < 1000 else "needs_attention"
            },
            "quality_indicators": {
                "deduplication_effectiveness": (
                    (stats.get("duplicates_filtered", 0) / max(total_operations, 1)) * 100
                ),
                "retry_rate": (
                    (stats.get("retry_attempts", 0) / max(total_operations, 1)) * 100
                ),
                "backup_usage_rate": (
                    (stats.get("backup_writes", 0) / max(total_operations, 1)) * 100
                )
            },
            "metadata": {
                "collection_timestamp": datetime.now(timezone.utc).isoformat(),
                "analysis_version": "1.0.0"
            }
        }
        
        return performance_metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@router.get("/dashboard", summary="Storage Dashboard Data")
async def get_dashboard_data(
    service: DataStorageService = Depends(get_storage_service),
    retention_manager: DataRetentionManager = Depends(get_retention_manager)
) -> Dict[str, Any]:
    """
    Get comprehensive dashboard data for storage monitoring UI.
    
    Returns:
        Dict containing all dashboard data
    """
    try:
        # Get all data in parallel
        import asyncio
        
        health_task = asyncio.create_task(service.health_check())
        stats_task = asyncio.create_task(service.get_storage_stats())
        
        health, stats = await asyncio.gather(health_task, stats_task)
        
        # Compile dashboard data
        dashboard = {
            "overview": {
                "status": health.get("overall", "unknown"),
                "total_writes": stats.get("total_writes", 0),
                "success_rate": stats.get("success_rate", 0.0),
                "batch_buffer_size": stats.get("batch_buffer_size", 0),
                "last_updated": datetime.now(timezone.utc).isoformat()
            },
            "health": health,
            "performance": {
                "throughput": {
                    "total_writes": stats.get("total_writes", 0),
                    "successful_writes": stats.get("successful_writes", 0),
                    "failed_writes": stats.get("failed_writes", 0),
                    "batch_writes": stats.get("batch_writes", 0)
                },
                "efficiency": {
                    "duplicates_filtered": stats.get("duplicates_filtered", 0),
                    "retry_attempts": stats.get("retry_attempts", 0),
                    "backup_writes": stats.get("backup_writes", 0)
                }
            },
            "retention": {
                "total_policies": len(retention_manager.retention_policies),
                "policy_names": list(retention_manager.retention_policies.keys()),
                "last_cleanup": "not_implemented"
            },
            "alerts": []
        }
        
        # Add performance alerts
        if stats.get("success_rate", 100) < 95:
            dashboard["alerts"].append({
                "type": "warning",
                "message": f"Success rate below 95%: {stats.get('success_rate', 0):.1f}%"
            })
        
        if stats.get("batch_buffer_size", 0) > 5000:
            dashboard["alerts"].append({
                "type": "warning",
                "message": f"Large batch buffer: {stats.get('batch_buffer_size', 0)} records"
            })
        
        if stats.get("retry_attempts", 0) > stats.get("total_writes", 1) * 0.1:
            dashboard["alerts"].append({
                "type": "warning",
                "message": "High retry rate detected"
            })
        
        return dashboard
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")