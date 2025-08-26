"""
Data storage service for market data with InfluxDB integration.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import hashlib
import json

from ..ingestion.interfaces import IDataSink, IDataStorageService
from ..models.market_data import MarketDataType, DataGranularity
from .sinks import InfluxDBDataSink, InMemoryDataSink


class DataStorageService(IDataStorageService):
    """
    Data storage service with batch writing, deduplication, and retention policies.
    """
    
    def __init__(
        self,
        primary_sink: IDataSink,
        backup_sink: Optional[IDataSink] = None,
        batch_size: int = 1000,
        batch_timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        deduplication_window: int = 300  # 5 minutes in seconds
    ):
        self.primary_sink = primary_sink
        self.backup_sink = backup_sink
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.deduplication_window = deduplication_window
        
        self.logger = logging.getLogger(__name__)
        
        # Batch management
        self._batch_buffer: List[Dict[str, Any]] = []
        self._batch_metadata: List[Dict[str, Any]] = []
        self._batch_lock = asyncio.Lock()
        self._last_batch_time = datetime.now()
        self._batch_task: Optional[asyncio.Task] = None
        
        # Deduplication
        self._dedup_cache: Dict[str, datetime] = {}
        self._dedup_lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            "total_writes": 0,
            "successful_writes": 0,
            "failed_writes": 0,
            "duplicates_filtered": 0,
            "batch_writes": 0,
            "retry_attempts": 0,
            "backup_writes": 0
        }
        
        # Batch processor task (will be started when needed)
        self._batch_task: Optional[asyncio.Task] = None
        self._batch_processor_started = False
    
    async def store_data(
        self, 
        data: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store single data record with deduplication and batching."""
        try:
            # Ensure batch processor is started
            await self._ensure_batch_processor()
            
            # Generate deduplication key
            dedup_key = self._generate_dedup_key(data)
            
            # Check for duplicates
            if await self._is_duplicate(dedup_key):
                self._stats["duplicates_filtered"] += 1
                self._stats["total_writes"] += 1  # Count duplicate as total write
                self.logger.debug(f"Duplicate data filtered for {data.get('symbol', 'unknown')}")
                return True
            
            # Add to batch buffer
            async with self._batch_lock:
                self._batch_buffer.append(data)
                self._batch_metadata.append(metadata or {})
                
                # Trigger batch write if buffer is full
                if len(self._batch_buffer) >= self.batch_size:
                    await self._flush_batch()
            
            self._stats["total_writes"] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing data: {e}")
            self._stats["failed_writes"] += 1
            return False
    
    async def store_batch(
        self, 
        data_batch: List[Dict[str, Any]], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store batch of data records with deduplication."""
        try:
            # Filter duplicates from batch
            filtered_batch = []
            filtered_metadata = []
            
            for data in data_batch:
                dedup_key = self._generate_dedup_key(data)
                if not await self._is_duplicate(dedup_key):
                    filtered_batch.append(data)
                    filtered_metadata.append(metadata or {})
                else:
                    self._stats["duplicates_filtered"] += 1
            
            if not filtered_batch:
                self.logger.debug("All records in batch were duplicates")
                return True
            
            # Write batch with retry logic
            success = await self._write_with_retry(filtered_batch, metadata)
            
            if success:
                self._stats["successful_writes"] += len(filtered_batch)
                self._stats["batch_writes"] += 1
            else:
                self._stats["failed_writes"] += len(filtered_batch)
            
            self._stats["total_writes"] += len(filtered_batch)
            return success
            
        except Exception as e:
            self.logger.error(f"Error storing batch: {e}")
            self._stats["failed_writes"] += len(data_batch)
            return False
    
    async def create_retention_policy(
        self, 
        policy_name: str, 
        duration: str, 
        replication: int = 1
    ) -> bool:
        """Create data retention policy."""
        try:
            if hasattr(self.primary_sink, 'create_retention_policy'):
                return await self.primary_sink.create_retention_policy(
                    policy_name, duration, replication
                )
            
            self.logger.warning("Primary sink does not support retention policies")
            return False
            
        except Exception as e:
            self.logger.error(f"Error creating retention policy: {e}")
            return False
    
    async def cleanup_old_data(self, older_than: datetime) -> int:
        """Clean up data older than specified date."""
        try:
            if hasattr(self.primary_sink, 'cleanup_old_data'):
                return await self.primary_sink.cleanup_old_data(older_than)
            
            self.logger.warning("Primary sink does not support data cleanup")
            return 0
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            return 0
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage performance statistics."""
        # Clean up old dedup cache entries
        await self._cleanup_dedup_cache()
        
        stats = {
            **self._stats,
            "batch_buffer_size": len(self._batch_buffer),
            "dedup_cache_size": len(self._dedup_cache),
            "success_rate": (
                self._stats["successful_writes"] / max(self._stats["total_writes"], 1) * 100
            )
        }
        
        # Add sink-specific stats if available
        if hasattr(self.primary_sink, 'get_stats'):
            stats["primary_sink_stats"] = self.primary_sink.get_stats()
        
        if self.backup_sink and hasattr(self.backup_sink, 'get_stats'):
            stats["backup_sink_stats"] = self.backup_sink.get_stats()
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of storage service."""
        health = {
            "overall": "healthy",
            "components": {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Check primary sink
            primary_healthy = await self.primary_sink.health_check()
            health["components"]["primary_sink"] = {
                "status": "healthy" if primary_healthy else "unhealthy",
                "type": type(self.primary_sink).__name__
            }
            
            # Check backup sink if available
            if self.backup_sink:
                backup_healthy = await self.backup_sink.health_check()
                health["components"]["backup_sink"] = {
                    "status": "healthy" if backup_healthy else "unhealthy",
                    "type": type(self.backup_sink).__name__
                }
            
            # Check batch processor
            batch_processor_healthy = (
                self._batch_processor_started and
                self._batch_task is not None and 
                not self._batch_task.done() and 
                not self._batch_task.cancelled()
            ) or not self._batch_processor_started  # Healthy if not started yet
            health["components"]["batch_processor"] = {
                "status": "healthy" if batch_processor_healthy else "unhealthy",
                "buffer_size": len(self._batch_buffer)
            }
            
            # Overall health determination
            if not primary_healthy:
                health["overall"] = "degraded" if self.backup_sink else "unhealthy"
            elif not batch_processor_healthy:
                health["overall"] = "degraded"
            
            return health
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            health["overall"] = "unhealthy"
            health["error"] = str(e)
            return health
    
    async def flush_all_batches(self) -> bool:
        """Force flush all pending batches."""
        try:
            async with self._batch_lock:
                if self._batch_buffer:
                    return await self._flush_batch()
            return True
            
        except Exception as e:
            self.logger.error(f"Error flushing batches: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown storage service gracefully."""
        self.logger.info("Shutting down storage service...")
        
        try:
            # Cancel batch processor
            if self._batch_task:
                self._batch_task.cancel()
                try:
                    await self._batch_task
                except asyncio.CancelledError:
                    pass
            
            # Flush any remaining batches
            await self.flush_all_batches()
            
            # Close sinks
            if hasattr(self.primary_sink, 'close'):
                await self.primary_sink.close()
            
            if self.backup_sink and hasattr(self.backup_sink, 'close'):
                await self.backup_sink.close()
            
            self.logger.info("Storage service shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def _generate_dedup_key(self, data: Dict[str, Any]) -> str:
        """Generate deduplication key from data."""
        key_data = {
            "symbol": data.get("symbol"),
            "data_type": data.get("data_type"),
            "timestamp": data.get("timestamp")
        }
        
        # Add granularity for intraday data
        if data.get("data_type") == "intraday":
            key_data["granularity"] = data.get("granularity")
        
        # Create hash of key data
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def _is_duplicate(self, dedup_key: str) -> bool:
        """Check if data is duplicate within deduplication window."""
        async with self._dedup_lock:
            if dedup_key in self._dedup_cache:
                # Check if within deduplication window
                cached_time = self._dedup_cache[dedup_key]
                if (datetime.now() - cached_time).seconds < self.deduplication_window:
                    return True
            
            # Update cache with current time
            self._dedup_cache[dedup_key] = datetime.now()
            return False
    
    async def _cleanup_dedup_cache(self):
        """Clean up old entries from deduplication cache."""
        async with self._dedup_lock:
            cutoff_time = datetime.now() - timedelta(seconds=self.deduplication_window * 2)
            keys_to_remove = [
                key for key, timestamp in self._dedup_cache.items() 
                if timestamp < cutoff_time
            ]
            
            for key in keys_to_remove:
                del self._dedup_cache[key]
    
    async def _write_with_retry(
        self, 
        data: List[Dict[str, Any]], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Write data with retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                # Try primary sink first
                success = await self.primary_sink.write_batch(data, metadata)
                
                if success:
                    return True
                
                # Track retry attempt
                if attempt < self.max_retries:
                    self._stats["retry_attempts"] += 1
                
                # If primary failed and we have backup, try backup
                if self.backup_sink:
                    self.logger.warning(f"Primary sink failed, trying backup (attempt {attempt + 1})")
                    backup_success = await self.backup_sink.write_batch(data, metadata)
                    if backup_success:
                        self._stats["backup_writes"] += 1
                        return True
                
                # If not last attempt, wait before retry
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                
            except Exception as e:
                self.logger.error(f"Write attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    self._stats["retry_attempts"] += 1
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        return False
    
    async def _flush_batch(self) -> bool:
        """Flush current batch to storage."""
        if not self._batch_buffer:
            return True
        
        batch_data = self._batch_buffer.copy()
        batch_metadata = {"batch_size": len(batch_data)}
        
        # Clear buffer
        self._batch_buffer.clear()
        self._batch_metadata.clear()
        self._last_batch_time = datetime.now()
        
        # Write batch
        success = await self._write_with_retry(batch_data, batch_metadata)
        
        if success:
            self._stats["successful_writes"] += len(batch_data)
            self._stats["batch_writes"] += 1
            self.logger.debug(f"Successfully wrote batch of {len(batch_data)} records")
        else:
            self._stats["failed_writes"] += len(batch_data)
            self.logger.error(f"Failed to write batch of {len(batch_data)} records")
        
        return success
    
    async def _ensure_batch_processor(self):
        """Ensure batch processor is started."""
        if not self._batch_processor_started:
            self._batch_processor_started = True
            await self._start_batch_processor()
    
    async def _start_batch_processor(self):
        """Start background batch processing task."""
        async def batch_processor():
            while True:
                try:
                    await asyncio.sleep(1)  # Check every second
                    
                    async with self._batch_lock:
                        # Check if batch timeout exceeded
                        time_since_last_batch = (datetime.now() - self._last_batch_time).seconds
                        
                        if (self._batch_buffer and 
                            time_since_last_batch >= self.batch_timeout):
                            await self._flush_batch()
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Batch processor error: {e}")
                    await asyncio.sleep(5)  # Wait before retrying
        
        self._batch_task = asyncio.create_task(batch_processor())


class DataRetentionManager:
    """Manages data retention policies and cleanup."""
    
    def __init__(self, storage_service: DataStorageService):
        self.storage_service = storage_service
        self.logger = logging.getLogger(__name__)
        
        # Default retention policies
        self.retention_policies = {
            "intraday_1m": timedelta(days=7),      # 1-minute data for 7 days
            "intraday_5m": timedelta(days=30),     # 5-minute data for 30 days
            "intraday_15m": timedelta(days=90),    # 15-minute data for 90 days
            "intraday_1h": timedelta(days=365),    # Hourly data for 1 year
            "daily": timedelta(days=365 * 5),      # Daily data for 5 years
            "quote": timedelta(days=1),            # Real-time quotes for 1 day
            "news": timedelta(days=90),            # News data for 90 days
        }
    
    async def apply_retention_policies(self) -> Dict[str, int]:
        """Apply all retention policies and return cleanup stats."""
        cleanup_stats = {}
        
        for policy_name, retention_period in self.retention_policies.items():
            try:
                cutoff_date = datetime.now(timezone.utc) - retention_period
                cleaned_count = await self.storage_service.cleanup_old_data(cutoff_date)
                
                cleanup_stats[policy_name] = cleaned_count
                
                if cleaned_count > 0:
                    self.logger.info(f"Cleaned {cleaned_count} records for policy {policy_name}")
                
            except Exception as e:
                self.logger.error(f"Error applying retention policy {policy_name}: {e}")
                cleanup_stats[policy_name] = -1
        
        return cleanup_stats
    
    def add_retention_policy(self, name: str, retention_period: timedelta):
        """Add custom retention policy."""
        self.retention_policies[name] = retention_period
        self.logger.info(f"Added retention policy: {name} - {retention_period}")
    
    def remove_retention_policy(self, name: str) -> bool:
        """Remove retention policy."""
        if name in self.retention_policies:
            del self.retention_policies[name]
            self.logger.info(f"Removed retention policy: {name}")
            return True
        return False