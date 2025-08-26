"""
Main data ingestion service implementation following SOLID principles.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

from .interfaces import (
    IDataIngestionService, IDataSource, IDataTransformer, 
    IDataSink, IMetricsCollector, IIngestionScheduler, ICircuitBreaker
)
from .transformers import MarketDataTransformer, BatchDataTransformer
from .metrics import IngestionMetrics, TimingContextManager, create_metrics_collector
from ..models.market_data import MarketDataType, DataGranularity, MarketDataResponse


class JobStatus(str, Enum):
    """Status of ingestion jobs."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class IngestionJob:
    """Represents a data ingestion job."""
    job_id: str
    symbol: str
    data_type: MarketDataType
    granularity: Optional[DataGranularity]
    interval: int  # seconds
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    failure_count: int = 0
    max_failures: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleCircuitBreaker(ICircuitBreaker):
    """Simple circuit breaker implementation."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.logger = logging.getLogger(__name__)
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
                self.failure_count = 0
            else:
                raise Exception("Circuit breaker is open - too many failures")
        
        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except self.expected_exception as e:
            self.on_failure()
            raise e
    
    def get_state(self) -> str:
        """Get current circuit breaker state."""
        return self.state
    
    def get_failure_count(self) -> int:
        """Get current failure count."""
        return self.failure_count
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None
    
    def on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"
    
    def on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class IngestionScheduler(IIngestionScheduler):
    """Simple scheduler for data ingestion jobs."""
    
    def __init__(self):
        self.jobs: Dict[str, IngestionJob] = {}
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()
    
    async def schedule_ingestion(
        self, 
        symbol: str, 
        data_type: MarketDataType, 
        interval: int,
        granularity: Optional[DataGranularity] = None
    ) -> str:
        """Schedule data ingestion for a symbol."""
        async with self._lock:
            job_id = str(uuid.uuid4())
            job = IngestionJob(
                job_id=job_id,
                symbol=symbol,
                data_type=data_type,
                granularity=granularity,
                interval=interval,
                next_run=datetime.now(timezone.utc)
            )
            self.jobs[job_id] = job
            self.logger.info(f"Scheduled ingestion job {job_id} for {symbol}")
            return job_id
    
    async def cancel_ingestion(self, job_id: str) -> bool:
        """Cancel a scheduled ingestion job."""
        async with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id].status = JobStatus.CANCELLED
                self.logger.info(f"Cancelled ingestion job {job_id}")
                return True
            return False
    
    async def get_active_jobs(self) -> List[Dict[str, Any]]:
        """Get list of active ingestion jobs."""
        async with self._lock:
            active_jobs = []
            for job in self.jobs.values():
                if job.status not in [JobStatus.COMPLETED, JobStatus.CANCELLED]:
                    active_jobs.append({
                        "job_id": job.job_id,
                        "symbol": job.symbol,
                        "data_type": job.data_type.value,
                        "granularity": job.granularity.value if job.granularity else None,
                        "interval": job.interval,
                        "status": job.status.value,
                        "created_at": job.created_at.isoformat(),
                        "last_run": job.last_run.isoformat() if job.last_run else None,
                        "next_run": job.next_run.isoformat() if job.next_run else None,
                        "failure_count": job.failure_count
                    })
            return active_jobs
    
    async def pause_ingestion(self, job_id: str) -> bool:
        """Pause an ingestion job."""
        async with self._lock:
            if job_id in self.jobs and self.jobs[job_id].status == JobStatus.RUNNING:
                self.jobs[job_id].status = JobStatus.PAUSED
                return True
            return False
    
    async def resume_ingestion(self, job_id: str) -> bool:
        """Resume a paused ingestion job."""
        async with self._lock:
            if job_id in self.jobs and self.jobs[job_id].status == JobStatus.PAUSED:
                self.jobs[job_id].status = JobStatus.RUNNING
                return True
            return False
    
    async def get_ready_jobs(self) -> List[IngestionJob]:
        """Get jobs that are ready to run."""
        async with self._lock:
            ready_jobs = []
            now = datetime.now(timezone.utc)
            
            for job in self.jobs.values():
                if (job.status in [JobStatus.PENDING, JobStatus.RUNNING] and
                    job.next_run and job.next_run <= now and
                    job.failure_count < job.max_failures):
                    ready_jobs.append(job)
            
            return ready_jobs


class DataIngestionService(IDataIngestionService):
    """Main data ingestion service implementation."""
    
    def __init__(
        self,
        data_source: IDataSource,
        data_transformer: Optional[IDataTransformer] = None,
        data_sink: Optional[IDataSink] = None,
        metrics_collector: Optional[IMetricsCollector] = None,
        scheduler: Optional[IIngestionScheduler] = None,
        circuit_breaker: Optional[ICircuitBreaker] = None,
        max_concurrent_jobs: int = 10
    ):
        self.data_source = data_source
        self.data_transformer = data_transformer or MarketDataTransformer()
        self.data_sink = data_sink
        self.metrics_collector = metrics_collector or create_metrics_collector()
        self.scheduler = scheduler or IngestionScheduler()
        self.circuit_breaker = circuit_breaker or SimpleCircuitBreaker()
        self.max_concurrent_jobs = max_concurrent_jobs
        
        self.logger = logging.getLogger(__name__)
        self._running_jobs: Set[str] = set()
        self._shutdown_event = asyncio.Event()
        self._background_task: Optional[asyncio.Task] = None
    
    async def ingest_symbol_data(
        self, 
        symbol: str, 
        data_type: MarketDataType,
        granularity: Optional[DataGranularity] = None
    ) -> Dict[str, Any]:
        """Ingest data for a single symbol."""
        start_time = time.time()
        tags = IngestionMetrics.create_tags(symbol=symbol, data_type=data_type.value)
        
        try:
            self.metrics_collector.increment_counter(IngestionMetrics.REQUESTS_TOTAL, 1, tags)
            
            # Validate symbol
            if not self.data_source.validate_symbol(symbol):
                self.metrics_collector.increment_counter(
                    IngestionMetrics.VALIDATION_ERRORS, 1, 
                    IngestionMetrics.create_tags(symbol=symbol, error_type="invalid_symbol")
                )
                return {
                    "success": False,
                    "error": f"Invalid symbol: {symbol}",
                    "symbol": symbol,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # Fetch data with circuit breaker protection
            raw_data = await self.circuit_breaker.call(
                self._fetch_data, symbol, data_type, granularity
            )
            
            if not raw_data.success:
                self.metrics_collector.increment_counter(IngestionMetrics.REQUESTS_FAILED, 1, tags)
                return {
                    "success": False,
                    "error": raw_data.error_message,
                    "symbol": symbol,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # Transform data
            async with TimingContextManager(
                self.metrics_collector, 
                IngestionMetrics.TRANSFORMATION_DURATION, 
                tags
            ):
                transformed_data = await self.data_transformer.transform(raw_data)
            
            if not transformed_data.get("success", False):
                self.metrics_collector.increment_counter(
                    IngestionMetrics.TRANSFORMATION_ERRORS, 1, tags
                )
                return transformed_data
            
            # Store data if sink is available
            if self.data_sink:
                async with TimingContextManager(
                    self.metrics_collector, 
                    IngestionMetrics.STORAGE_DURATION, 
                    tags
                ):
                    storage_success = await self.data_sink.write_data(transformed_data)
                    if not storage_success:
                        self.logger.warning(f"Failed to store data for {symbol}")
            
            # Update metrics
            self.metrics_collector.increment_counter(IngestionMetrics.REQUESTS_SUCCESS, 1, tags)
            self.metrics_collector.increment_counter(
                IngestionMetrics.DATA_POINTS_INGESTED, 
                len(transformed_data.get("records", [])), 
                tags
            )
            
            duration = time.time() - start_time
            self.metrics_collector.record_timing(IngestionMetrics.REQUEST_DURATION, duration, tags)
            
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Error ingesting data for {symbol}: {e}")
            self.metrics_collector.increment_counter(IngestionMetrics.REQUESTS_FAILED, 1, tags)
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def ingest_batch_data(self, symbols: List[str], data_type: MarketDataType) -> Dict[str, Any]:
        """Ingest data for multiple symbols."""
        start_time = time.time()
        tags = IngestionMetrics.create_tags(data_type=data_type.value)
        
        try:
            self.metrics_collector.increment_counter(
                IngestionMetrics.REQUESTS_TOTAL, len(symbols), tags
            )
            
            # Process symbols concurrently but limit concurrency
            semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
            
            async def process_symbol(symbol: str):
                async with semaphore:
                    return await self.ingest_symbol_data(symbol, data_type)
            
            results = await asyncio.gather(
                *[process_symbol(symbol) for symbol in symbols],
                return_exceptions=True
            )
            
            # Aggregate results
            successful = sum(1 for result in results if isinstance(result, dict) and result.get("success", False))
            failed = len(symbols) - successful
            
            batch_result = {
                "success": True,
                "total_symbols": len(symbols),
                "successful": successful,
                "failed": failed,
                "results": results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration": time.time() - start_time
            }
            
            return batch_result
            
        except Exception as e:
            self.logger.error(f"Error in batch ingestion: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def start_continuous_ingestion(self, symbols: List[str], interval: int = 300) -> str:
        """Start continuous data ingestion for symbols."""
        try:
            # Schedule jobs for each symbol
            job_ids = []
            for symbol in symbols:
                job_id = await self.scheduler.schedule_ingestion(
                    symbol=symbol,
                    data_type=MarketDataType.QUOTE,  # Default to quotes for continuous
                    interval=interval
                )
                job_ids.append(job_id)
            
            # Start background processing if not already running
            if self._background_task is None or self._background_task.done():
                self._background_task = asyncio.create_task(self._process_scheduled_jobs())
            
            batch_job_id = str(uuid.uuid4())
            self.logger.info(f"Started continuous ingestion batch {batch_job_id} for {len(symbols)} symbols")
            
            return batch_job_id
            
        except Exception as e:
            self.logger.error(f"Error starting continuous ingestion: {e}")
            raise
    
    async def stop_continuous_ingestion(self, job_id: str) -> bool:
        """Stop continuous data ingestion."""
        try:
            success = await self.scheduler.cancel_ingestion(job_id)
            if success:
                self.logger.info(f"Stopped continuous ingestion {job_id}")
            return success
        except Exception as e:
            self.logger.error(f"Error stopping continuous ingestion: {e}")
            return False
    
    async def get_ingestion_status(self) -> Dict[str, Any]:
        """Get current ingestion status and statistics."""
        try:
            active_jobs = await self.scheduler.get_active_jobs()
            metrics_summary = self.metrics_collector.get_metrics_summary()
            
            status = {
                "active_jobs": len(active_jobs),
                "running_jobs": len(self._running_jobs),
                "circuit_breaker_state": self.circuit_breaker.get_state(),
                "circuit_breaker_failures": self.circuit_breaker.get_failure_count(),
                "background_task_running": self._background_task is not None and not self._background_task.done(),
                "jobs": active_jobs,
                "metrics": metrics_summary,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting ingestion status: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health = {
            "overall": "healthy",
            "components": {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Check data sink health if available
            if self.data_sink:
                sink_healthy = await self.data_sink.health_check()
                health["components"]["data_sink"] = "healthy" if sink_healthy else "unhealthy"
                if not sink_healthy:
                    health["overall"] = "degraded"
            
            # Check circuit breaker state
            cb_state = self.circuit_breaker.get_state()
            health["components"]["circuit_breaker"] = cb_state
            if cb_state == "open":
                health["overall"] = "degraded"
            
            # Check scheduler
            active_jobs = await self.scheduler.get_active_jobs()
            health["components"]["scheduler"] = {
                "status": "healthy",
                "active_jobs": len(active_jobs)
            }
            
            # Check background task
            health["components"]["background_processor"] = {
                "running": self._background_task is not None and not self._background_task.done()
            }
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            health["overall"] = "unhealthy"
            health["error"] = str(e)
        
        return health
    
    async def _fetch_data(
        self, 
        symbol: str, 
        data_type: MarketDataType, 
        granularity: Optional[DataGranularity]
    ) -> MarketDataResponse:
        """Fetch data from data source."""
        if data_type == MarketDataType.QUOTE:
            return await self.data_source.get_quote(symbol)
        elif data_type == MarketDataType.DAILY:
            return await self.data_source.get_daily_data(symbol)
        elif data_type == MarketDataType.INTRADAY:
            interval = "5min"  # Default
            if granularity:
                interval = granularity.value
            return await self.data_source.get_intraday_data(symbol, interval)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    async def _process_scheduled_jobs(self):
        """Background task to process scheduled jobs."""
        self.logger.info("Started background job processor")
        
        while not self._shutdown_event.is_set():
            try:
                ready_jobs = await self.scheduler.get_ready_jobs()
                
                if ready_jobs:
                    # Limit concurrent job execution
                    available_slots = max(0, self.max_concurrent_jobs - len(self._running_jobs))
                    jobs_to_run = ready_jobs[:available_slots]
                    
                    for job in jobs_to_run:
                        if job.job_id not in self._running_jobs:
                            asyncio.create_task(self._execute_job(job))
                
                # Update metrics
                self.metrics_collector.set_gauge(
                    IngestionMetrics.ACTIVE_JOBS, 
                    len(self._running_jobs)
                )
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in background job processor: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    async def _execute_job(self, job: IngestionJob):
        """Execute a single ingestion job."""
        self._running_jobs.add(job.job_id)
        
        try:
            job.status = JobStatus.RUNNING
            job.last_run = datetime.now(timezone.utc)
            
            result = await self.ingest_symbol_data(
                job.symbol, job.data_type, job.granularity
            )
            
            if result.get("success", False):
                job.failure_count = 0
                job.status = JobStatus.PENDING  # Ready for next run
            else:
                job.failure_count += 1
                if job.failure_count >= job.max_failures:
                    job.status = JobStatus.FAILED
                    self.logger.error(f"Job {job.job_id} failed permanently after {job.failure_count} failures")
            
            # Schedule next run
            if job.status == JobStatus.PENDING:
                next_run = datetime.now(timezone.utc)
                next_run = next_run.replace(
                    second=(next_run.second + job.interval) % 60,
                    microsecond=0
                )
                if next_run.second < job.interval:
                    next_run = next_run.replace(minute=next_run.minute + 1)
                job.next_run = next_run
            
        except Exception as e:
            self.logger.error(f"Error executing job {job.job_id}: {e}")
            job.failure_count += 1
            job.status = JobStatus.FAILED if job.failure_count >= job.max_failures else JobStatus.PENDING
            
        finally:
            self._running_jobs.discard(job.job_id)
    
    async def shutdown(self):
        """Shutdown the ingestion service."""
        self.logger.info("Shutting down ingestion service")
        self._shutdown_event.set()
        
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        
        # Wait for running jobs to complete (with timeout)
        if self._running_jobs:
            self.logger.info(f"Waiting for {len(self._running_jobs)} running jobs to complete")
            timeout = 30  # seconds
            start_time = time.time()
            
            while self._running_jobs and (time.time() - start_time) < timeout:
                await asyncio.sleep(1)
        
        self.logger.info("Ingestion service shutdown complete")