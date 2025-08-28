"""
Comprehensive performance profiling and monitoring utilities.
Following SOLID principles and TDD methodology.
"""

import asyncio
import logging
import time
import sys
import os
import functools
import tracemalloc
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextlib import contextmanager, asynccontextmanager
from collections import defaultdict
import threading
import json

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import cProfile
    import pstats
    import io
    HAS_PROFILING = True
except ImportError:
    HAS_PROFILING = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Represents a single performance metric."""
    name: str
    value: Union[float, int]
    unit: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EndpointProfile:
    """Profile data for an API endpoint."""
    endpoint: str
    method: str
    total_requests: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    error_count: int = 0
    percentiles: Dict[str, float] = field(default_factory=dict)
    durations: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    cpu_usage: List[float] = field(default_factory=list)

    @property
    def avg_duration(self) -> float:
        """Calculate average duration."""
        return self.total_duration / max(1, self.total_requests)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.total_requests - self.error_count) / self.total_requests * 100.0

    def add_request(self, duration: float, error: bool = False, memory_mb: float = 0.0, cpu_percent: float = 0.0):
        """Add request performance data."""
        self.total_requests += 1
        self.total_duration += duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        
        if error:
            self.error_count += 1
        
        self.durations.append(duration)
        if memory_mb > 0:
            self.memory_usage.append(memory_mb)
        if cpu_percent > 0:
            self.cpu_usage.append(cpu_percent)
        
        # Keep only last 1000 measurements to prevent memory bloat
        if len(self.durations) > 1000:
            self.durations = self.durations[-1000:]
        if len(self.memory_usage) > 1000:
            self.memory_usage = self.memory_usage[-1000:]
        if len(self.cpu_usage) > 1000:
            self.cpu_usage = self.cpu_usage[-1000:]
        
        # Calculate percentiles if we have enough data
        if len(self.durations) >= 10:
            sorted_durations = sorted(self.durations)
            n = len(sorted_durations)
            self.percentiles = {
                "p50": sorted_durations[int(n * 0.5)],
                "p90": sorted_durations[int(n * 0.9)],
                "p95": sorted_durations[int(n * 0.95)],
                "p99": sorted_durations[int(n * 0.99)],
            }


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self):
        self.process = psutil.Process() if HAS_PSUTIL else None
        self.baseline_memory = None
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self.metrics: List[PerformanceMetric] = []
        self._lock = asyncio.Lock()
    
    async def start_monitoring(self, interval: float = 1.0):
        """Start continuous system monitoring."""
        if not HAS_PSUTIL:
            logger.warning("psutil not available - system monitoring disabled")
            return
        
        self._monitoring = True
        self.baseline_memory = self.get_memory_usage()
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info("Started system monitoring")
    
    async def stop_monitoring(self):
        """Stop system monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped system monitoring")
    
    async def _monitor_loop(self, interval: float):
        """Continuous monitoring loop."""
        while self._monitoring:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval * 2)  # Back off on error
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics."""
        async with self._lock:
            timestamp = datetime.now(timezone.utc)
            
            # Memory metrics
            memory_mb = self.get_memory_usage()
            self.metrics.append(PerformanceMetric(
                name="memory_usage_mb",
                value=memory_mb,
                unit="MB",
                timestamp=timestamp
            ))
            
            # CPU metrics
            cpu_percent = self.get_cpu_usage()
            self.metrics.append(PerformanceMetric(
                name="cpu_usage_percent",
                value=cpu_percent,
                unit="percent",
                timestamp=timestamp
            ))
            
            # Keep only last 1000 metrics
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-1000:]
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not self.process:
            return 0.0
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if not self.process:
            return 0.0
        try:
            return self.process.cpu_percent()
        except Exception:
            return 0.0
    
    def get_memory_growth(self) -> float:
        """Get memory growth since baseline in MB."""
        if self.baseline_memory is None:
            return 0.0
        current = self.get_memory_usage()
        return current - self.baseline_memory
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        async with self._lock:
            if not self.metrics:
                return {}
            
            memory_values = [m.value for m in self.metrics if m.name == "memory_usage_mb"]
            cpu_values = [m.value for m in self.metrics if m.name == "cpu_usage_percent"]
            
            summary = {
                "memory": {
                    "current_mb": self.get_memory_usage(),
                    "baseline_mb": self.baseline_memory or 0.0,
                    "growth_mb": self.get_memory_growth(),
                    "avg_mb": sum(memory_values) / len(memory_values) if memory_values else 0.0,
                    "max_mb": max(memory_values) if memory_values else 0.0,
                },
                "cpu": {
                    "current_percent": self.get_cpu_usage(),
                    "avg_percent": sum(cpu_values) / len(cpu_values) if cpu_values else 0.0,
                    "max_percent": max(cpu_values) if cpu_values else 0.0,
                },
                "metrics_count": len(self.metrics),
                "monitoring_active": self._monitoring
            }
            
            return summary


class DatabaseProfiler:
    """Database query performance profiler."""
    
    def __init__(self):
        self.query_profiles: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_duration": 0.0,
            "min_duration": float('inf'),
            "max_duration": 0.0,
            "durations": []
        })
        self._lock = asyncio.Lock()
    
    @asynccontextmanager
    async def profile_query(self, query_type: str, query_signature: str = ""):
        """Context manager to profile database queries."""
        start_time = time.time()
        try:
            yield
            success = True
        except Exception as e:
            success = False
            raise
        finally:
            duration = time.time() - start_time
            await self._record_query(query_type, query_signature, duration, success)
    
    async def _record_query(self, query_type: str, query_signature: str, duration: float, success: bool):
        """Record query performance data."""
        async with self._lock:
            key = f"{query_type}:{query_signature}" if query_signature else query_type
            profile = self.query_profiles[key]
            
            profile["count"] += 1
            profile["total_duration"] += duration
            profile["min_duration"] = min(profile["min_duration"], duration)
            profile["max_duration"] = max(profile["max_duration"], duration)
            profile["durations"].append(duration)
            
            if not success:
                profile["error_count"] = profile.get("error_count", 0) + 1
            
            # Keep only last 1000 measurements
            if len(profile["durations"]) > 1000:
                profile["durations"] = profile["durations"][-1000:]
    
    async def get_query_stats(self) -> Dict[str, Any]:
        """Get database query statistics."""
        async with self._lock:
            stats = {}
            
            for key, profile in self.query_profiles.items():
                durations = profile["durations"]
                if durations:
                    sorted_durations = sorted(durations)
                    n = len(sorted_durations)
                    percentiles = {
                        "p50": sorted_durations[int(n * 0.5)],
                        "p90": sorted_durations[int(n * 0.9)],
                        "p95": sorted_durations[int(n * 0.95)],
                        "p99": sorted_durations[int(n * 0.99)],
                    } if n >= 10 else {}
                else:
                    percentiles = {}
                
                stats[key] = {
                    "count": profile["count"],
                    "avg_duration": profile["total_duration"] / max(1, profile["count"]),
                    "min_duration": profile["min_duration"] if profile["min_duration"] != float('inf') else 0.0,
                    "max_duration": profile["max_duration"],
                    "error_count": profile.get("error_count", 0),
                    "error_rate": (profile.get("error_count", 0) / max(1, profile["count"])) * 100.0,
                    "percentiles": percentiles
                }
            
            return stats


class PerformanceProfiler:
    """Main performance profiler orchestrating all monitoring components."""
    
    def __init__(self, enable_memory_tracking: bool = True):
        self.system_monitor = SystemMonitor()
        self.db_profiler = DatabaseProfiler()
        self.endpoint_profiles: Dict[str, EndpointProfile] = {}
        self.enable_memory_tracking = enable_memory_tracking
        self._lock = asyncio.Lock()
        self._profiling_active = False
        
        if enable_memory_tracking:
            tracemalloc.start()
            logger.info("Memory tracking enabled")
    
    async def start_profiling(self):
        """Start performance profiling."""
        self._profiling_active = True
        await self.system_monitor.start_monitoring()
        logger.info("Performance profiling started")
    
    async def stop_profiling(self):
        """Stop performance profiling."""
        self._profiling_active = False
        await self.system_monitor.stop_monitoring()
        logger.info("Performance profiling stopped")
    
    @asynccontextmanager
    async def profile_endpoint(self, endpoint: str, method: str = "GET"):
        """Profile an API endpoint."""
        start_time = time.time()
        start_memory = self.system_monitor.get_memory_usage()
        start_cpu = self.system_monitor.get_cpu_usage()
        
        try:
            yield
            error = False
        except Exception as e:
            error = True
            raise
        finally:
            duration = time.time() - start_time
            end_memory = self.system_monitor.get_memory_usage()
            end_cpu = self.system_monitor.get_cpu_usage()
            
            memory_delta = end_memory - start_memory
            cpu_avg = (start_cpu + end_cpu) / 2
            
            await self._record_endpoint_performance(
                endpoint, method, duration, error, memory_delta, cpu_avg
            )
    
    async def _record_endpoint_performance(
        self, 
        endpoint: str, 
        method: str, 
        duration: float, 
        error: bool,
        memory_delta: float,
        cpu_avg: float
    ):
        """Record endpoint performance data."""
        async with self._lock:
            key = f"{method}:{endpoint}"
            
            if key not in self.endpoint_profiles:
                self.endpoint_profiles[key] = EndpointProfile(
                    endpoint=endpoint,
                    method=method
                )
            
            self.endpoint_profiles[key].add_request(
                duration=duration,
                error=error,
                memory_mb=memory_delta,
                cpu_percent=cpu_avg
            )
    
    def profile_function(self, func_name: Optional[str] = None):
        """Decorator to profile function performance."""
        def decorator(func: Callable):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    finally:
                        duration = time.time() - start_time
                        logger.debug(f"Function {name} took {duration:.4f}s")
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        return result
                    finally:
                        duration = time.time() - start_time
                        logger.debug(f"Function {name} took {duration:.4f}s")
                return sync_wrapper
        
        return decorator
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        async with self._lock:
            system_summary = await self.system_monitor.get_metrics_summary()
            db_stats = await self.db_profiler.get_query_stats()
            
            endpoint_summary = {}
            for key, profile in self.endpoint_profiles.items():
                endpoint_summary[key] = {
                    "total_requests": profile.total_requests,
                    "avg_duration_ms": profile.avg_duration * 1000,
                    "min_duration_ms": profile.min_duration * 1000,
                    "max_duration_ms": profile.max_duration * 1000,
                    "error_count": profile.error_count,
                    "success_rate": profile.success_rate,
                    "percentiles": {k: v * 1000 for k, v in profile.percentiles.items()},
                    "avg_memory_delta_mb": sum(profile.memory_usage) / len(profile.memory_usage) if profile.memory_usage else 0.0,
                    "avg_cpu_percent": sum(profile.cpu_usage) / len(profile.cpu_usage) if profile.cpu_usage else 0.0
                }
            
            report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "profiling_active": self._profiling_active,
                "system": system_summary,
                "database": db_stats,
                "endpoints": endpoint_summary,
                "memory_tracking": {
                    "enabled": self.enable_memory_tracking,
                    "tracemalloc_active": tracemalloc.is_tracing() if self.enable_memory_tracking else False
                }
            }
            
            return report
    
    async def export_report(self, file_path: str):
        """Export performance report to file."""
        report = await self.get_performance_report()
        
        try:
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Performance report exported to {file_path}")
        except Exception as e:
            logger.error(f"Failed to export report to {file_path}: {e}")
            raise


# Global profiler instance
_profiler: Optional[PerformanceProfiler] = None


def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler()
    return _profiler


async def init_profiler():
    """Initialize performance profiler."""
    profiler = get_performance_profiler()
    await profiler.start_profiling()


async def shutdown_profiler():
    """Shutdown performance profiler."""
    global _profiler
    if _profiler:
        await _profiler.stop_profiling()


# Convenience functions for FastAPI integration
def profile_endpoint(endpoint_name: str, method: str = "GET"):
    """Decorator for profiling FastAPI endpoints."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            profiler = get_performance_profiler()
            async with profiler.profile_endpoint(endpoint_name, method):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def profile_database_query(query_type: str, query_signature: str = ""):
    """Sync context manager for profiling database queries."""
    profiler = get_performance_profiler()
    start_time = time.time()
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        # Store for async processing
        asyncio.create_task(
            profiler.db_profiler._record_query(query_type, query_signature, duration, True)
        )


class MemoryProfiler:
    """Memory usage profiler and leak detector."""
    
    def __init__(self):
        self.snapshots: List[Any] = []
        self.baseline_snapshot: Optional[Any] = None
    
    def take_snapshot(self, description: str = ""):
        """Take memory snapshot."""
        if not tracemalloc.is_tracing():
            logger.warning("tracemalloc not active - cannot take snapshot")
            return None
        
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append({
            "snapshot": snapshot,
            "description": description,
            "timestamp": datetime.now(timezone.utc)
        })
        
        # Keep only last 10 snapshots
        if len(self.snapshots) > 10:
            self.snapshots = self.snapshots[-10:]
        
        if self.baseline_snapshot is None:
            self.baseline_snapshot = snapshot
        
        return snapshot
    
    def set_baseline(self):
        """Set current memory state as baseline."""
        if tracemalloc.is_tracing():
            self.baseline_snapshot = tracemalloc.take_snapshot()
            logger.info("Memory baseline set")
    
    def get_memory_growth(self, top_limit: int = 10) -> Dict[str, Any]:
        """Get memory growth analysis."""
        if not self.baseline_snapshot or not self.snapshots:
            return {"error": "No baseline or snapshots available"}
        
        current_snapshot = self.snapshots[-1]["snapshot"]
        top_stats = current_snapshot.compare_to(self.baseline_snapshot, 'lineno')
        
        growth_analysis = {
            "total_growth_mb": sum(stat.size_diff for stat in top_stats) / 1024 / 1024,
            "top_growth": []
        }
        
        for stat in top_stats[:top_limit]:
            growth_analysis["top_growth"].append({
                "file": stat.traceback.format()[0] if stat.traceback else "unknown",
                "size_diff_mb": stat.size_diff / 1024 / 1024,
                "count_diff": stat.count_diff,
                "current_size_mb": stat.size / 1024 / 1024
            })
        
        return growth_analysis


# Global memory profiler
_memory_profiler: Optional[MemoryProfiler] = None


def get_memory_profiler() -> MemoryProfiler:
    """Get global memory profiler instance."""
    global _memory_profiler
    if _memory_profiler is None:
        _memory_profiler = MemoryProfiler()
    return _memory_profiler