"""
Metrics collection implementations for data ingestion monitoring.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from collections import defaultdict, deque
from threading import Lock
import asyncio

from .interfaces import IMetricsCollector


class InMemoryMetricsCollector(IMetricsCollector):
    """In-memory metrics collector for development and testing."""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.logger = logging.getLogger(__name__)
        self._lock = Lock()
        
        # Metrics storage
        self._counters = defaultdict(float)
        self._gauges = defaultdict(float)
        self._timings = defaultdict(lambda: deque(maxlen=max_history_size))
        self._histograms = defaultdict(lambda: deque(maxlen=max_history_size))
        
        # Metadata
        self._counter_tags = defaultdict(dict)
        self._gauge_tags = defaultdict(dict)
        self._timing_tags = defaultdict(dict)
        self._histogram_tags = defaultdict(dict)
        
        self._last_updated = defaultdict(lambda: datetime.now(timezone.utc))
    
    def increment_counter(self, metric_name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            key = self._build_metric_key(metric_name, tags)
            self._counters[key] += value
            if tags:
                self._counter_tags[key] = tags.copy()
            self._last_updated[key] = datetime.now(timezone.utc)
    
    def set_gauge(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        with self._lock:
            key = self._build_metric_key(metric_name, tags)
            self._gauges[key] = value
            if tags:
                self._gauge_tags[key] = tags.copy()
            self._last_updated[key] = datetime.now(timezone.utc)
    
    def record_timing(self, metric_name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record timing metric."""
        with self._lock:
            key = self._build_metric_key(metric_name, tags)
            self._timings[key].append({
                "value": duration,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            if tags:
                self._timing_tags[key] = tags.copy()
            self._last_updated[key] = datetime.now(timezone.utc)
    
    def record_histogram(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record histogram metric."""
        with self._lock:
            key = self._build_metric_key(metric_name, tags)
            self._histograms[key].append({
                "value": value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            if tags:
                self._histogram_tags[key] = tags.copy()
            self._last_updated[key] = datetime.now(timezone.utc)
    
    def _build_metric_key(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Build metric key with tags."""
        if not tags:
            return metric_name
        
        tag_pairs = sorted(tags.items())
        tag_string = ",".join([f"{k}={v}" for k, v in tag_pairs])
        return f"{metric_name}[{tag_string}]"
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        with self._lock:
            summary = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "timings": {},
                "histograms": {},
                "collection_time": datetime.now(timezone.utc).isoformat()
            }
            
            # Summarize timings
            for key, timing_data in self._timings.items():
                if timing_data:
                    values = [item["value"] for item in timing_data]
                    summary["timings"][key] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "last_updated": self._last_updated[key].isoformat()
                    }
            
            # Summarize histograms
            for key, hist_data in self._histograms.items():
                if hist_data:
                    values = [item["value"] for item in hist_data]
                    summary["histograms"][key] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "last_updated": self._last_updated[key].isoformat()
                    }
            
            return summary
    
    def reset_metrics(self):
        """Reset all collected metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._timings.clear()
            self._histograms.clear()
            self._counter_tags.clear()
            self._gauge_tags.clear()
            self._timing_tags.clear()
            self._histogram_tags.clear()
            self._last_updated.clear()


class PrometheusMetricsCollector(IMetricsCollector):
    """Prometheus metrics collector for production use."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._metrics_registry = {}
        
        try:
            from prometheus_client import Counter, Gauge, Histogram, Summary
            self.Counter = Counter
            self.Gauge = Gauge
            self.Histogram = Histogram
            self.Summary = Summary
            self._prometheus_available = True
        except ImportError:
            self.logger.warning("Prometheus client not available, using in-memory fallback")
            self._prometheus_available = False
            self._fallback_collector = InMemoryMetricsCollector()
    
    def increment_counter(self, metric_name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        if not self._prometheus_available:
            return self._fallback_collector.increment_counter(metric_name, value, tags)
        
        try:
            counter = self._get_or_create_counter(metric_name, tags)
            if tags:
                counter.labels(**tags).inc(value)
            else:
                counter.inc(value)
        except Exception as e:
            self.logger.error(f"Error recording counter {metric_name}: {e}")
    
    def set_gauge(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        if not self._prometheus_available:
            return self._fallback_collector.set_gauge(metric_name, value, tags)
        
        try:
            gauge = self._get_or_create_gauge(metric_name, tags)
            if tags:
                gauge.labels(**tags).set(value)
            else:
                gauge.set(value)
        except Exception as e:
            self.logger.error(f"Error setting gauge {metric_name}: {e}")
    
    def record_timing(self, metric_name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record timing metric."""
        if not self._prometheus_available:
            return self._fallback_collector.record_timing(metric_name, duration, tags)
        
        try:
            histogram = self._get_or_create_histogram(metric_name, tags)
            if tags:
                histogram.labels(**tags).observe(duration)
            else:
                histogram.observe(duration)
        except Exception as e:
            self.logger.error(f"Error recording timing {metric_name}: {e}")
    
    def record_histogram(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record histogram metric."""
        if not self._prometheus_available:
            return self._fallback_collector.record_histogram(metric_name, value, tags)
        
        try:
            histogram = self._get_or_create_histogram(metric_name, tags)
            if tags:
                histogram.labels(**tags).observe(value)
            else:
                histogram.observe(value)
        except Exception as e:
            self.logger.error(f"Error recording histogram {metric_name}: {e}")
    
    def _get_or_create_counter(self, metric_name: str, tags: Optional[Dict[str, str]] = None):
        """Get or create Prometheus Counter."""
        key = f"counter_{metric_name}"
        if key not in self._metrics_registry:
            label_names = list(tags.keys()) if tags else []
            self._metrics_registry[key] = self.Counter(
                metric_name, 
                f"Counter metric: {metric_name}",
                labelnames=label_names
            )
        return self._metrics_registry[key]
    
    def _get_or_create_gauge(self, metric_name: str, tags: Optional[Dict[str, str]] = None):
        """Get or create Prometheus Gauge."""
        key = f"gauge_{metric_name}"
        if key not in self._metrics_registry:
            label_names = list(tags.keys()) if tags else []
            self._metrics_registry[key] = self.Gauge(
                metric_name,
                f"Gauge metric: {metric_name}",
                labelnames=label_names
            )
        return self._metrics_registry[key]
    
    def _get_or_create_histogram(self, metric_name: str, tags: Optional[Dict[str, str]] = None):
        """Get or create Prometheus Histogram."""
        key = f"histogram_{metric_name}"
        if key not in self._metrics_registry:
            label_names = list(tags.keys()) if tags else []
            self._metrics_registry[key] = self.Histogram(
                metric_name,
                f"Histogram metric: {metric_name}",
                labelnames=label_names
            )
        return self._metrics_registry[key]


class TimingContextManager:
    """Context manager for timing operations."""
    
    def __init__(self, metrics_collector: IMetricsCollector, metric_name: str, tags: Optional[Dict[str, str]] = None):
        self.metrics_collector = metrics_collector
        self.metric_name = metric_name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.metrics_collector.record_timing(self.metric_name, duration, self.tags)
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.metrics_collector.record_timing(self.metric_name, duration, self.tags)


class IngestionMetrics:
    """Pre-defined metrics for data ingestion monitoring."""
    
    # Counter metrics
    REQUESTS_TOTAL = "ingestion_requests_total"
    REQUESTS_SUCCESS = "ingestion_requests_success_total"
    REQUESTS_FAILED = "ingestion_requests_failed_total"
    DATA_POINTS_INGESTED = "ingestion_data_points_total"
    VALIDATION_ERRORS = "ingestion_validation_errors_total"
    TRANSFORMATION_ERRORS = "ingestion_transformation_errors_total"
    
    # Gauge metrics
    ACTIVE_SYMBOLS = "ingestion_active_symbols"
    QUEUE_SIZE = "ingestion_queue_size"
    ACTIVE_JOBS = "ingestion_active_jobs"
    CIRCUIT_BREAKER_STATE = "ingestion_circuit_breaker_state"
    
    # Timing/Histogram metrics
    REQUEST_DURATION = "ingestion_request_duration_seconds"
    TRANSFORMATION_DURATION = "ingestion_transformation_duration_seconds"
    VALIDATION_DURATION = "ingestion_validation_duration_seconds"
    STORAGE_DURATION = "ingestion_storage_duration_seconds"
    
    # Tags
    SYMBOL_TAG = "symbol"
    DATA_TYPE_TAG = "data_type"
    SOURCE_TAG = "source"
    ERROR_TYPE_TAG = "error_type"
    JOB_ID_TAG = "job_id"
    
    @staticmethod
    def create_tags(symbol: str = None, data_type: str = None, source: str = None, 
                   error_type: str = None, job_id: str = None) -> Dict[str, str]:
        """Create metrics tags dictionary."""
        tags = {}
        if symbol:
            tags[IngestionMetrics.SYMBOL_TAG] = symbol
        if data_type:
            tags[IngestionMetrics.DATA_TYPE_TAG] = data_type
        if source:
            tags[IngestionMetrics.SOURCE_TAG] = source
        if error_type:
            tags[IngestionMetrics.ERROR_TYPE_TAG] = error_type
        if job_id:
            tags[IngestionMetrics.JOB_ID_TAG] = job_id
        return tags


def create_metrics_collector(collector_type: str = "inmemory") -> IMetricsCollector:
    """Factory function to create metrics collector."""
    if collector_type.lower() == "prometheus":
        return PrometheusMetricsCollector()
    else:
        return InMemoryMetricsCollector()


def timing_decorator(metrics_collector: IMetricsCollector, metric_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator for timing function execution."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                async with TimingContextManager(metrics_collector, metric_name, tags):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with TimingContextManager(metrics_collector, metric_name, tags):
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator