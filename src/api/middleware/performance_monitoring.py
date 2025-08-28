"""
Performance monitoring middleware for FastAPI applications.
Integrates with the performance profiler for comprehensive monitoring.
"""

import asyncio
import time
import logging
from typing import Callable, Dict, Any, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import HTTPException
from datetime import datetime, timezone

from src.utils.performance_profiler import get_performance_profiler, get_memory_profiler

logger = logging.getLogger(__name__)


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware to monitor API endpoint performance automatically.
    Integrates with the global performance profiler.
    """
    
    def __init__(
        self,
        app,
        enable_detailed_logging: bool = False,
        slow_request_threshold: float = 1.0,  # seconds
        memory_snapshot_interval: int = 100,  # requests
        exclude_paths: Optional[set] = None
    ):
        super().__init__(app)
        self.enable_detailed_logging = enable_detailed_logging
        self.slow_request_threshold = slow_request_threshold
        self.memory_snapshot_interval = memory_snapshot_interval
        self.exclude_paths = exclude_paths or {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}
        self.request_count = 0
        self.profiler = get_performance_profiler()
        self.memory_profiler = get_memory_profiler()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with performance monitoring."""
        # Skip monitoring for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        start_time = time.time()
        start_memory = 0
        exception_occurred = False
        response = None
        
        # Track memory usage
        try:
            import psutil
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            pass
        
        # Increment request count
        self.request_count += 1
        
        # Take memory snapshot at intervals
        if self.request_count % self.memory_snapshot_interval == 0:
            self.memory_profiler.take_snapshot(f"request_{self.request_count}")
        
        try:
            # Profile the request
            async with self.profiler.profile_endpoint(
                endpoint=request.url.path,
                method=request.method
            ):
                response = await call_next(request)
            
        except Exception as e:
            exception_occurred = True
            # Still record the performance data for failed requests
            duration = time.time() - start_time
            await self._log_request_performance(
                request, None, duration, start_memory, exception_occurred
            )
            raise
        
        # Calculate final metrics
        duration = time.time() - start_time
        await self._log_request_performance(
            request, response, duration, start_memory, exception_occurred
        )
        
        return response
    
    async def _log_request_performance(
        self,
        request: Request,
        response: Optional[Response],
        duration: float,
        start_memory: float,
        exception_occurred: bool
    ):
        """Log request performance data."""
        try:
            # Get current memory usage
            end_memory = 0
            try:
                import psutil
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
            except ImportError:
                pass
            
            memory_delta = end_memory - start_memory if start_memory > 0 else 0
            
            # Determine status code
            status_code = response.status_code if response else 500
            
            # Basic performance logging
            log_data = {
                "method": request.method,
                "path": request.url.path,
                "duration_ms": duration * 1000,
                "status_code": status_code,
                "memory_delta_mb": memory_delta,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown")[:100],
                "content_length": response.headers.get("content-length", "0") if response else "0"
            }
            
            # Log slow requests
            if duration > self.slow_request_threshold:
                logger.warning(f"Slow request detected: {log_data}")
            elif self.enable_detailed_logging:
                logger.info(f"Request completed: {log_data}")
            
            # Additional monitoring for errors
            if exception_occurred or (response and status_code >= 400):
                logger.warning(f"Request error: {log_data}")
        
        except Exception as e:
            logger.error(f"Error logging request performance: {e}")


class DatabaseQueryMiddleware:
    """
    Context manager to monitor database query performance.
    Use this in database operations for automatic profiling.
    """
    
    def __init__(self, query_type: str, query_signature: str = ""):
        self.query_type = query_type
        self.query_signature = query_signature
        self.profiler = get_performance_profiler()
        self.start_time = None
    
    async def __aenter__(self):
        """Enter async context."""
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context and record metrics."""
        if self.start_time:
            duration = time.time() - self.start_time
            success = exc_type is None
            
            await self.profiler.db_profiler._record_query(
                self.query_type,
                self.query_signature,
                duration,
                success
            )


class MemoryMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware to monitor memory usage and detect potential leaks.
    """
    
    def __init__(
        self,
        app,
        memory_threshold_mb: float = 500.0,
        snapshot_interval: int = 50,
        leak_detection_enabled: bool = True
    ):
        super().__init__(app)
        self.memory_threshold_mb = memory_threshold_mb
        self.snapshot_interval = snapshot_interval
        self.leak_detection_enabled = leak_detection_enabled
        self.request_count = 0
        self.memory_profiler = get_memory_profiler()
        self.baseline_set = False
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor memory usage during request processing."""
        # Set baseline on first request
        if not self.baseline_set:
            self.memory_profiler.set_baseline()
            self.baseline_set = True
        
        self.request_count += 1
        
        # Take snapshot at intervals for leak detection
        if (self.leak_detection_enabled and 
            self.request_count % self.snapshot_interval == 0):
            
            snapshot = self.memory_profiler.take_snapshot(
                f"request_interval_{self.request_count}"
            )
            
            # Check for memory growth
            if self.request_count > self.snapshot_interval:
                growth = self.memory_profiler.get_memory_growth(top_limit=5)
                
                if growth.get("total_growth_mb", 0) > self.memory_threshold_mb:
                    logger.warning(
                        f"Significant memory growth detected: "
                        f"{growth['total_growth_mb']:.2f} MB after {self.request_count} requests"
                    )
                    
                    # Log top memory consumers
                    for item in growth.get("top_growth", []):
                        logger.warning(f"Memory growth: {item}")
        
        # Process request normally
        response = await call_next(request)
        
        return response


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """
    Circuit breaker middleware to prevent cascade failures.
    """
    
    def __init__(
        self,
        app,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        error_rate_threshold: float = 0.5,
        min_requests: int = 10
    ):
        super().__init__(app)
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.error_rate_threshold = error_rate_threshold
        self.min_requests = min_requests
        
        # Circuit breaker state per endpoint
        self.circuit_states: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply circuit breaker logic to requests."""
        endpoint_key = f"{request.method}:{request.url.path}"
        
        # Check circuit breaker state
        async with self._lock:
            circuit = self.circuit_states.get(endpoint_key, {
                "state": "closed",  # closed, open, half-open
                "failure_count": 0,
                "success_count": 0,
                "last_failure_time": 0,
                "total_requests": 0
            })
            
            # Check if circuit should be opened
            if circuit["state"] == "closed":
                if (circuit["total_requests"] >= self.min_requests and
                    circuit["failure_count"] / circuit["total_requests"] > self.error_rate_threshold):
                    circuit["state"] = "open"
                    circuit["last_failure_time"] = time.time()
                    logger.warning(f"Circuit breaker opened for {endpoint_key}")
            
            # Check if circuit should move to half-open
            elif circuit["state"] == "open":
                if time.time() - circuit["last_failure_time"] > self.recovery_timeout:
                    circuit["state"] = "half-open"
                    logger.info(f"Circuit breaker moving to half-open for {endpoint_key}")
                else:
                    # Circuit is still open - reject request
                    logger.warning(f"Circuit breaker rejecting request for {endpoint_key}")
                    return Response(
                        content="Service temporarily unavailable - circuit breaker open",
                        status_code=503,
                        headers={"Retry-After": str(self.recovery_timeout)}
                    )
            
            self.circuit_states[endpoint_key] = circuit
        
        # Process request
        start_time = time.time()
        success = False
        
        try:
            response = await call_next(request)
            success = response.status_code < 500
            
        except Exception as e:
            success = False
            response = Response(
                content="Internal server error",
                status_code=500
            )
        
        # Update circuit breaker state
        async with self._lock:
            circuit = self.circuit_states[endpoint_key]
            circuit["total_requests"] += 1
            
            if success:
                circuit["success_count"] += 1
                if circuit["state"] == "half-open":
                    # Successful request in half-open state - close circuit
                    circuit["state"] = "closed"
                    circuit["failure_count"] = 0
                    logger.info(f"Circuit breaker closed for {endpoint_key}")
            else:
                circuit["failure_count"] += 1
                if circuit["state"] == "half-open":
                    # Failed request in half-open state - reopen circuit
                    circuit["state"] = "open"
                    circuit["last_failure_time"] = time.time()
                    logger.warning(f"Circuit breaker reopened for {endpoint_key}")
        
        return response
    
    async def get_circuit_states(self) -> Dict[str, Any]:
        """Get current circuit breaker states."""
        async with self._lock:
            return dict(self.circuit_states)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add request tracing and correlation IDs.
    """
    
    def __init__(self, app, header_name: str = "X-Trace-ID"):
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add tracing to requests."""
        import uuid
        
        # Get or generate trace ID
        trace_id = request.headers.get(self.header_name.lower())
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        # Add to request state for use in endpoints
        request.state.trace_id = trace_id
        
        # Process request
        response = await call_next(request)
        
        # Add trace ID to response headers
        response.headers[self.header_name] = trace_id
        
        return response


# Convenience function to get all performance metrics
async def get_comprehensive_metrics() -> Dict[str, Any]:
    """Get comprehensive performance metrics from all monitoring components."""
    profiler = get_performance_profiler()
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "performance_report": await profiler.get_performance_report(),
        "memory_analysis": get_memory_profiler().get_memory_growth(),
        "system_resources": await profiler.system_monitor.get_metrics_summary(),
        "database_queries": await profiler.db_profiler.get_query_stats()
    }