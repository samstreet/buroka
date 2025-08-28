"""
API usage monitoring and analytics system.
Provides comprehensive metrics, rate limiting analytics, and usage patterns.
"""

import time
import json
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


class MetricType(str, Enum):
    """Types of metrics to track."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    RATE = "rate"


@dataclass
class APICallMetric:
    """Represents a single API call metric."""
    timestamp: float
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    user_id: Optional[str] = None
    api_key_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_size: Optional[int] = None
    response_size: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class UsageStatistics:
    """Usage statistics for a time period."""
    total_requests: int = 0
    successful_requests: int = 0
    error_requests: int = 0
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    total_data_transferred: int = 0
    unique_users: int = 0
    unique_ips: int = 0
    top_endpoints: Dict[str, int] = None
    error_breakdown: Dict[str, int] = None
    
    def __post_init__(self):
        if self.top_endpoints is None:
            self.top_endpoints = {}
        if self.error_breakdown is None:
            self.error_breakdown = {}


class APIUsageTracker:
    """Tracks API usage patterns and metrics."""
    
    def __init__(self, retention_hours: int = 24, max_metrics: int = 10000):
        self.retention_hours = retention_hours
        self.max_metrics = max_metrics
        self.logger = logging.getLogger(__name__)
        
        # Storage for metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.endpoint_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "errors": 0,
            "last_called": None
        })
        
        # Rate limiting tracking
        self.rate_limit_violations: deque = deque(maxlen=1000)
        self.suspicious_ips: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "requests": 0,
            "errors": 0,
            "first_seen": None,
            "last_seen": None,
            "violations": 0
        })
        
        # User analytics
        self.user_analytics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_requests": 0,
            "first_request": None,
            "last_request": None,
            "favorite_endpoints": defaultdict(int),
            "avg_response_time": 0.0,
            "error_rate": 0.0
        })
        
        # Performance tracking
        self.performance_alerts: List[Dict[str, Any]] = []
        self.slow_queries: deque = deque(maxlen=100)
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    self._cleanup_old_metrics()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in cleanup task: {e}")
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._cleanup_task = loop.create_task(cleanup_loop())
        except RuntimeError:
            # No event loop running, cleanup will be manual
            pass
    
    def record_api_call(self, metric: APICallMetric) -> None:
        """Record an API call metric."""
        try:
            # Add to metrics storage
            self.metrics.append(metric)
            
            # Update endpoint statistics
            endpoint_key = f"{metric.method}:{metric.endpoint}"
            stats = self.endpoint_stats[endpoint_key]
            stats["count"] += 1
            stats["total_time"] += metric.response_time_ms
            stats["last_called"] = metric.timestamp
            
            if metric.status_code >= 400:
                stats["errors"] += 1
            
            # Track slow queries
            if metric.response_time_ms > 1000:  # > 1 second
                self.slow_queries.append({
                    "timestamp": metric.timestamp,
                    "endpoint": metric.endpoint,
                    "method": metric.method,
                    "response_time": metric.response_time_ms,
                    "user_id": metric.user_id
                })
            
            # Update user analytics
            if metric.user_id:
                user_stats = self.user_analytics[metric.user_id]
                user_stats["total_requests"] += 1
                user_stats["last_request"] = metric.timestamp
                if user_stats["first_request"] is None:
                    user_stats["first_request"] = metric.timestamp
                
                user_stats["favorite_endpoints"][endpoint_key] += 1
                
                # Update average response time
                old_avg = user_stats["avg_response_time"]
                count = user_stats["total_requests"]
                user_stats["avg_response_time"] = (old_avg * (count - 1) + metric.response_time_ms) / count
            
            # Track suspicious IP activity
            if metric.ip_address:
                ip_stats = self.suspicious_ips[metric.ip_address]
                ip_stats["requests"] += 1
                ip_stats["last_seen"] = metric.timestamp
                if ip_stats["first_seen"] is None:
                    ip_stats["first_seen"] = metric.timestamp
                
                if metric.status_code >= 400:
                    ip_stats["errors"] += 1
                
                # Check for suspicious patterns
                self._check_suspicious_activity(metric.ip_address, ip_stats)
            
            # Performance alerting
            if metric.response_time_ms > 5000:  # > 5 seconds
                self.performance_alerts.append({
                    "timestamp": metric.timestamp,
                    "type": "slow_response",
                    "endpoint": metric.endpoint,
                    "response_time": metric.response_time_ms,
                    "severity": "high" if metric.response_time_ms > 10000 else "medium"
                })
            
        except Exception as e:
            self.logger.error(f"Error recording API call metric: {e}")
    
    def record_rate_limit_violation(
        self,
        ip_address: str,
        user_id: Optional[str] = None,
        endpoint: str = None,
        violation_type: str = "rate_limit"
    ) -> None:
        """Record a rate limiting violation."""
        violation = {
            "timestamp": time.time(),
            "ip_address": ip_address,
            "user_id": user_id,
            "endpoint": endpoint,
            "type": violation_type
        }
        
        self.rate_limit_violations.append(violation)
        
        # Update suspicious IP tracking
        if ip_address in self.suspicious_ips:
            self.suspicious_ips[ip_address]["violations"] += 1
    
    def _check_suspicious_activity(self, ip_address: str, ip_stats: Dict[str, Any]) -> None:
        """Check for suspicious activity patterns."""
        now = time.time()
        hour_ago = now - 3600
        
        # Count recent requests from this IP
        recent_requests = sum(
            1 for metric in self.metrics 
            if metric.ip_address == ip_address and metric.timestamp > hour_ago
        )
        
        # High request rate
        if recent_requests > 100:  # More than 100 requests per hour from single IP
            self.performance_alerts.append({
                "timestamp": now,
                "type": "suspicious_activity",
                "ip_address": ip_address,
                "requests_per_hour": recent_requests,
                "severity": "medium"
            })
        
        # High error rate
        error_rate = ip_stats["errors"] / max(ip_stats["requests"], 1)
        if error_rate > 0.5 and ip_stats["requests"] > 10:  # >50% error rate with >10 requests
            self.performance_alerts.append({
                "timestamp": now,
                "type": "high_error_rate",
                "ip_address": ip_address,
                "error_rate": error_rate,
                "severity": "high"
            })
    
    def get_usage_statistics(self, hours: int = 1) -> UsageStatistics:
        """Get usage statistics for the specified time period."""
        try:
            cutoff_time = time.time() - (hours * 3600)
            recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
            
            if not recent_metrics:
                return UsageStatistics()
            
            stats = UsageStatistics()
            stats.total_requests = len(recent_metrics)
            
            response_times = []
            unique_users = set()
            unique_ips = set()
            endpoint_counts = defaultdict(int)
            error_counts = defaultdict(int)
            total_request_size = 0
            total_response_size = 0
            
            for metric in recent_metrics:
                # Success/error counts
                if metric.status_code < 400:
                    stats.successful_requests += 1
                else:
                    stats.error_requests += 1
                    error_counts[f"{metric.status_code}"] += 1
                
                # Response times
                response_times.append(metric.response_time_ms)
                
                # Unique counters
                if metric.user_id:
                    unique_users.add(metric.user_id)
                if metric.ip_address:
                    unique_ips.add(metric.ip_address)
                
                # Endpoint popularity
                endpoint_key = f"{metric.method} {metric.endpoint}"
                endpoint_counts[endpoint_key] += 1
                
                # Data transfer
                if metric.request_size:
                    total_request_size += metric.request_size
                if metric.response_size:
                    total_response_size += metric.response_size
            
            # Calculate statistics
            if response_times:
                stats.avg_response_time = sum(response_times) / len(response_times)
                stats.min_response_time = min(response_times)
                stats.max_response_time = max(response_times)
            
            stats.unique_users = len(unique_users)
            stats.unique_ips = len(unique_ips)
            stats.total_data_transferred = total_request_size + total_response_size
            
            # Top endpoints (top 10)
            stats.top_endpoints = dict(sorted(
                endpoint_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
            
            # Error breakdown
            stats.error_breakdown = dict(error_counts)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating usage statistics: {e}")
            return UsageStatistics()
    
    def get_endpoint_analytics(self, endpoint: str, method: str = None) -> Dict[str, Any]:
        """Get analytics for a specific endpoint."""
        endpoint_patterns = [endpoint]
        if method:
            endpoint_patterns.append(f"{method}:{endpoint}")
        
        analytics = {
            "total_calls": 0,
            "success_calls": 0,
            "error_calls": 0,
            "avg_response_time": 0.0,
            "total_response_time": 0.0,
            "error_rate": 0.0,
            "last_called": None,
            "hourly_distribution": defaultdict(int),
            "status_code_distribution": defaultdict(int),
            "user_distribution": defaultdict(int)
        }
        
        # Find matching metrics
        matching_metrics = [
            m for m in self.metrics
            if any(pattern in f"{m.method}:{m.endpoint}" for pattern in endpoint_patterns)
        ]
        
        if not matching_metrics:
            return analytics
        
        for metric in matching_metrics:
            analytics["total_calls"] += 1
            analytics["total_response_time"] += metric.response_time_ms
            
            if metric.status_code < 400:
                analytics["success_calls"] += 1
            else:
                analytics["error_calls"] += 1
            
            # Status code distribution
            analytics["status_code_distribution"][str(metric.status_code)] += 1
            
            # User distribution
            if metric.user_id:
                analytics["user_distribution"][metric.user_id] += 1
            
            # Hourly distribution
            hour = datetime.fromtimestamp(metric.timestamp).hour
            analytics["hourly_distribution"][hour] += 1
            
            # Update last called
            if analytics["last_called"] is None or metric.timestamp > analytics["last_called"]:
                analytics["last_called"] = metric.timestamp
        
        # Calculate derived metrics
        if analytics["total_calls"] > 0:
            analytics["avg_response_time"] = analytics["total_response_time"] / analytics["total_calls"]
            analytics["error_rate"] = analytics["error_calls"] / analytics["total_calls"]
        
        return analytics
    
    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get analytics for a specific user."""
        user_stats = self.user_analytics.get(user_id, {})
        if not user_stats:
            return {"error": "User not found"}
        
        # Get recent activity
        user_metrics = [m for m in self.metrics if m.user_id == user_id]
        
        analytics = {
            "total_requests": user_stats["total_requests"],
            "first_request": user_stats["first_request"],
            "last_request": user_stats["last_request"],
            "avg_response_time": user_stats["avg_response_time"],
            "favorite_endpoints": dict(user_stats["favorite_endpoints"]),
            "recent_activity": [],
            "error_rate": 0.0,
            "daily_activity": defaultdict(int)
        }
        
        # Calculate error rate from recent metrics
        if user_metrics:
            error_count = sum(1 for m in user_metrics if m.status_code >= 400)
            analytics["error_rate"] = error_count / len(user_metrics)
            
            # Recent activity (last 10 requests)
            recent_metrics = sorted(user_metrics, key=lambda x: x.timestamp, reverse=True)[:10]
            analytics["recent_activity"] = [
                {
                    "timestamp": m.timestamp,
                    "endpoint": m.endpoint,
                    "method": m.method,
                    "status_code": m.status_code,
                    "response_time": m.response_time_ms
                }
                for m in recent_metrics
            ]
            
            # Daily activity
            for metric in user_metrics:
                date_key = datetime.fromtimestamp(metric.timestamp).strftime("%Y-%m-%d")
                analytics["daily_activity"][date_key] += 1
        
        return analytics
    
    def get_performance_alerts(self, severity: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent performance alerts."""
        alerts = self.performance_alerts[-limit:] if limit else self.performance_alerts
        
        if severity:
            alerts = [a for a in alerts if a.get("severity") == severity]
        
        return sorted(alerts, key=lambda x: x.get("timestamp", 0), reverse=True)
    
    def get_rate_limit_analytics(self) -> Dict[str, Any]:
        """Get rate limiting analytics."""
        return {
            "total_violations": len(self.rate_limit_violations),
            "recent_violations": len([
                v for v in self.rate_limit_violations
                if v["timestamp"] > time.time() - 3600
            ]),
            "top_violating_ips": self._get_top_violating_ips(),
            "violation_patterns": self._get_violation_patterns()
        }
    
    def _get_top_violating_ips(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top rate limit violating IPs."""
        ip_violations = defaultdict(int)
        for violation in self.rate_limit_violations:
            ip_violations[violation["ip_address"]] += 1
        
        return [
            {"ip_address": ip, "violations": count}
            for ip, count in sorted(ip_violations.items(), key=lambda x: x[1], reverse=True)[:limit]
        ]
    
    def _get_violation_patterns(self) -> Dict[str, Any]:
        """Analyze rate limit violation patterns."""
        if not self.rate_limit_violations:
            return {}
        
        hourly_violations = defaultdict(int)
        endpoint_violations = defaultdict(int)
        
        for violation in self.rate_limit_violations:
            hour = datetime.fromtimestamp(violation["timestamp"]).hour
            hourly_violations[hour] += 1
            
            if violation.get("endpoint"):
                endpoint_violations[violation["endpoint"]] += 1
        
        return {
            "hourly_distribution": dict(hourly_violations),
            "endpoint_distribution": dict(endpoint_violations)
        }
    
    def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics beyond retention period."""
        try:
            cutoff_time = time.time() - (self.retention_hours * 3600)
            
            # Clean up metrics
            self.metrics = deque(
                (m for m in self.metrics if m.timestamp > cutoff_time),
                maxlen=self.max_metrics
            )
            
            # Clean up rate limit violations
            self.rate_limit_violations = deque(
                (v for v in self.rate_limit_violations if v["timestamp"] > cutoff_time),
                maxlen=1000
            )
            
            # Clean up performance alerts (keep last 1000)
            if len(self.performance_alerts) > 1000:
                self.performance_alerts = self.performance_alerts[-1000:]
            
            self.logger.debug(f"Cleaned up metrics older than {self.retention_hours} hours")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old metrics: {e}")
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        if format == "json":
            export_data = {
                "metadata": {
                    "export_timestamp": time.time(),
                    "total_metrics": len(self.metrics),
                    "retention_hours": self.retention_hours
                },
                "usage_statistics": asdict(self.get_usage_statistics(24)),
                "endpoint_stats": dict(self.endpoint_stats),
                "performance_alerts": self.get_performance_alerts(),
                "rate_limit_analytics": self.get_rate_limit_analytics()
            }
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")


class APIMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for API usage monitoring and analytics."""
    
    def __init__(self, app, tracker: Optional[APIUsageTracker] = None):
        super().__init__(app)
        self.tracker = tracker or APIUsageTracker()
        self.logger = logging.getLogger(__name__)
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Monitor API call and collect metrics."""
        start_time = time.time()
        
        # Extract request information
        user_id = getattr(request.state, 'user_id', None)
        api_key_id = getattr(request.state, 'api_key_id', None)
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent", "")
        
        # Get request size
        request_size = None
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                request_size = int(content_length)
            except ValueError:
                pass
        
        error_message = None
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Get response size
            response_size = None
            if hasattr(response, 'headers') and 'content-length' in response.headers:
                try:
                    response_size = int(response.headers['content-length'])
                except ValueError:
                    pass
            
            # Create metric
            metric = APICallMetric(
                timestamp=start_time,
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                response_time_ms=response_time_ms,
                user_id=user_id,
                api_key_id=api_key_id,
                ip_address=ip_address,
                user_agent=user_agent,
                request_size=request_size,
                response_size=response_size
            )
            
            # Record the metric
            self.tracker.record_api_call(metric)
            
            return response
            
        except Exception as e:
            # Calculate response time even for errors
            response_time_ms = (time.time() - start_time) * 1000
            error_message = str(e)
            
            # Create error metric
            metric = APICallMetric(
                timestamp=start_time,
                endpoint=request.url.path,
                method=request.method,
                status_code=500,  # Assume 500 for unhandled exceptions
                response_time_ms=response_time_ms,
                user_id=user_id,
                api_key_id=api_key_id,
                ip_address=ip_address,
                user_agent=user_agent,
                request_size=request_size,
                error_message=error_message
            )
            
            # Record the error metric
            self.tracker.record_api_call(metric)
            
            # Re-raise the exception
            raise


# Global tracker instance
_usage_tracker = None


def get_usage_tracker() -> APIUsageTracker:
    """Get the global API usage tracker."""
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = APIUsageTracker()
    return _usage_tracker