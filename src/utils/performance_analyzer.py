"""
Performance analyzer for profiling and identifying bottlenecks.
"""

import asyncio
import cProfile
import io
import pstats
import time
import tracemalloc
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import psutil
from prometheus_client import Counter, Gauge, Histogram, Summary

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


performance_metrics = {
    'cpu_usage': Gauge('app_cpu_usage_percent', 'CPU usage percentage'),
    'memory_usage': Gauge('app_memory_usage_mb', 'Memory usage in MB'),
    'active_tasks': Gauge('app_active_async_tasks', 'Number of active async tasks'),
    'db_pool_size': Gauge('app_db_pool_size', 'Database connection pool size'),
    'cache_hit_rate': Gauge('app_cache_hit_rate', 'Cache hit rate percentage'),
    'request_duration': Histogram('app_request_duration_seconds', 'Request duration in seconds', 
                                  buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]),
    'error_rate': Counter('app_errors_total', 'Total number of errors'),
}


@dataclass
class ProfileResult:
    """Result of a profiling session."""
    name: str
    duration: float
    cpu_percent: float
    memory_mb: float
    top_functions: List[Tuple[str, float]]
    memory_allocations: List[Tuple[str, int]]
    async_metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class PerformanceAnalyzer:
    """Comprehensive performance analysis tool."""
    
    def __init__(self):
        self.profiles: List[ProfileResult] = []
        self.bottlenecks: Dict[str, List[str]] = {}
        self.baseline_metrics: Optional[Dict[str, float]] = None
        
    @contextmanager
    def profile_code(self, name: str):
        """Profile a code block."""
        profiler = cProfile.Profile()
        tracemalloc.start()
        start_time = time.time()
        process = psutil.Process()
        
        cpu_before = process.cpu_percent()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        profiler.enable()
        try:
            yield
        finally:
            profiler.disable()
            
            duration = time.time() - start_time
            cpu_after = process.cpu_percent()
            mem_after = process.memory_info().rss / 1024 / 1024
            
            # Get profiling stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(10)
            
            # Parse top functions
            top_functions = []
            for line in s.getvalue().split('\n')[5:15]:  # Skip header lines
                if line and '{' in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        func_name = parts[-1]
                        cum_time = float(parts[3])
                        top_functions.append((func_name, cum_time))
            
            # Get memory allocations
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            memory_allocations = [(stat.traceback.format()[0], stat.size) for stat in top_stats]
            
            tracemalloc.stop()
            
            result = ProfileResult(
                name=name,
                duration=duration,
                cpu_percent=cpu_after - cpu_before,
                memory_mb=mem_after - mem_before,
                top_functions=top_functions,
                memory_allocations=memory_allocations
            )
            
            self._analyze_bottlenecks(result)
            self.profiles.append(result)
            
            logger.info(f"Profile '{name}' completed: {duration:.3f}s, "
                       f"CPU: {result.cpu_percent:.1f}%, Memory: {result.memory_mb:.1f}MB")
    
    @asynccontextmanager
    async def profile_async(self, name: str):
        """Profile async code with event loop metrics."""
        start_time = time.time()
        process = psutil.Process()
        
        cpu_before = process.cpu_percent()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        # Track async tasks
        tasks_before = len(asyncio.all_tasks())
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            cpu_after = process.cpu_percent()
            mem_after = process.memory_info().rss / 1024 / 1024
            tasks_after = len(asyncio.all_tasks())
            
            # Get event loop stats
            loop = asyncio.get_event_loop()
            loop_time = loop.time()
            
            result = ProfileResult(
                name=name,
                duration=duration,
                cpu_percent=cpu_after - cpu_before,
                memory_mb=mem_after - mem_before,
                top_functions=[],
                memory_allocations=[],
                async_metrics={
                    'tasks_created': tasks_after - tasks_before,
                    'loop_time': loop_time,
                    'tasks_active': tasks_after
                }
            )
            
            self._analyze_bottlenecks(result)
            self.profiles.append(result)
            
            logger.info(f"Async profile '{name}': {duration:.3f}s, "
                       f"Tasks: {result.async_metrics['tasks_created']}")
    
    def _analyze_bottlenecks(self, result: ProfileResult):
        """Analyze results for bottlenecks."""
        bottlenecks = []
        
        # CPU bottlenecks
        if result.cpu_percent > 80:
            bottlenecks.append(f"High CPU usage: {result.cpu_percent:.1f}%")
            result.recommendations.append("Consider optimizing CPU-intensive operations")
        
        # Memory bottlenecks
        if result.memory_mb > 100:
            bottlenecks.append(f"High memory allocation: {result.memory_mb:.1f}MB")
            result.recommendations.append("Review memory usage and consider object pooling")
        
        # Duration bottlenecks
        if result.duration > 1.0:
            bottlenecks.append(f"Long execution time: {result.duration:.3f}s")
            result.recommendations.append("Consider breaking down into smaller operations")
        
        # Async bottlenecks
        if result.async_metrics.get('tasks_created', 0) > 100:
            bottlenecks.append(f"Too many async tasks: {result.async_metrics['tasks_created']}")
            result.recommendations.append("Consider using task batching or connection pooling")
        
        if bottlenecks:
            self.bottlenecks[result.name] = bottlenecks
    
    def set_baseline(self):
        """Set current system metrics as baseline."""
        process = psutil.Process()
        self.baseline_metrics = {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'open_files': len(process.open_files()),
            'connections': len(process.connections()),
            'threads': process.num_threads()
        }
        logger.info(f"Baseline metrics set: {self.baseline_metrics}")
    
    def compare_to_baseline(self) -> Dict[str, float]:
        """Compare current metrics to baseline."""
        if not self.baseline_metrics:
            logger.warning("No baseline metrics set")
            return {}
        
        process = psutil.Process()
        current = {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'open_files': len(process.open_files()),
            'connections': len(process.connections()),
            'threads': process.num_threads()
        }
        
        delta = {
            key: current[key] - self.baseline_metrics[key]
            for key in current
        }
        
        return delta
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance report."""
        report = ["=" * 80]
        report.append("PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Summary
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total profiles: {len(self.profiles)}")
        report.append(f"Bottlenecks found: {len(self.bottlenecks)}")
        report.append("")
        
        # Bottlenecks
        if self.bottlenecks:
            report.append("BOTTLENECKS IDENTIFIED")
            report.append("-" * 40)
            for name, issues in self.bottlenecks.items():
                report.append(f"\n{name}:")
                for issue in issues:
                    report.append(f"  - {issue}")
            report.append("")
        
        # Detailed profiles
        report.append("DETAILED PROFILES")
        report.append("-" * 40)
        for profile in self.profiles:
            report.append(f"\n{profile.name}:")
            report.append(f"  Duration: {profile.duration:.3f}s")
            report.append(f"  CPU: {profile.cpu_percent:.1f}%")
            report.append(f"  Memory: {profile.memory_mb:.1f}MB")
            
            if profile.top_functions:
                report.append("  Top Functions:")
                for func, time_spent in profile.top_functions[:5]:
                    report.append(f"    - {func}: {time_spent:.3f}s")
            
            if profile.async_metrics:
                report.append("  Async Metrics:")
                for key, value in profile.async_metrics.items():
                    report.append(f"    - {key}: {value}")
            
            if profile.recommendations:
                report.append("  Recommendations:")
                for rec in profile.recommendations:
                    report.append(f"    - {rec}")
        
        # Baseline comparison
        if self.baseline_metrics:
            delta = self.compare_to_baseline()
            report.append("\nBASELINE COMPARISON")
            report.append("-" * 40)
            for key, value in delta.items():
                sign = "+" if value > 0 else ""
                report.append(f"  {key}: {sign}{value:.2f}")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get specific optimization suggestions based on analysis."""
        suggestions = []
        
        # Analyze all profiles
        avg_cpu = sum(p.cpu_percent for p in self.profiles) / len(self.profiles) if self.profiles else 0
        avg_memory = sum(p.memory_mb for p in self.profiles) / len(self.profiles) if self.profiles else 0
        avg_duration = sum(p.duration for p in self.profiles) / len(self.profiles) if self.profiles else 0
        
        if avg_cpu > 50:
            suggestions.append("HIGH CPU: Consider using connection pooling and caching")
            suggestions.append("HIGH CPU: Profile hot functions and optimize algorithms")
        
        if avg_memory > 50:
            suggestions.append("HIGH MEMORY: Implement object pooling for frequently created objects")
            suggestions.append("HIGH MEMORY: Review data structures and use generators where possible")
        
        if avg_duration > 0.5:
            suggestions.append("SLOW OPERATIONS: Consider implementing async/await patterns")
            suggestions.append("SLOW OPERATIONS: Add caching for expensive computations")
        
        # Check for specific patterns
        slow_functions = []
        for profile in self.profiles:
            for func, time_spent in profile.top_functions:
                if time_spent > 0.1:
                    slow_functions.append(func)
        
        if slow_functions:
            suggestions.append(f"OPTIMIZE: Focus on these slow functions: {', '.join(set(slow_functions)[:3])}")
        
        return suggestions


class DatabaseProfiler:
    """Profile database operations."""
    
    def __init__(self):
        self.query_times: List[Tuple[str, float]] = []
        self.slow_queries: List[Tuple[str, float]] = []
        self.connection_stats: Dict[str, int] = {}
    
    @contextmanager
    def profile_query(self, query: str):
        """Profile a database query."""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.query_times.append((query[:100], duration))
            
            if duration > 0.1:  # Slow query threshold
                self.slow_queries.append((query, duration))
                logger.warning(f"Slow query detected ({duration:.3f}s): {query[:100]}")
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query performance statistics."""
        if not self.query_times:
            return {}
        
        times = [t for _, t in self.query_times]
        return {
            'total_queries': len(self.query_times),
            'avg_time': sum(times) / len(times),
            'max_time': max(times),
            'min_time': min(times),
            'slow_queries': len(self.slow_queries),
            'slow_query_percentage': (len(self.slow_queries) / len(self.query_times)) * 100
        }


class KafkaProfiler:
    """Profile Kafka operations."""
    
    def __init__(self):
        self.message_times: List[float] = []
        self.batch_sizes: List[int] = []
        self.lag_measurements: List[int] = []
    
    def record_message(self, processing_time: float):
        """Record message processing time."""
        self.message_times.append(processing_time)
        performance_metrics['request_duration'].observe(processing_time)
    
    def record_batch(self, size: int, processing_time: float):
        """Record batch processing."""
        self.batch_sizes.append(size)
        if size > 0:
            per_message_time = processing_time / size
            for _ in range(size):
                self.message_times.append(per_message_time)
    
    def record_lag(self, lag: int):
        """Record consumer lag."""
        self.lag_measurements.append(lag)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Kafka performance statistics."""
        if not self.message_times:
            return {}
        
        return {
            'messages_processed': len(self.message_times),
            'avg_processing_time': sum(self.message_times) / len(self.message_times),
            'max_processing_time': max(self.message_times),
            'avg_batch_size': sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0,
            'avg_lag': sum(self.lag_measurements) / len(self.lag_measurements) if self.lag_measurements else 0,
            'throughput': len(self.message_times) / sum(self.message_times) if sum(self.message_times) > 0 else 0
        }


# Global instances
performance_analyzer = PerformanceAnalyzer()
db_profiler = DatabaseProfiler()
kafka_profiler = KafkaProfiler()