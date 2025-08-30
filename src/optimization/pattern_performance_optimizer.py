import asyncio
import time
import cProfile
import pstats
import io
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import pandas as pd
from functools import lru_cache, wraps
import redis
import pickle
import hashlib
import logging
from memory_profiler import profile
import tracemalloc

from src.data.models.pattern_models import PatternFilter, Pattern
from src.services.pattern_detection_service import PatternDetectionService
from src.data.storage.pattern_repository import PatternRepository

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    operation: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    cache_hits: int
    cache_misses: int
    db_queries: int
    patterns_processed: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def throughput(self) -> float:
        return self.patterns_processed / self.execution_time if self.execution_time > 0 else 0
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0


@dataclass
class OptimizationResult:
    original_metrics: PerformanceMetrics
    optimized_metrics: PerformanceMetrics
    improvement_percentage: float
    optimization_techniques: List[str]
    recommendations: List[str]


class PerformanceProfiler:
    
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.memory_snapshots = []
        self.metrics_history: List[PerformanceMetrics] = []
        
    def profile_function(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.profiler.enable()
            tracemalloc.start()
            
            start_time = time.time()
            start_memory = tracemalloc.get_traced_memory()[0]
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = tracemalloc.get_traced_memory()[0]
            
            self.profiler.disable()
            tracemalloc.stop()
            
            execution_time = end_time - start_time
            memory_usage = (end_memory - start_memory) / 1024 / 1024  # MB
            
            logger.info(f"{func.__name__} - Time: {execution_time:.3f}s, Memory: {memory_usage:.2f}MB")
            
            return result
        return wrapper
    
    def get_profile_stats(self) -> str:
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        return s.getvalue()
    
    @staticmethod
    def measure_operation(operation_name: str):
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                start_memory = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
                
                result = await func(*args, **kwargs)
                
                end_time = time.perf_counter()
                end_memory = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
                
                metrics = PerformanceMetrics(
                    operation=operation_name,
                    execution_time=end_time - start_time,
                    memory_usage=(end_memory - start_memory) / 1024 / 1024,
                    cpu_usage=0,  # Would need psutil for accurate CPU
                    cache_hits=0,
                    cache_misses=0,
                    db_queries=0,
                    patterns_processed=len(result) if isinstance(result, list) else 1
                )
                
                logger.debug(f"{operation_name} metrics: {metrics}")
                return result
                
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                
                logger.debug(f"{operation_name} took {end_time - start_time:.3f}s")
                return result
                
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator


class PatternCache:
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, 
                 ttl: int = 300, max_memory_items: int = 1000):
        self.redis_client = redis_client
        self.ttl = ttl
        self.memory_cache: Dict[str, Any] = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.max_memory_items = max_memory_items
        self.access_times: Dict[str, datetime] = {}
        
    def _generate_key(self, **kwargs) -> str:
        key_data = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        # Try memory cache first
        if key in self.memory_cache:
            self.cache_stats['hits'] += 1
            self.access_times[key] = datetime.utcnow()
            return self.memory_cache[key]
        
        # Try Redis if available
        if self.redis_client:
            try:
                data = self.redis_client.get(f"pattern:{key}")
                if data:
                    self.cache_stats['hits'] += 1
                    result = pickle.loads(data)
                    self._add_to_memory_cache(key, result)
                    return result
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        self.cache_stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any):
        # Add to memory cache
        self._add_to_memory_cache(key, value)
        
        # Add to Redis if available
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"pattern:{key}",
                    self.ttl,
                    pickle.dumps(value)
                )
            except Exception as e:
                logger.error(f"Redis set error: {e}")
    
    def _add_to_memory_cache(self, key: str, value: Any):
        # Implement LRU eviction if needed
        if len(self.memory_cache) >= self.max_memory_items:
            self._evict_lru()
        
        self.memory_cache[key] = value
        self.access_times[key] = datetime.utcnow()
    
    def _evict_lru(self):
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times, key=self.access_times.get)
        del self.memory_cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self):
        self.memory_cache.clear()
        self.access_times.clear()
        if self.redis_client:
            for key in self.redis_client.scan_iter("pattern:*"):
                self.redis_client.delete(key)
    
    def get_stats(self) -> Dict[str, Any]:
        total = self.cache_stats['hits'] + self.cache_stats['misses']
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': self.cache_stats['hits'] / total if total > 0 else 0,
            'memory_items': len(self.memory_cache),
            'memory_size_mb': sum(len(pickle.dumps(v)) for v in self.memory_cache.values()) / 1024 / 1024
        }


class QueryOptimizer:
    
    def __init__(self, repository: PatternRepository):
        self.repository = repository
        self.query_cache = {}
        self.query_stats = defaultdict(lambda: {'count': 0, 'total_time': 0})
        
    def optimize_filter(self, filter_params: PatternFilter) -> PatternFilter:
        # Add indexes hints
        if filter_params.symbols and len(filter_params.symbols) == 1:
            filter_params.order_by = 'symbol'
        
        # Limit default queries
        if not filter_params.limit:
            filter_params.limit = 100
        
        # Use more selective filters first
        if filter_params.min_confidence and filter_params.min_confidence > 0.8:
            filter_params.order_by = 'confidence'
        
        return filter_params
    
    @lru_cache(maxsize=128)
    def cached_query(self, query_hash: str):
        return self.repository.find_patterns(PatternFilter())
    
    def batch_queries(self, filters: List[PatternFilter]) -> List[List[Pattern]]:
        # Group similar queries
        grouped = defaultdict(list)
        for f in filters:
            key = (tuple(f.pattern_types) if f.pattern_types else None,
                   tuple(f.timeframes) if f.timeframes else None)
            grouped[key].append(f)
        
        results = []
        for group in grouped.values():
            # Combine filters where possible
            if len(group) > 1:
                combined = self._combine_filters(group)
                patterns, _ = self.repository.find_patterns(combined)
                # Distribute results
                for f in group:
                    filtered = self._filter_patterns(patterns, f)
                    results.append(filtered)
            else:
                patterns, _ = self.repository.find_patterns(group[0])
                results.append(patterns)
        
        return results
    
    def _combine_filters(self, filters: List[PatternFilter]) -> PatternFilter:
        combined = PatternFilter()
        
        # Combine symbols
        all_symbols = set()
        for f in filters:
            if f.symbols:
                all_symbols.update(f.symbols)
        if all_symbols:
            combined.symbols = list(all_symbols)
        
        # Use most restrictive confidence
        combined.min_confidence = max(f.min_confidence for f in filters if f.min_confidence)
        
        # Combine date ranges
        combined.start_date = min(f.start_date for f in filters if f.start_date)
        combined.end_date = max(f.end_date for f in filters if f.end_date)
        
        return combined
    
    def _filter_patterns(self, patterns: List[Pattern], filter_params: PatternFilter) -> List[Pattern]:
        filtered = patterns
        
        if filter_params.symbols:
            filtered = [p for p in filtered if p.symbol in filter_params.symbols]
        
        if filter_params.min_confidence:
            filtered = [p for p in filtered if p.confidence >= filter_params.min_confidence]
        
        return filtered[:filter_params.limit]
    
    def analyze_slow_queries(self) -> List[Dict[str, Any]]:
        slow_queries = []
        
        for query, stats in self.query_stats.items():
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
            if avg_time > 1.0:  # Queries taking more than 1 second
                slow_queries.append({
                    'query': query,
                    'avg_time': avg_time,
                    'count': stats['count'],
                    'total_time': stats['total_time']
                })
        
        return sorted(slow_queries, key=lambda x: x['avg_time'], reverse=True)


class ParallelProcessor:
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        
    async def process_symbols_parallel(self, 
                                      symbols: List[str],
                                      detector: Callable,
                                      data_getter: Callable) -> Dict[str, List[Any]]:
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self._process_symbol(symbol, detector, data_getter))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return dict(zip(symbols, results))
    
    async def _process_symbol(self, symbol: str, detector: Callable, data_getter: Callable) -> List[Any]:
        data = await data_getter(symbol)
        if data is not None:
            return await detector(data, symbol)
        return []
    
    def batch_process(self, items: List[Any], processor: Callable, 
                     batch_size: int = 100) -> List[Any]:
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = list(self.thread_pool.map(processor, batch))
            results.extend(batch_results)
        
        return results
    
    def parallel_map(self, func: Callable, items: List[Any], 
                     use_processes: bool = False) -> List[Any]:
        pool = self.process_pool if use_processes else self.thread_pool
        return list(pool.map(func, items))
    
    def shutdown(self):
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class PatternPerformanceOptimizer:
    
    def __init__(self, detection_service: PatternDetectionService,
                 repository: PatternRepository,
                 redis_client: Optional[redis.Redis] = None):
        self.detection_service = detection_service
        self.repository = repository
        self.cache = PatternCache(redis_client)
        self.query_optimizer = QueryOptimizer(repository)
        self.parallel_processor = ParallelProcessor()
        self.profiler = PerformanceProfiler()
        
    async def optimize_pattern_detection(self, 
                                        data: pd.DataFrame,
                                        symbol: str,
                                        use_cache: bool = True,
                                        use_parallel: bool = True) -> OptimizationResult:
        # Measure original performance
        original_metrics = await self._measure_original_performance(data, symbol)
        
        # Apply optimizations
        optimizations = []
        
        if use_cache:
            optimizations.append("Caching")
            self._enable_caching()
        
        if use_parallel:
            optimizations.append("Parallel Processing")
            self._enable_parallel_processing()
        
        # Apply query optimization
        optimizations.append("Query Optimization")
        self._optimize_queries()
        
        # Measure optimized performance
        optimized_metrics = await self._measure_optimized_performance(data, symbol)
        
        # Calculate improvement
        improvement = ((original_metrics.execution_time - optimized_metrics.execution_time) / 
                      original_metrics.execution_time * 100)
        
        recommendations = self._generate_recommendations(original_metrics, optimized_metrics)
        
        return OptimizationResult(
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            improvement_percentage=improvement,
            optimization_techniques=optimizations,
            recommendations=recommendations
        )
    
    async def _measure_original_performance(self, data: pd.DataFrame, symbol: str) -> PerformanceMetrics:
        start_time = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        
        # Run pattern detection without optimizations
        patterns = await self.detection_service.scan_for_patterns(
            data, symbol, timeframe='1h'
        )
        
        end_time = time.perf_counter()
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        return PerformanceMetrics(
            operation="original_detection",
            execution_time=end_time - start_time,
            memory_usage=(end_memory - start_memory) / 1024 / 1024,
            cpu_usage=0,
            cache_hits=0,
            cache_misses=0,
            db_queries=1,
            patterns_processed=len(patterns)
        )
    
    async def _measure_optimized_performance(self, data: pd.DataFrame, symbol: str) -> PerformanceMetrics:
        start_time = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        
        # Check cache first
        cache_key = self.cache._generate_key(symbol=symbol, data_hash=hash(data.values.tobytes()))
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            patterns = cached_result
        else:
            # Run optimized pattern detection
            patterns = await self._optimized_detection(data, symbol)
            self.cache.set(cache_key, patterns)
        
        end_time = time.perf_counter()
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        cache_stats = self.cache.get_stats()
        
        return PerformanceMetrics(
            operation="optimized_detection",
            execution_time=end_time - start_time,
            memory_usage=(end_memory - start_memory) / 1024 / 1024,
            cpu_usage=0,
            cache_hits=cache_stats['hits'],
            cache_misses=cache_stats['misses'],
            db_queries=0 if cached_result else 1,
            patterns_processed=len(patterns) if patterns else 0
        )
    
    async def _optimized_detection(self, data: pd.DataFrame, symbol: str) -> List[Any]:
        # Use parallel processing for multiple detectors
        detector_tasks = []
        
        for detector_name, detector in self.detection_service.detectors.items():
            task = asyncio.create_task(detector.detect(data, symbol, timeframe='1h'))
            detector_tasks.append(task)
        
        all_patterns = await asyncio.gather(*detector_tasks)
        
        # Flatten results
        patterns = []
        for detector_patterns in all_patterns:
            patterns.extend(detector_patterns)
        
        return patterns
    
    def _enable_caching(self):
        self.cache.clear()
        logger.info("Caching enabled")
    
    def _enable_parallel_processing(self):
        logger.info(f"Parallel processing enabled with {self.parallel_processor.max_workers} workers")
    
    def _optimize_queries(self):
        # Pre-compile common queries
        common_filters = [
            PatternFilter(min_confidence=0.7, limit=100),
            PatternFilter(pattern_types=['candlestick'], limit=50),
            PatternFilter(timeframes=['1h', '4h'], limit=100)
        ]
        
        for filter_params in common_filters:
            self.query_optimizer.cached_query(str(filter_params))
    
    def _generate_recommendations(self, original: PerformanceMetrics, 
                                 optimized: PerformanceMetrics) -> List[str]:
        recommendations = []
        
        # Time-based recommendations
        if optimized.execution_time > 1.0:
            recommendations.append("Consider implementing more aggressive caching")
            recommendations.append("Reduce pattern detection complexity")
        
        # Memory-based recommendations
        if optimized.memory_usage > 100:
            recommendations.append("High memory usage detected. Consider streaming processing")
            recommendations.append("Implement memory pooling for pattern objects")
        
        # Cache recommendations
        if optimized.cache_hit_rate < 0.5:
            recommendations.append("Low cache hit rate. Increase cache TTL or size")
        
        # Parallel processing recommendations
        if optimized.patterns_processed / optimized.execution_time < 100:
            recommendations.append("Low throughput. Consider increasing parallelism")
        
        # Database recommendations
        if optimized.db_queries > 10:
            recommendations.append("High database query count. Implement query batching")
        
        return recommendations
    
    async def load_test(self, num_symbols: int = 10, 
                       patterns_per_symbol: int = 100,
                       duration_seconds: int = 60) -> Dict[str, Any]:
        
        start_time = time.time()
        total_patterns = 0
        errors = 0
        latencies = []
        
        # Generate test data
        test_symbols = [f"TEST_{i}" for i in range(num_symbols)]
        test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=500, freq='1h'),
            'open': np.random.uniform(99, 101, 500),
            'high': np.random.uniform(101, 103, 500),
            'low': np.random.uniform(97, 99, 500),
            'close': np.random.uniform(99, 101, 500),
            'volume': np.random.uniform(900000, 1100000, 500)
        })
        
        while time.time() - start_time < duration_seconds:
            for symbol in test_symbols:
                try:
                    operation_start = time.perf_counter()
                    
                    patterns = await self.detection_service.scan_for_patterns(
                        test_data, symbol, timeframe='1h'
                    )
                    
                    operation_time = time.perf_counter() - operation_start
                    latencies.append(operation_time)
                    total_patterns += len(patterns)
                    
                except Exception as e:
                    logger.error(f"Load test error: {e}")
                    errors += 1
        
        elapsed_time = time.time() - start_time
        
        return {
            'duration': elapsed_time,
            'total_patterns': total_patterns,
            'patterns_per_second': total_patterns / elapsed_time,
            'avg_latency': np.mean(latencies) if latencies else 0,
            'p95_latency': np.percentile(latencies, 95) if latencies else 0,
            'p99_latency': np.percentile(latencies, 99) if latencies else 0,
            'errors': errors,
            'error_rate': errors / (errors + len(latencies)) if latencies else 0,
            'cache_stats': self.cache.get_stats()
        }
    
    def generate_performance_report(self) -> str:
        report = []
        report.append("=" * 50)
        report.append("Pattern Detection Performance Report")
        report.append("=" * 50)
        
        # Cache statistics
        cache_stats = self.cache.get_stats()
        report.append("\nCache Performance:")
        report.append(f"  Hit Rate: {cache_stats['hit_rate']:.2%}")
        report.append(f"  Memory Items: {cache_stats['memory_items']}")
        report.append(f"  Memory Size: {cache_stats['memory_size_mb']:.2f} MB")
        
        # Query analysis
        slow_queries = self.query_optimizer.analyze_slow_queries()
        if slow_queries:
            report.append("\nSlow Queries:")
            for sq in slow_queries[:5]:
                report.append(f"  Query: {sq['query'][:50]}...")
                report.append(f"    Avg Time: {sq['avg_time']:.3f}s")
                report.append(f"    Count: {sq['count']}")
        
        # Profile stats
        report.append("\nProfile Statistics:")
        report.append(self.profiler.get_profile_stats())
        
        return "\n".join(report)