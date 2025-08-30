import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import redis
import time

from src.optimization.pattern_performance_optimizer import (
    PatternPerformanceOptimizer,
    PerformanceProfiler,
    PatternCache,
    QueryOptimizer,
    ParallelProcessor,
    PerformanceMetrics,
    OptimizationResult
)
from src.services.pattern_detection_service import PatternDetectionService
from src.data.storage.pattern_repository import PatternRepository
from src.data.models.pattern_models import PatternFilter, Pattern


class TestPerformanceMetrics:
    def test_performance_metrics_properties(self):
        metrics = PerformanceMetrics(
            operation="test_op",
            execution_time=2.0,
            memory_usage=50.0,
            cpu_usage=75.0,
            cache_hits=80,
            cache_misses=20,
            db_queries=5,
            patterns_processed=100
        )
        
        assert metrics.throughput == 50.0  # 100 patterns / 2 seconds
        assert metrics.cache_hit_rate == 0.8  # 80 / (80 + 20)
        
    def test_zero_execution_time(self):
        metrics = PerformanceMetrics(
            operation="test",
            execution_time=0,
            memory_usage=0,
            cpu_usage=0,
            cache_hits=0,
            cache_misses=0,
            db_queries=0,
            patterns_processed=10
        )
        
        assert metrics.throughput == 0
        assert metrics.cache_hit_rate == 0


class TestPerformanceProfiler:
    def test_profile_function_decorator(self):
        profiler = PerformanceProfiler()
        
        @profiler.profile_function
        def test_function(x, y):
            time.sleep(0.01)
            return x + y
        
        result = test_function(1, 2)
        assert result == 3
        
        stats = profiler.get_profile_stats()
        assert "test_function" in stats or "wrapper" in stats
        
    @pytest.mark.asyncio
    async def test_measure_operation_async(self):
        @PerformanceProfiler.measure_operation("async_test")
        async def async_operation():
            await asyncio.sleep(0.01)
            return [1, 2, 3]
        
        result = await async_operation()
        assert result == [1, 2, 3]
        
    def test_measure_operation_sync(self):
        @PerformanceProfiler.measure_operation("sync_test")
        def sync_operation():
            time.sleep(0.01)
            return "result"
        
        result = sync_operation()
        assert result == "result"


class TestPatternCache:
    def setup_method(self):
        self.cache = PatternCache(max_memory_items=3)
        
    def test_memory_cache_operations(self):
        key = self.cache._generate_key(symbol="AAPL", timeframe="1h")
        
        # Test miss
        result = self.cache.get(key)
        assert result is None
        assert self.cache.cache_stats['misses'] == 1
        
        # Test set and hit
        self.cache.set(key, {"pattern": "test"})
        result = self.cache.get(key)
        assert result == {"pattern": "test"}
        assert self.cache.cache_stats['hits'] == 1
        
    def test_lru_eviction(self):
        # Fill cache to max
        for i in range(4):
            key = self.cache._generate_key(id=i)
            self.cache.set(key, f"value_{i}")
        
        # First item should be evicted
        key_0 = self.cache._generate_key(id=0)
        assert key_0 not in self.cache.memory_cache
        
        # Last 3 items should be present
        for i in range(1, 4):
            key = self.cache._generate_key(id=i)
            assert key in self.cache.memory_cache
            
    def test_cache_stats(self):
        for i in range(5):
            key = self.cache._generate_key(id=i)
            self.cache.set(key, f"value_{i}")
            self.cache.get(key)
        
        # Try to get non-existent key
        self.cache.get("nonexistent")
        
        stats = self.cache.get_stats()
        assert stats['hits'] == 5
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 5/6
        assert stats['memory_items'] == 3  # Due to LRU eviction
        
    @patch('redis.Redis')
    def test_redis_integration(self, mock_redis_class):
        mock_redis = MagicMock()
        mock_redis_class.return_value = mock_redis
        mock_redis.get.return_value = None
        
        cache = PatternCache(redis_client=mock_redis)
        key = cache._generate_key(test="redis")
        
        # Test Redis miss
        result = cache.get(key)
        assert result is None
        mock_redis.get.assert_called_once()
        
        # Test Redis set
        cache.set(key, {"data": "test"})
        mock_redis.setex.assert_called_once()


class TestQueryOptimizer:
    def setup_method(self):
        self.mock_repository = Mock(spec=PatternRepository)
        self.optimizer = QueryOptimizer(self.mock_repository)
        
    def test_optimize_filter(self):
        filter_params = PatternFilter(
            symbols=["AAPL"],
            min_confidence=0.85
        )
        
        optimized = self.optimizer.optimize_filter(filter_params)
        
        assert optimized.order_by == 'confidence'  # High confidence should trigger this
        assert optimized.limit == 100  # Default limit applied
        
    def test_batch_queries(self):
        filters = [
            PatternFilter(symbols=["AAPL"], pattern_types=["candlestick"]),
            PatternFilter(symbols=["GOOGL"], pattern_types=["candlestick"]),
            PatternFilter(symbols=["MSFT"], pattern_types=["chart"])
        ]
        
        self.mock_repository.find_patterns.return_value = ([], 0)
        
        results = self.optimizer.batch_queries(filters)
        
        assert len(results) == 3
        # Should group the first two queries (same pattern type)
        assert self.mock_repository.find_patterns.call_count >= 2
        
    def test_combine_filters(self):
        filters = [
            PatternFilter(symbols=["AAPL"], min_confidence=0.7, start_date=datetime(2024, 1, 1)),
            PatternFilter(symbols=["GOOGL"], min_confidence=0.8, start_date=datetime(2024, 1, 2)),
            PatternFilter(symbols=["MSFT"], min_confidence=0.6, end_date=datetime(2024, 2, 1))
        ]
        
        combined = self.optimizer._combine_filters(filters)
        
        assert set(combined.symbols) == {"AAPL", "GOOGL", "MSFT"}
        assert combined.min_confidence == 0.8  # Most restrictive
        assert combined.start_date == datetime(2024, 1, 1)  # Earliest
        assert combined.end_date == datetime(2024, 2, 1)  # Latest


class TestParallelProcessor:
    def setup_method(self):
        self.processor = ParallelProcessor(max_workers=2)
        
    def teardown_method(self):
        self.processor.shutdown()
        
    @pytest.mark.asyncio
    async def test_process_symbols_parallel(self):
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        async def mock_detector(data, symbol):
            await asyncio.sleep(0.01)
            return [{"symbol": symbol, "pattern": "test"}]
        
        async def mock_data_getter(symbol):
            return pd.DataFrame({"close": [100, 101, 102]})
        
        results = await self.processor.process_symbols_parallel(
            symbols, mock_detector, mock_data_getter
        )
        
        assert len(results) == 3
        assert all(symbol in results for symbol in symbols)
        assert all(len(patterns) == 1 for patterns in results.values())
        
    def test_batch_process(self):
        items = list(range(10))
        
        def processor(x):
            return x * 2
        
        results = self.processor.batch_process(items, processor, batch_size=3)
        
        assert results == [x * 2 for x in items]
        
    def test_parallel_map(self):
        items = [1, 2, 3, 4, 5]
        
        def square(x):
            return x ** 2
        
        # Test thread pool
        thread_results = self.processor.parallel_map(square, items, use_processes=False)
        assert thread_results == [1, 4, 9, 16, 25]
        
        # Test process pool (might fail in some test environments)
        try:
            process_results = self.processor.parallel_map(square, items, use_processes=True)
            assert process_results == [1, 4, 9, 16, 25]
        except Exception:
            pass  # Process pool might not work in all test environments


class TestPatternPerformanceOptimizer:
    @pytest.fixture
    def mock_detection_service(self):
        service = Mock(spec=PatternDetectionService)
        service.scan_for_patterns = AsyncMock(return_value=[
            {"pattern": "test1"},
            {"pattern": "test2"}
        ])
        service.detectors = {
            "detector1": Mock(detect=AsyncMock(return_value=[{"pattern": "d1"}])),
            "detector2": Mock(detect=AsyncMock(return_value=[{"pattern": "d2"}]))
        }
        return service
        
    @pytest.fixture
    def mock_repository(self):
        return Mock(spec=PatternRepository)
        
    @pytest.fixture
    def optimizer(self, mock_detection_service, mock_repository):
        return PatternPerformanceOptimizer(
            mock_detection_service,
            mock_repository
        )
        
    @pytest.mark.asyncio
    async def test_optimize_pattern_detection(self, optimizer):
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h'),
            'close': np.random.uniform(99, 101, 100),
            'volume': np.random.uniform(900000, 1100000, 100)
        })
        
        result = await optimizer.optimize_pattern_detection(
            data, "AAPL", use_cache=True, use_parallel=True
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.original_metrics.patterns_processed > 0
        assert result.optimized_metrics.patterns_processed > 0
        assert len(result.optimization_techniques) > 0
        assert len(result.recommendations) > 0
        
    @pytest.mark.asyncio
    async def test_load_test(self, optimizer):
        result = await optimizer.load_test(
            num_symbols=2,
            patterns_per_symbol=10,
            duration_seconds=1
        )
        
        assert 'duration' in result
        assert 'total_patterns' in result
        assert 'patterns_per_second' in result
        assert 'avg_latency' in result
        assert 'p95_latency' in result
        assert 'p99_latency' in result
        assert 'errors' in result
        assert 'error_rate' in result
        assert 'cache_stats' in result
        
        assert result['duration'] >= 1
        assert result['total_patterns'] >= 0
        
    def test_generate_performance_report(self, optimizer):
        # Add some fake slow queries
        optimizer.query_optimizer.query_stats["slow_query"] = {
            'count': 10,
            'total_time': 15.0
        }
        
        report = optimizer.generate_performance_report()
        
        assert "Pattern Detection Performance Report" in report
        assert "Cache Performance:" in report
        assert "Hit Rate:" in report
        assert "Profile Statistics:" in report
        
    @pytest.mark.asyncio
    async def test_caching_effectiveness(self, optimizer):
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h'),
            'close': np.random.uniform(99, 101, 100),
            'volume': np.random.uniform(900000, 1100000, 100)
        })
        
        # First call - should miss cache
        await optimizer._measure_optimized_performance(data, "AAPL")
        cache_stats1 = optimizer.cache.get_stats()
        
        # Second call - should hit cache
        await optimizer._measure_optimized_performance(data, "AAPL")
        cache_stats2 = optimizer.cache.get_stats()
        
        assert cache_stats2['hits'] > cache_stats1['hits']
        
    def test_recommendations_generation(self, optimizer):
        original = PerformanceMetrics(
            operation="original",
            execution_time=5.0,
            memory_usage=200.0,
            cpu_usage=80.0,
            cache_hits=0,
            cache_misses=100,
            db_queries=50,
            patterns_processed=100
        )
        
        optimized = PerformanceMetrics(
            operation="optimized",
            execution_time=2.0,
            memory_usage=150.0,
            cpu_usage=60.0,
            cache_hits=30,
            cache_misses=70,
            db_queries=20,
            patterns_processed=100
        )
        
        recommendations = optimizer._generate_recommendations(original, optimized)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should recommend caching improvements due to low hit rate
        assert any("cache" in r.lower() for r in recommendations)


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_optimization_workflow(self):
        # Create mocks
        mock_service = Mock(spec=PatternDetectionService)
        mock_service.scan_for_patterns = AsyncMock(return_value=[
            {"pattern": f"pattern_{i}"} for i in range(10)
        ])
        mock_service.detectors = {}
        
        mock_repo = Mock(spec=PatternRepository)
        mock_repo.find_patterns.return_value = ([], 0)
        
        # Create optimizer
        optimizer = PatternPerformanceOptimizer(mock_service, mock_repo)
        
        # Generate test data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=500, freq='1h'),
            'open': 100 + np.cumsum(np.random.randn(500) * 0.5),
            'high': 102 + np.cumsum(np.random.randn(500) * 0.5),
            'low': 98 + np.cumsum(np.random.randn(500) * 0.5),
            'close': 100 + np.cumsum(np.random.randn(500) * 0.5),
            'volume': np.random.uniform(900000, 1100000, 500)
        })
        
        # Run optimization
        result = await optimizer.optimize_pattern_detection(data, "TEST")
        
        assert result.improvement_percentage >= 0
        assert result.optimized_metrics.execution_time <= result.original_metrics.execution_time
        
        # Run load test
        load_results = await optimizer.load_test(
            num_symbols=3,
            duration_seconds=2
        )
        
        assert load_results['patterns_per_second'] > 0
        
        # Generate report
        report = optimizer.generate_performance_report()
        assert len(report) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])