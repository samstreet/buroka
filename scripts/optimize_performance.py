#!/usr/bin/env python3
"""
Performance optimization script for the market analysis system.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import get_logger
from src.utils.performance_analyzer import (
    performance_analyzer,
    db_profiler,
    kafka_profiler
)
from src.data.storage.database_optimizer import DatabaseOptimizer
from src.data.messaging.optimized_kafka import OptimizedKafkaConfig
from src.testing.load_test import LoadTester

logger = get_logger(__name__)


class PerformanceOptimizer:
    """Main performance optimization orchestrator."""
    
    def __init__(self):
        self.optimizations_applied = []
        self.performance_gains = {}
    
    async def optimize_database(self):
        """Optimize database performance."""
        logger.info("Optimizing database performance...")
        
        optimizer = DatabaseOptimizer()
        
        # Create indexes
        with performance_analyzer.profile_code("database_indexing"):
            indexes_created = await optimizer.create_indexes()
            self.optimizations_applied.append(f"Created {len(indexes_created)} database indexes")
        
        # Optimize queries
        with performance_analyzer.profile_code("query_optimization"):
            await optimizer.optimize_common_queries()
            self.optimizations_applied.append("Optimized common database queries")
        
        # Configure connection pooling
        pool_config = {
            'min_size': 10,
            'max_size': 50,
            'max_queries': 1000,
            'max_inactive_connection_lifetime': 300
        }
        await optimizer.configure_connection_pool(pool_config)
        self.optimizations_applied.append("Configured database connection pooling")
        
        logger.info("Database optimization completed")
    
    async def optimize_kafka(self):
        """Optimize Kafka configuration."""
        logger.info("Optimizing Kafka configuration...")
        
        config = OptimizedKafkaConfig()
        
        # Producer optimizations
        producer_config = {
            'batch_size': 32768,  # 32KB batches
            'linger_ms': 10,  # Wait up to 10ms for batching
            'compression_type': 'snappy',
            'buffer_memory': 67108864,  # 64MB buffer
            'max_in_flight_requests_per_connection': 5,
            'acks': 1  # Leader acknowledgment only for better throughput
        }
        
        config.apply_producer_config(producer_config)
        self.optimizations_applied.append("Applied Kafka producer optimizations")
        
        # Consumer optimizations
        consumer_config = {
            'fetch_min_bytes': 1024,  # 1KB minimum fetch
            'fetch_max_wait_ms': 100,  # Max wait time
            'max_partition_fetch_bytes': 1048576,  # 1MB per partition
            'session_timeout_ms': 30000,
            'heartbeat_interval_ms': 3000,
            'enable_auto_commit': False  # Manual commit for better control
        }
        
        config.apply_consumer_config(consumer_config)
        self.optimizations_applied.append("Applied Kafka consumer optimizations")
        
        logger.info("Kafka optimization completed")
    
    async def optimize_caching(self):
        """Optimize caching strategies."""
        logger.info("Optimizing caching strategies...")
        
        from src.utils.redis_client import get_redis_client
        
        redis_client = await get_redis_client()
        
        # Configure cache settings
        cache_config = {
            'default_ttl': 300,  # 5 minutes default TTL
            'max_memory': '256mb',
            'eviction_policy': 'allkeys-lru',
            'maxmemory_samples': 5
        }
        
        for key, value in cache_config.items():
            await redis_client.config_set(key.replace('_', '-'), value)
        
        self.optimizations_applied.append("Configured Redis caching strategy")
        
        # Implement cache warming for frequently accessed data
        await self._warm_cache()
        self.optimizations_applied.append("Warmed cache with frequently accessed data")
        
        logger.info("Caching optimization completed")
    
    async def _warm_cache(self):
        """Warm cache with frequently accessed data."""
        from src.api.routers.market_data import get_market_data_cached
        
        # Pre-load popular symbols
        popular_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        
        for symbol in popular_symbols:
            try:
                await get_market_data_cached(symbol)
                logger.info(f"Warmed cache for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to warm cache for {symbol}: {e}")
    
    async def optimize_api_performance(self):
        """Optimize API performance."""
        logger.info("Optimizing API performance...")
        
        # Import FastAPI app
        from src.main import app
        from fastapi import Response
        from starlette.middleware.gzip import GZipMiddleware
        
        # Add compression middleware
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        self.optimizations_applied.append("Added response compression")
        
        # Configure response caching headers
        @app.middleware("http")
        async def add_cache_headers(request, call_next):
            response = await call_next(request)
            
            # Cache static responses
            if request.url.path.startswith("/api/v1/market-data"):
                response.headers["Cache-Control"] = "public, max-age=60"
            elif request.url.path == "/health":
                response.headers["Cache-Control"] = "public, max-age=10"
            
            return response
        
        self.optimizations_applied.append("Configured HTTP caching headers")
        
        logger.info("API optimization completed")
    
    async def run_optimization_suite(self):
        """Run complete optimization suite."""
        logger.info("Starting comprehensive performance optimization...")
        
        # Set baseline
        performance_analyzer.set_baseline()
        
        # Run initial load test
        logger.info("Running baseline load test...")
        tester = LoadTester()
        baseline_results = await tester.test_api_endpoint("/health", num_requests=100)
        baseline_throughput = baseline_results.requests_per_second
        
        # Apply optimizations
        await self.optimize_database()
        await self.optimize_kafka()
        await self.optimize_caching()
        await self.optimize_api_performance()
        
        # Run post-optimization load test
        logger.info("Running post-optimization load test...")
        optimized_results = await tester.test_api_endpoint("/health", num_requests=100)
        optimized_throughput = optimized_results.requests_per_second
        
        # Calculate improvements
        improvement = ((optimized_throughput - baseline_throughput) / baseline_throughput) * 100
        self.performance_gains['throughput'] = improvement
        
        # Compare to baseline
        metrics_delta = performance_analyzer.compare_to_baseline()
        
        # Generate report
        self.generate_optimization_report(
            baseline_throughput,
            optimized_throughput,
            metrics_delta
        )
        
        logger.info(f"Optimization complete. Throughput improved by {improvement:.2f}%")
    
    def generate_optimization_report(
        self,
        baseline_throughput: float,
        optimized_throughput: float,
        metrics_delta: dict
    ):
        """Generate optimization report."""
        report = ["=" * 80]
        report.append("PERFORMANCE OPTIMIZATION REPORT")
        report.append("=" * 80]
        report.append("")
        
        report.append("OPTIMIZATIONS APPLIED")
        report.append("-" * 40)
        for optimization in self.optimizations_applied:
            report.append(f"✓ {optimization}")
        report.append("")
        
        report.append("PERFORMANCE IMPROVEMENTS")
        report.append("-" * 40)
        report.append(f"Baseline Throughput: {baseline_throughput:.2f} req/s")
        report.append(f"Optimized Throughput: {optimized_throughput:.2f} req/s")
        report.append(f"Improvement: {self.performance_gains.get('throughput', 0):.2f}%")
        report.append("")
        
        report.append("RESOURCE USAGE CHANGES")
        report.append("-" * 40)
        for metric, delta in metrics_delta.items():
            sign = "+" if delta > 0 else ""
            report.append(f"{metric}: {sign}{delta:.2f}")
        report.append("")
        
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        suggestions = performance_analyzer.get_optimization_suggestions()
        for suggestion in suggestions:
            report.append(f"• {suggestion}")
        
        report.append("\n" + "=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        with open("optimization_report.txt", "w") as f:
            f.write(report_text)
        
        print(report_text)


async def main():
    """Main optimization entry point."""
    optimizer = PerformanceOptimizer()
    await optimizer.run_optimization_suite()
    
    # Generate performance analysis report
    perf_report = performance_analyzer.generate_report()
    with open("performance_analysis.txt", "w") as f:
        f.write(perf_report)
    
    print("\nOptimization complete. Reports saved to:")
    print("  - optimization_report.txt")
    print("  - performance_analysis.txt")


if __name__ == "__main__":
    asyncio.run(main())