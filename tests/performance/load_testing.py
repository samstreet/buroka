"""
Comprehensive load testing framework for the Market Analysis System.
Tests API endpoints, database operations, and Kafka messaging under load.
"""

import asyncio
import aiohttp
import time
import json
import logging
import statistics
import random
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from src.utils.performance_profiler import get_performance_profiler

logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    base_url: str = "http://localhost:8000"
    duration_seconds: int = 300  # 5 minutes
    ramp_up_seconds: int = 60   # 1 minute ramp up
    concurrent_users: int = 50
    requests_per_second: int = 100
    
    # Test scenarios
    api_test_enabled: bool = True
    websocket_test_enabled: bool = True
    database_test_enabled: bool = True
    kafka_test_enabled: bool = True
    
    # Test data
    symbols: List[str] = field(default_factory=lambda: ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'])
    test_data_size: int = 1000


@dataclass
class LoadTestResult:
    """Results from load testing."""
    test_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    error_types: Dict[str, int] = field(default_factory=dict)
    throughput_rps: float = 0.0
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    @property
    def avg_response_time(self) -> float:
        """Calculate average response time."""
        return statistics.mean(self.response_times) if self.response_times else 0.0
    
    @property
    def percentiles(self) -> Dict[str, float]:
        """Calculate response time percentiles."""
        if not self.response_times:
            return {}
        
        sorted_times = sorted(self.response_times)
        n = len(sorted_times)
        
        return {
            "p50": sorted_times[int(n * 0.5)],
            "p90": sorted_times[int(n * 0.9)],
            "p95": sorted_times[int(n * 0.95)],
            "p99": sorted_times[int(n * 0.99)],
            "min": min(sorted_times),
            "max": max(sorted_times)
        }


class APILoadTester:
    """Load tester for API endpoints."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results = LoadTestResult(test_name="API Load Test")
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
    
    async def initialize(self):
        """Initialize HTTP session with optimized settings."""
        connector = aiohttp.TCPConnector(
            limit=self.config.concurrent_users * 2,
            limit_per_host=self.config.concurrent_users,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'MarketAnalysisLoadTester/1.0'}
        )
    
    async def run_load_test(self) -> LoadTestResult:
        """Run comprehensive API load test."""
        await self.initialize()
        
        try:
            self._running = True
            self.results.start_time = datetime.now(timezone.utc)
            
            # Create test scenarios
            scenarios = [
                self._test_health_endpoint,
                self._test_market_data_endpoint,
                self._test_indicators_endpoint,
                self._test_system_info_endpoint
            ]
            
            # Run load test with gradual ramp-up
            await self._run_with_ramp_up(scenarios)
            
            self.results.end_time = datetime.now(timezone.utc)
            
            # Calculate final metrics
            test_duration = (self.results.end_time - self.results.start_time).total_seconds()
            if test_duration > 0:
                self.results.throughput_rps = self.results.total_requests / test_duration
            
            return self.results
            
        finally:
            await self.cleanup()
    
    async def _run_with_ramp_up(self, scenarios: List[Callable]):
        """Run load test with gradual user ramp-up."""
        ramp_up_interval = self.config.ramp_up_seconds / self.config.concurrent_users
        test_tasks = []
        
        # Ramp up users gradually
        for user_id in range(self.config.concurrent_users):
            # Wait for ramp-up interval
            if user_id > 0:
                await asyncio.sleep(ramp_up_interval)
            
            # Start user simulation
            user_task = asyncio.create_task(self._simulate_user(user_id, scenarios))
            test_tasks.append(user_task)
            
            logger.debug(f"Started user {user_id + 1}/{self.config.concurrent_users}")
        
        # Wait for test duration
        await asyncio.sleep(self.config.duration_seconds)
        self._running = False
        
        # Wait for all tasks to complete
        await asyncio.gather(*test_tasks, return_exceptions=True)
    
    async def _simulate_user(self, user_id: int, scenarios: List[Callable]):
        """Simulate a single user's behavior."""
        requests_per_user = self.config.requests_per_second / self.config.concurrent_users
        request_interval = 1.0 / requests_per_user if requests_per_user > 0 else 1.0
        
        while self._running:
            try:
                # Randomly select a scenario
                scenario = random.choice(scenarios)
                await scenario(user_id)
                
                # Wait before next request
                await asyncio.sleep(request_interval + random.uniform(-0.1, 0.1))
                
            except Exception as e:
                logger.error(f"Error in user {user_id} simulation: {e}")
                await asyncio.sleep(1)  # Back off on error
    
    async def _test_health_endpoint(self, user_id: int):
        """Test health check endpoint."""
        await self._make_request(f"{self.config.base_url}/health", "GET")
    
    async def _test_market_data_endpoint(self, user_id: int):
        """Test market data endpoint."""
        symbol = random.choice(self.config.symbols)
        url = f"{self.config.base_url}/api/v1/market-data/{symbol}"
        await self._make_request(url, "GET")
    
    async def _test_indicators_endpoint(self, user_id: int):
        """Test technical indicators endpoint."""
        symbol = random.choice(self.config.symbols)
        indicator = random.choice(['sma', 'ema', 'rsi', 'macd'])
        url = f"{self.config.base_url}/api/v1/indicators/{indicator}/{symbol}"
        await self._make_request(url, "GET")
    
    async def _test_system_info_endpoint(self, user_id: int):
        """Test system information endpoint."""
        await self._make_request(f"{self.config.base_url}/api/v1/info", "GET")
    
    async def _make_request(self, url: str, method: str, data: Optional[Dict] = None):
        """Make HTTP request and track performance."""
        start_time = time.time()
        
        try:
            if method == "GET":
                async with self._session.get(url) as response:
                    await response.text()  # Consume response
                    status_code = response.status
            elif method == "POST":
                async with self._session.post(url, json=data) as response:
                    await response.text()
                    status_code = response.status
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response_time = time.time() - start_time
            
            # Record results
            self.results.total_requests += 1
            self.results.response_times.append(response_time)
            
            if 200 <= status_code < 400:
                self.results.successful_requests += 1
            else:
                self.results.failed_requests += 1
                error_type = f"HTTP_{status_code}"
                self.results.error_types[error_type] = self.results.error_types.get(error_type, 0) + 1
        
        except Exception as e:
            response_time = time.time() - start_time
            self.results.total_requests += 1
            self.results.failed_requests += 1
            self.results.response_times.append(response_time)
            
            error_type = type(e).__name__
            self.results.error_types[error_type] = self.results.error_types.get(error_type, 0) + 1
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._session:
            await self._session.close()


class DatabaseLoadTester:
    """Load tester for database operations."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results = LoadTestResult(test_name="Database Load Test")
        self._running = False
    
    async def run_load_test(self) -> LoadTestResult:
        """Run database load test."""
        self.results.start_time = datetime.now(timezone.utc)
        self._running = True
        
        try:
            # Create concurrent database operations
            tasks = []
            operations_per_second = self.config.requests_per_second
            
            for i in range(self.config.concurrent_users):
                task = asyncio.create_task(self._simulate_database_user(i, operations_per_second))
                tasks.append(task)
            
            # Run for specified duration
            await asyncio.sleep(self.config.duration_seconds)
            self._running = False
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            self.results.end_time = datetime.now(timezone.utc)
            
            # Calculate throughput
            test_duration = (self.results.end_time - self.results.start_time).total_seconds()
            if test_duration > 0:
                self.results.throughput_rps = self.results.total_requests / test_duration
            
            return self.results
            
        except Exception as e:
            logger.error(f"Database load test failed: {e}")
            raise
    
    async def _simulate_database_user(self, user_id: int, operations_per_second: float):
        """Simulate database operations for a user."""
        operation_interval = 1.0 / (operations_per_second / self.config.concurrent_users)
        
        while self._running:
            try:
                # Select random operation
                operation = random.choice([
                    self._test_read_operation,
                    self._test_write_operation,
                    self._test_aggregation_operation
                ])
                
                await operation()
                await asyncio.sleep(operation_interval)
                
            except Exception as e:
                logger.error(f"Database operation error for user {user_id}: {e}")
                await asyncio.sleep(1)
    
    async def _test_read_operation(self):
        """Test database read operation."""
        start_time = time.time()
        
        try:
            # Simulate database read
            await asyncio.sleep(random.uniform(0.01, 0.05))  # Simulate DB read time
            
            response_time = time.time() - start_time
            self.results.total_requests += 1
            self.results.successful_requests += 1
            self.results.response_times.append(response_time)
            
        except Exception as e:
            response_time = time.time() - start_time
            self.results.total_requests += 1
            self.results.failed_requests += 1
            self.results.response_times.append(response_time)
            
            error_type = type(e).__name__
            self.results.error_types[error_type] = self.results.error_types.get(error_type, 0) + 1
    
    async def _test_write_operation(self):
        """Test database write operation."""
        start_time = time.time()
        
        try:
            # Simulate database write
            await asyncio.sleep(random.uniform(0.02, 0.08))  # Simulate DB write time
            
            response_time = time.time() - start_time
            self.results.total_requests += 1
            self.results.successful_requests += 1
            self.results.response_times.append(response_time)
            
        except Exception as e:
            response_time = time.time() - start_time
            self.results.total_requests += 1
            self.results.failed_requests += 1
            self.results.response_times.append(response_time)
            
            error_type = type(e).__name__
            self.results.error_types[error_type] = self.results.error_types.get(error_type, 0) + 1
    
    async def _test_aggregation_operation(self):
        """Test database aggregation operation."""
        start_time = time.time()
        
        try:
            # Simulate database aggregation
            await asyncio.sleep(random.uniform(0.05, 0.15))  # Simulate aggregation time
            
            response_time = time.time() - start_time
            self.results.total_requests += 1
            self.results.successful_requests += 1
            self.results.response_times.append(response_time)
            
        except Exception as e:
            response_time = time.time() - start_time
            self.results.total_requests += 1
            self.results.failed_requests += 1
            self.results.response_times.append(response_time)
            
            error_type = type(e).__name__
            self.results.error_types[error_type] = self.results.error_types.get(error_type, 0) + 1


class KafkaLoadTester:
    """Load tester for Kafka messaging."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.producer_results = LoadTestResult(test_name="Kafka Producer Load Test")
        self.consumer_results = LoadTestResult(test_name="Kafka Consumer Load Test")
        self._running = False
    
    async def run_load_test(self) -> Tuple[LoadTestResult, LoadTestResult]:
        """Run Kafka load test for both producer and consumer."""
        self._running = True
        
        # Start producer and consumer tests concurrently
        producer_task = asyncio.create_task(self._test_kafka_producer())
        consumer_task = asyncio.create_task(self._test_kafka_consumer())
        
        # Wait for both to complete
        await asyncio.gather(producer_task, consumer_task, return_exceptions=True)
        
        return self.producer_results, self.consumer_results
    
    async def _test_kafka_producer(self):
        """Test Kafka message production under load."""
        self.producer_results.start_time = datetime.now(timezone.utc)
        
        try:
            messages_per_second = self.config.requests_per_second
            message_interval = 1.0 / messages_per_second
            
            end_time = time.time() + self.config.duration_seconds
            
            while time.time() < end_time and self._running:
                start_time = time.time()
                
                try:
                    # Create test message
                    symbol = random.choice(self.config.symbols)
                    message = {
                        "symbol": symbol,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "price": round(random.uniform(100, 500), 2),
                        "volume": random.randint(1000, 10000)
                    }
                    
                    # Simulate message sending
                    await asyncio.sleep(random.uniform(0.001, 0.005))  # Simulate send time
                    
                    response_time = time.time() - start_time
                    self.producer_results.total_requests += 1
                    self.producer_results.successful_requests += 1
                    self.producer_results.response_times.append(response_time)
                    
                except Exception as e:
                    response_time = time.time() - start_time
                    self.producer_results.total_requests += 1
                    self.producer_results.failed_requests += 1
                    self.producer_results.response_times.append(response_time)
                    
                    error_type = type(e).__name__
                    self.producer_results.error_types[error_type] = \
                        self.producer_results.error_types.get(error_type, 0) + 1
                
                await asyncio.sleep(max(0, message_interval - (time.time() - start_time)))
            
            self.producer_results.end_time = datetime.now(timezone.utc)
            
            # Calculate throughput
            test_duration = (self.producer_results.end_time - self.producer_results.start_time).total_seconds()
            if test_duration > 0:
                self.producer_results.throughput_rps = self.producer_results.total_requests / test_duration
        
        except Exception as e:
            logger.error(f"Kafka producer test failed: {e}")
    
    async def _test_kafka_consumer(self):
        """Test Kafka message consumption under load."""
        self.consumer_results.start_time = datetime.now(timezone.utc)
        
        try:
            end_time = time.time() + self.config.duration_seconds
            
            while time.time() < end_time and self._running:
                start_time = time.time()
                
                try:
                    # Simulate message processing
                    await asyncio.sleep(random.uniform(0.001, 0.01))  # Simulate processing time
                    
                    response_time = time.time() - start_time
                    self.consumer_results.total_requests += 1
                    self.consumer_results.successful_requests += 1
                    self.consumer_results.response_times.append(response_time)
                    
                except Exception as e:
                    response_time = time.time() - start_time
                    self.consumer_results.total_requests += 1
                    self.consumer_results.failed_requests += 1
                    self.consumer_results.response_times.append(response_time)
                    
                    error_type = type(e).__name__
                    self.consumer_results.error_types[error_type] = \
                        self.consumer_results.error_types.get(error_type, 0) + 1
                
                await asyncio.sleep(0.01)  # Small delay between messages
            
            self.consumer_results.end_time = datetime.now(timezone.utc)
            
            # Calculate throughput
            test_duration = (self.consumer_results.end_time - self.consumer_results.start_time).total_seconds()
            if test_duration > 0:
                self.consumer_results.throughput_rps = self.consumer_results.total_requests / test_duration
        
        except Exception as e:
            logger.error(f"Kafka consumer test failed: {e}")


class ComprehensiveLoadTester:
    """Main load tester orchestrating all test components."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: Dict[str, Any] = {}
        self.profiler = get_performance_profiler()
    
    async def run_full_load_test(self) -> Dict[str, Any]:
        """Run comprehensive load test across all components."""
        logger.info(f"Starting comprehensive load test with config: {self.config}")
        
        test_start = datetime.now(timezone.utc)
        
        # Initialize profiler
        await self.profiler.start_profiling()
        
        try:
            # Run different test components based on configuration
            test_tasks = []
            
            if self.config.api_test_enabled:
                api_tester = APILoadTester(self.config)
                test_tasks.append(('api', api_tester.run_load_test()))
            
            if self.config.database_test_enabled:
                db_tester = DatabaseLoadTester(self.config)
                test_tasks.append(('database', db_tester.run_load_test()))
            
            if self.config.kafka_test_enabled:
                kafka_tester = KafkaLoadTester(self.config)
                test_tasks.append(('kafka', kafka_tester.run_load_test()))
            
            # Run all tests concurrently
            results = []
            for test_name, test_task in test_tasks:
                try:
                    result = await test_task
                    results.append((test_name, result))
                except Exception as e:
                    logger.error(f"Test {test_name} failed: {e}")
                    results.append((test_name, None))
            
            # Process results
            self.results = {
                'test_config': {
                    'duration_seconds': self.config.duration_seconds,
                    'concurrent_users': self.config.concurrent_users,
                    'requests_per_second': self.config.requests_per_second,
                    'symbols': self.config.symbols
                },
                'test_start': test_start.isoformat(),
                'test_end': datetime.now(timezone.utc).isoformat(),
                'test_results': {}
            }
            
            # Process individual test results
            for test_name, result in results:
                if result is not None:
                    if test_name == 'kafka' and isinstance(result, tuple):
                        # Handle Kafka tuple result (producer, consumer)
                        producer_result, consumer_result = result
                        self.results['test_results']['kafka_producer'] = self._format_test_result(producer_result)
                        self.results['test_results']['kafka_consumer'] = self._format_test_result(consumer_result)
                    else:
                        self.results['test_results'][test_name] = self._format_test_result(result)
            
            # Get system performance during test
            performance_report = await self.profiler.get_performance_report()
            self.results['performance_during_test'] = performance_report
            
            # Calculate overall metrics
            self.results['overall_metrics'] = self._calculate_overall_metrics()
            
            logger.info("Comprehensive load test completed")
            return self.results
            
        finally:
            await self.profiler.stop_profiling()
    
    def _format_test_result(self, result: LoadTestResult) -> Dict[str, Any]:
        """Format test result for JSON serialization."""
        return {
            'test_name': result.test_name,
            'total_requests': result.total_requests,
            'successful_requests': result.successful_requests,
            'failed_requests': result.failed_requests,
            'success_rate_percent': result.success_rate,
            'throughput_rps': result.throughput_rps,
            'avg_response_time_ms': result.avg_response_time * 1000,
            'percentiles_ms': {k: v * 1000 for k, v in result.percentiles.items()},
            'error_types': result.error_types,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat() if result.end_time else None
        }
    
    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall test metrics."""
        total_requests = 0
        total_successful = 0
        total_failed = 0
        total_throughput = 0
        
        for test_name, test_result in self.results['test_results'].items():
            total_requests += test_result['total_requests']
            total_successful += test_result['successful_requests']
            total_failed += test_result['failed_requests']
            total_throughput += test_result['throughput_rps']
        
        return {
            'total_requests': total_requests,
            'total_successful_requests': total_successful,
            'total_failed_requests': total_failed,
            'overall_success_rate_percent': (total_successful / max(1, total_requests)) * 100,
            'combined_throughput_rps': total_throughput,
            'performance_target_met': {
                '1000_msg_per_sec': total_throughput >= 1000,
                'success_rate_above_99': (total_successful / max(1, total_requests)) >= 0.99
            }
        }
    
    async def export_results(self, file_path: str):
        """Export load test results to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"Load test results exported to {file_path}")
        except Exception as e:
            logger.error(f"Failed to export results: {e}")


# Utility functions for load testing
def create_test_data(size: int) -> List[Dict[str, Any]]:
    """Create test data for load testing."""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NFLX', 'NVDA']
    test_data = []
    
    base_time = datetime.now(timezone.utc) - timedelta(days=1)
    
    for i in range(size):
        symbol = random.choice(symbols)
        timestamp = base_time + timedelta(minutes=i)
        
        test_data.append({
            'symbol': symbol,
            'timestamp': timestamp.isoformat(),
            'open': round(random.uniform(100, 500), 2),
            'high': round(random.uniform(100, 500), 2),
            'low': round(random.uniform(100, 500), 2),
            'close': round(random.uniform(100, 500), 2),
            'volume': random.randint(1000, 100000)
        })
    
    return test_data


async def run_performance_regression_test(
    baseline_file: Optional[str] = None,
    current_config: Optional[LoadTestConfig] = None
) -> Dict[str, Any]:
    """Run performance regression test against baseline."""
    config = current_config or LoadTestConfig(
        duration_seconds=60,
        concurrent_users=25,
        requests_per_second=500
    )
    
    tester = ComprehensiveLoadTester(config)
    current_results = await tester.run_full_load_test()
    
    if baseline_file:
        try:
            with open(baseline_file, 'r') as f:
                baseline_results = json.load(f)
            
            # Compare results
            regression_analysis = _compare_performance_results(baseline_results, current_results)
            current_results['regression_analysis'] = regression_analysis
            
        except Exception as e:
            logger.warning(f"Could not load baseline file {baseline_file}: {e}")
    
    return current_results


def _compare_performance_results(baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """Compare current performance results with baseline."""
    comparison = {
        'performance_degradation': False,
        'significant_changes': [],
        'metrics_comparison': {}
    }
    
    # Compare overall metrics
    baseline_overall = baseline.get('overall_metrics', {})
    current_overall = current.get('overall_metrics', {})
    
    # Check throughput regression (>10% decrease is significant)
    baseline_throughput = baseline_overall.get('combined_throughput_rps', 0)
    current_throughput = current_overall.get('combined_throughput_rps', 0)
    
    if baseline_throughput > 0:
        throughput_change = (current_throughput - baseline_throughput) / baseline_throughput * 100
        comparison['metrics_comparison']['throughput_change_percent'] = throughput_change
        
        if throughput_change < -10:  # More than 10% decrease
            comparison['performance_degradation'] = True
            comparison['significant_changes'].append(f"Throughput decreased by {abs(throughput_change):.1f}%")
    
    # Check success rate regression
    baseline_success = baseline_overall.get('overall_success_rate_percent', 100)
    current_success = current_overall.get('overall_success_rate_percent', 100)
    
    success_change = current_success - baseline_success
    comparison['metrics_comparison']['success_rate_change_percent'] = success_change
    
    if success_change < -1:  # More than 1% decrease in success rate
        comparison['performance_degradation'] = True
        comparison['significant_changes'].append(f"Success rate decreased by {abs(success_change):.1f}%")
    
    return comparison