"""
Load testing framework for performance validation.
"""

import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
from aiokafka import AIOKafkaProducer
from prometheus_client import CollectorRegistry, Gauge, Histogram, push_to_gateway

from src.config import settings
from src.utils.logging_config import get_logger
from src.utils.performance_analyzer import performance_analyzer

logger = get_logger(__name__)


@dataclass
class LoadTestResult:
    """Results from a load test run."""
    test_name: str
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    messages_sent: int = 0
    messages_per_second: float = 0


class LoadTester:
    """Main load testing orchestrator."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[LoadTestResult] = []
        self.response_times: List[float] = []
        self.errors: List[str] = []
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self.request_duration = Histogram(
            'load_test_request_duration_seconds',
            'Request duration in seconds',
            registry=self.registry
        )
        self.error_gauge = Gauge(
            'load_test_errors_total',
            'Total number of errors',
            registry=self.registry
        )
    
    async def test_api_endpoint(
        self,
        endpoint: str,
        method: str = "GET",
        payload: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        num_requests: int = 1000,
        concurrent_requests: int = 10
    ) -> LoadTestResult:
        """Load test an API endpoint."""
        logger.info(f"Starting load test for {method} {endpoint}")
        
        start_time = time.time()
        successful = 0
        failed = 0
        response_times = []
        
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(concurrent_requests)
            
            async def make_request():
                async with semaphore:
                    try:
                        request_start = time.time()
                        
                        async with session.request(
                            method,
                            f"{self.base_url}{endpoint}",
                            json=payload,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            await response.text()
                            request_time = time.time() - request_start
                            
                            response_times.append(request_time)
                            self.request_duration.observe(request_time)
                            
                            if response.status < 400:
                                return True
                            else:
                                self.errors.append(f"HTTP {response.status}")
                                return False
                    except Exception as e:
                        self.errors.append(str(e))
                        self.error_gauge.inc()
                        return False
            
            # Execute requests
            tasks = [make_request() for _ in range(num_requests)]
            results = await asyncio.gather(*tasks)
            
            successful = sum(1 for r in results if r)
            failed = sum(1 for r in results if not r)
        
        duration = time.time() - start_time
        
        # Calculate metrics
        if response_times:
            result = LoadTestResult(
                test_name=f"{method} {endpoint}",
                duration=duration,
                total_requests=num_requests,
                successful_requests=successful,
                failed_requests=failed,
                avg_response_time=np.mean(response_times),
                p50_response_time=np.percentile(response_times, 50),
                p95_response_time=np.percentile(response_times, 95),
                p99_response_time=np.percentile(response_times, 99),
                requests_per_second=num_requests / duration,
                error_rate=(failed / num_requests) * 100
            )
        else:
            result = LoadTestResult(
                test_name=f"{method} {endpoint}",
                duration=duration,
                total_requests=num_requests,
                successful_requests=0,
                failed_requests=num_requests,
                avg_response_time=0,
                p50_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=0,
                error_rate=100
            )
        
        self.results.append(result)
        logger.info(f"Load test completed: {result.requests_per_second:.2f} req/s, "
                   f"Error rate: {result.error_rate:.2f}%")
        
        return result
    
    async def test_kafka_throughput(
        self,
        topic: str = "market-data-raw",
        num_messages: int = 10000,
        message_size: int = 1024,
        batch_size: int = 100
    ) -> LoadTestResult:
        """Test Kafka message throughput."""
        logger.info(f"Starting Kafka throughput test: {num_messages} messages")
        
        producer = AIOKafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            compression_type='gzip',
            batch_size=batch_size * message_size
        )
        
        await producer.start()
        
        try:
            start_time = time.time()
            successful = 0
            failed = 0
            
            # Generate test messages
            for i in range(0, num_messages, batch_size):
                batch = []
                for j in range(min(batch_size, num_messages - i)):
                    message = {
                        'id': i + j,
                        'timestamp': datetime.now().isoformat(),
                        'data': 'x' * (message_size - 100),  # Account for metadata
                        'value': random.random() * 1000
                    }
                    batch.append(producer.send(
                        topic,
                        value=str(message).encode('utf-8')
                    ))
                
                # Send batch
                results = await asyncio.gather(*batch, return_exceptions=True)
                successful += sum(1 for r in results if not isinstance(r, Exception))
                failed += sum(1 for r in results if isinstance(r, Exception))
            
            duration = time.time() - start_time
            
            result = LoadTestResult(
                test_name=f"Kafka {topic}",
                duration=duration,
                total_requests=num_messages,
                successful_requests=successful,
                failed_requests=failed,
                avg_response_time=duration / num_messages,
                p50_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=0,
                error_rate=(failed / num_messages) * 100,
                messages_sent=successful,
                messages_per_second=successful / duration
            )
            
            self.results.append(result)
            logger.info(f"Kafka test completed: {result.messages_per_second:.2f} msg/s")
            
            return result
            
        finally:
            await producer.stop()
    
    async def test_database_operations(
        self,
        num_operations: int = 1000,
        operation_type: str = "read"
    ) -> LoadTestResult:
        """Test database operation throughput."""
        from src.data.storage.optimized_clients import OptimizedInfluxDBClient
        
        logger.info(f"Starting database {operation_type} test: {num_operations} operations")
        
        client = OptimizedInfluxDBClient()
        start_time = time.time()
        successful = 0
        failed = 0
        operation_times = []
        
        for i in range(num_operations):
            op_start = time.time()
            try:
                if operation_type == "write":
                    # Test write operation
                    point = {
                        'measurement': 'load_test',
                        'tags': {'test': 'performance'},
                        'time': datetime.now(),
                        'fields': {
                            'value': random.random() * 1000,
                            'iteration': i
                        }
                    }
                    await client.write_point(point)
                else:
                    # Test read operation
                    query = f"SELECT * FROM market_data WHERE time > now() - 1h LIMIT 10"
                    await client.query(query)
                
                successful += 1
                operation_times.append(time.time() - op_start)
            except Exception as e:
                failed += 1
                self.errors.append(str(e))
        
        duration = time.time() - start_time
        
        if operation_times:
            result = LoadTestResult(
                test_name=f"Database {operation_type}",
                duration=duration,
                total_requests=num_operations,
                successful_requests=successful,
                failed_requests=failed,
                avg_response_time=np.mean(operation_times),
                p50_response_time=np.percentile(operation_times, 50),
                p95_response_time=np.percentile(operation_times, 95),
                p99_response_time=np.percentile(operation_times, 99),
                requests_per_second=num_operations / duration,
                error_rate=(failed / num_operations) * 100
            )
        else:
            result = LoadTestResult(
                test_name=f"Database {operation_type}",
                duration=duration,
                total_requests=num_operations,
                successful_requests=0,
                failed_requests=num_operations,
                avg_response_time=0,
                p50_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                requests_per_second=0,
                error_rate=100
            )
        
        self.results.append(result)
        logger.info(f"Database test completed: {result.requests_per_second:.2f} ops/s")
        
        return result
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive load test suite."""
        logger.info("Starting comprehensive load test suite")
        
        with performance_analyzer.profile_code("comprehensive_load_test"):
            # Test API endpoints
            api_results = await asyncio.gather(
                self.test_api_endpoint("/health", num_requests=1000),
                self.test_api_endpoint("/api/v1/market-data/AAPL", num_requests=500),
                self.test_api_endpoint(
                    "/api/v1/auth/login",
                    method="POST",
                    payload={"username": "test", "password": "test"},
                    num_requests=100
                ),
            )
            
            # Test Kafka throughput
            kafka_result = await self.test_kafka_throughput(
                num_messages=10000,
                batch_size=100
            )
            
            # Test database operations
            db_results = await asyncio.gather(
                self.test_database_operations(1000, "read"),
                self.test_database_operations(500, "write")
            )
        
        # Aggregate results
        total_requests = sum(r.total_requests for r in self.results)
        total_successful = sum(r.successful_requests for r in self.results)
        total_failed = sum(r.failed_requests for r in self.results)
        avg_error_rate = (total_failed / total_requests) * 100 if total_requests > 0 else 0
        
        # Check against targets
        targets_met = {
            'api_latency': all(r.p95_response_time < 0.2 for r in api_results),  # 200ms target
            'kafka_throughput': kafka_result.messages_per_second > 1000,  # 1000 msg/s target
            'error_rate': avg_error_rate < 1,  # <1% error rate
            'database_performance': all(r.avg_response_time < 0.1 for r in db_results)  # 100ms target
        }
        
        summary = {
            'total_tests': len(self.results),
            'total_requests': total_requests,
            'total_successful': total_successful,
            'total_failed': total_failed,
            'overall_error_rate': avg_error_rate,
            'targets_met': targets_met,
            'all_targets_met': all(targets_met.values()),
            'detailed_results': [
                {
                    'test': r.test_name,
                    'rps': r.requests_per_second,
                    'p95': r.p95_response_time,
                    'error_rate': r.error_rate
                }
                for r in self.results
            ]
        }
        
        return summary
    
    def generate_report(self) -> str:
        """Generate load test report."""
        report = ["=" * 80]
        report.append("LOAD TEST REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        for result in self.results:
            report.append(f"\nTest: {result.test_name}")
            report.append("-" * 40)
            report.append(f"Duration: {result.duration:.2f}s")
            report.append(f"Total Requests: {result.total_requests}")
            report.append(f"Successful: {result.successful_requests}")
            report.append(f"Failed: {result.failed_requests}")
            report.append(f"Error Rate: {result.error_rate:.2f}%")
            report.append(f"Requests/Second: {result.requests_per_second:.2f}")
            
            if result.avg_response_time > 0:
                report.append(f"Avg Response Time: {result.avg_response_time*1000:.2f}ms")
                report.append(f"P50 Response Time: {result.p50_response_time*1000:.2f}ms")
                report.append(f"P95 Response Time: {result.p95_response_time*1000:.2f}ms")
                report.append(f"P99 Response Time: {result.p99_response_time*1000:.2f}ms")
            
            if result.messages_sent > 0:
                report.append(f"Messages Sent: {result.messages_sent}")
                report.append(f"Messages/Second: {result.messages_per_second:.2f}")
        
        # Performance targets
        report.append("\nPERFORMANCE TARGETS")
        report.append("-" * 40)
        
        # Check API latency target (200ms)
        api_results = [r for r in self.results if "api" in r.test_name.lower()]
        if api_results:
            p95_met = all(r.p95_response_time < 0.2 for r in api_results)
            report.append(f"API Latency (<200ms P95): {'✓ PASS' if p95_met else '✗ FAIL'}")
        
        # Check Kafka throughput target (1000 msg/s)
        kafka_results = [r for r in self.results if "kafka" in r.test_name.lower()]
        if kafka_results:
            throughput_met = any(r.messages_per_second > 1000 for r in kafka_results)
            report.append(f"Kafka Throughput (>1000 msg/s): {'✓ PASS' if throughput_met else '✗ FAIL'}")
        
        # Check error rate target (<1%)
        if self.results:
            avg_error = sum(r.error_rate for r in self.results) / len(self.results)
            error_met = avg_error < 1
            report.append(f"Error Rate (<1%): {'✓ PASS' if error_met else '✗ FAIL'} ({avg_error:.2f}%)")
        
        # Top errors
        if self.errors:
            report.append("\nTOP ERRORS")
            report.append("-" * 40)
            from collections import Counter
            error_counts = Counter(self.errors).most_common(5)
            for error, count in error_counts:
                report.append(f"  {error}: {count} occurrences")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)


async def main():
    """Run load tests."""
    tester = LoadTester()
    
    # Run comprehensive test
    summary = await tester.run_comprehensive_test()
    
    # Generate and save report
    report = tester.generate_report()
    
    # Save to file
    with open("load_test_report.txt", "w") as f:
        f.write(report)
    
    # Print summary
    print(report)
    print(f"\nAll targets met: {summary['all_targets_met']}")
    
    return summary


if __name__ == "__main__":
    asyncio.run(main())