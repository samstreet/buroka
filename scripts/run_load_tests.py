#!/usr/bin/env python3
"""
Execute comprehensive load tests to validate 1000 messages/second target.
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.testing.load_test import LoadTester
from src.utils.performance_analyzer import performance_analyzer, kafka_profiler
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ComprehensiveLoadTester:
    """Execute comprehensive load tests with detailed reporting."""
    
    def __init__(self):
        self.tester = LoadTester()
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    async def test_kafka_throughput_target(self):
        """Test if Kafka can handle 1000 messages/second."""
        logger.info("=" * 80)
        logger.info("KAFKA THROUGHPUT TEST - TARGET: 1000 msg/s")
        logger.info("=" * 80)
        
        # Test different batch sizes to find optimal
        batch_sizes = [1, 10, 50, 100, 200, 500]
        best_throughput = 0
        best_batch_size = 0
        
        for batch_size in batch_sizes:
            logger.info(f"\nTesting batch size: {batch_size}")
            
            result = await self.tester.test_kafka_throughput(
                topic="market-data-raw",
                num_messages=10000,
                message_size=1024,
                batch_size=batch_size
            )
            
            throughput = result.messages_per_second
            logger.info(f"Batch size {batch_size}: {throughput:.2f} msg/s")
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size
            
            # Record for Kafka profiler
            kafka_profiler.record_batch(batch_size, result.duration)
        
        logger.info(f"\nBest configuration: Batch size {best_batch_size} = {best_throughput:.2f} msg/s")
        
        # Final test with best batch size
        logger.info(f"\nRunning final test with optimal batch size {best_batch_size}...")
        final_result = await self.tester.test_kafka_throughput(
            topic="market-data-raw",
            num_messages=100000,  # Larger test for accuracy
            message_size=1024,
            batch_size=best_batch_size
        )
        
        self.test_results['kafka_throughput'] = {
            'target': 1000,
            'achieved': final_result.messages_per_second,
            'target_met': final_result.messages_per_second >= 1000,
            'optimal_batch_size': best_batch_size,
            'error_rate': final_result.error_rate,
            'total_messages': final_result.messages_sent,
            'duration': final_result.duration
        }
        
        # Print results
        if final_result.messages_per_second >= 1000:
            logger.info(f"✅ TARGET MET: {final_result.messages_per_second:.2f} msg/s >= 1000 msg/s")
        else:
            logger.warning(f"❌ TARGET NOT MET: {final_result.messages_per_second:.2f} msg/s < 1000 msg/s")
        
        return final_result
    
    async def test_api_load(self):
        """Test API endpoints under load."""
        logger.info("\n" + "=" * 80)
        logger.info("API LOAD TEST")
        logger.info("=" * 80)
        
        endpoints = [
            {
                'path': '/health',
                'method': 'GET',
                'requests': 10000,
                'concurrent': 50,
                'target_p95': 0.2  # 200ms
            },
            {
                'path': '/api/v1/market-data/AAPL',
                'method': 'GET',
                'requests': 5000,
                'concurrent': 25,
                'target_p95': 0.2
            },
            {
                'path': '/api/v1/patterns/AAPL',
                'method': 'GET',
                'requests': 2000,
                'concurrent': 20,
                'target_p95': 0.5
            }
        ]
        
        api_results = []
        
        for endpoint in endpoints:
            logger.info(f"\nTesting {endpoint['path']}...")
            
            result = await self.tester.test_api_endpoint(
                endpoint=endpoint['path'],
                method=endpoint['method'],
                num_requests=endpoint['requests'],
                concurrent_requests=endpoint['concurrent']
            )
            
            api_results.append({
                'endpoint': endpoint['path'],
                'requests': endpoint['requests'],
                'rps': result.requests_per_second,
                'p95': result.p95_response_time,
                'target_p95': endpoint['target_p95'],
                'target_met': result.p95_response_time <= endpoint['target_p95'],
                'error_rate': result.error_rate
            })
            
            # Log result
            if result.p95_response_time <= endpoint['target_p95']:
                logger.info(f"✅ {endpoint['path']}: P95 {result.p95_response_time*1000:.2f}ms <= {endpoint['target_p95']*1000:.0f}ms")
            else:
                logger.warning(f"❌ {endpoint['path']}: P95 {result.p95_response_time*1000:.2f}ms > {endpoint['target_p95']*1000:.0f}ms")
        
        self.test_results['api_load'] = api_results
        return api_results
    
    async def test_database_load(self):
        """Test database operations under load."""
        logger.info("\n" + "=" * 80)
        logger.info("DATABASE LOAD TEST")
        logger.info("=" * 80)
        
        db_tests = [
            {
                'operation': 'read',
                'count': 5000,
                'target_avg': 0.1  # 100ms
            },
            {
                'operation': 'write',
                'count': 2000,
                'target_avg': 0.05  # 50ms
            }
        ]
        
        db_results = []
        
        for test in db_tests:
            logger.info(f"\nTesting database {test['operation']} operations...")
            
            result = await self.tester.test_database_operations(
                num_operations=test['count'],
                operation_type=test['operation']
            )
            
            db_results.append({
                'operation': test['operation'],
                'count': test['count'],
                'ops_per_sec': result.requests_per_second,
                'avg_time': result.avg_response_time,
                'target_avg': test['target_avg'],
                'target_met': result.avg_response_time <= test['target_avg'],
                'error_rate': result.error_rate
            })
            
            # Log result
            if result.avg_response_time <= test['target_avg']:
                logger.info(f"✅ DB {test['operation']}: Avg {result.avg_response_time*1000:.2f}ms <= {test['target_avg']*1000:.0f}ms")
            else:
                logger.warning(f"❌ DB {test['operation']}: Avg {result.avg_response_time*1000:.2f}ms > {test['target_avg']*1000:.0f}ms")
        
        self.test_results['database_load'] = db_results
        return db_results
    
    async def test_concurrent_load(self):
        """Test system under concurrent load from multiple sources."""
        logger.info("\n" + "=" * 80)
        logger.info("CONCURRENT LOAD TEST")
        logger.info("=" * 80)
        
        logger.info("\nSimulating realistic concurrent load...")
        
        # Run all tests concurrently
        tasks = [
            self.tester.test_kafka_throughput(
                topic="market-data-raw",
                num_messages=10000,
                batch_size=100
            ),
            self.tester.test_api_endpoint(
                "/api/v1/market-data/AAPL",
                num_requests=1000,
                concurrent_requests=25
            ),
            self.tester.test_database_operations(
                1000,
                "read"
            )
        ]
        
        start = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - start
        
        kafka_result, api_result, db_result = results
        
        self.test_results['concurrent_load'] = {
            'duration': duration,
            'kafka_throughput': kafka_result.messages_per_second,
            'api_rps': api_result.requests_per_second,
            'db_ops_per_sec': db_result.requests_per_second,
            'total_operations': (
                kafka_result.messages_sent + 
                api_result.successful_requests + 
                db_result.successful_requests
            ),
            'overall_throughput': (
                kafka_result.messages_sent + 
                api_result.successful_requests + 
                db_result.successful_requests
            ) / duration
        }
        
        logger.info(f"\nConcurrent test completed in {duration:.2f}s")
        logger.info(f"Overall throughput: {self.test_results['concurrent_load']['overall_throughput']:.2f} ops/s")
    
    async def run_full_test_suite(self):
        """Run complete load test suite."""
        self.start_time = datetime.now()
        
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE LOAD TEST SUITE")
        logger.info(f"Started: {self.start_time.isoformat()}")
        logger.info("=" * 80)
        
        # Set performance baseline
        performance_analyzer.set_baseline()
        
        # Run individual tests
        await self.test_kafka_throughput_target()
        await self.test_api_load()
        await self.test_database_load()
        await self.test_concurrent_load()
        
        self.end_time = datetime.now()
        
        # Generate final report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive load test report."""
        duration = (self.end_time - self.start_time).total_seconds()
        
        report = []
        report.append("=" * 80)
        report.append("LOAD TEST REPORT")
        report.append("=" * 80)
        report.append(f"Start Time: {self.start_time.isoformat()}")
        report.append(f"End Time: {self.end_time.isoformat()}")
        report.append(f"Total Duration: {duration:.2f} seconds")
        report.append("")
        
        # Kafka Throughput Results
        kafka_results = self.test_results.get('kafka_throughput', {})
        report.append("KAFKA THROUGHPUT TEST")
        report.append("-" * 40)
        report.append(f"Target: {kafka_results.get('target', 1000)} msg/s")
        report.append(f"Achieved: {kafka_results.get('achieved', 0):.2f} msg/s")
        report.append(f"Target Met: {'✅ YES' if kafka_results.get('target_met') else '❌ NO'}")
        report.append(f"Optimal Batch Size: {kafka_results.get('optimal_batch_size', 'N/A')}")
        report.append(f"Error Rate: {kafka_results.get('error_rate', 0):.2f}%")
        report.append("")
        
        # API Load Test Results
        report.append("API LOAD TEST RESULTS")
        report.append("-" * 40)
        api_results = self.test_results.get('api_load', [])
        for result in api_results:
            status = "✅" if result['target_met'] else "❌"
            report.append(f"{status} {result['endpoint']}")
            report.append(f"  - Requests/sec: {result['rps']:.2f}")
            report.append(f"  - P95 Latency: {result['p95']*1000:.2f}ms (target: {result['target_p95']*1000:.0f}ms)")
            report.append(f"  - Error Rate: {result['error_rate']:.2f}%")
        report.append("")
        
        # Database Load Test Results
        report.append("DATABASE LOAD TEST RESULTS")
        report.append("-" * 40)
        db_results = self.test_results.get('database_load', [])
        for result in db_results:
            status = "✅" if result['target_met'] else "❌"
            report.append(f"{status} {result['operation'].upper()} Operations")
            report.append(f"  - Operations/sec: {result['ops_per_sec']:.2f}")
            report.append(f"  - Avg Time: {result['avg_time']*1000:.2f}ms (target: {result['target_avg']*1000:.0f}ms)")
            report.append(f"  - Error Rate: {result['error_rate']:.2f}%")
        report.append("")
        
        # Concurrent Load Results
        concurrent = self.test_results.get('concurrent_load', {})
        report.append("CONCURRENT LOAD TEST RESULTS")
        report.append("-" * 40)
        report.append(f"Test Duration: {concurrent.get('duration', 0):.2f}s")
        report.append(f"Kafka Throughput: {concurrent.get('kafka_throughput', 0):.2f} msg/s")
        report.append(f"API Throughput: {concurrent.get('api_rps', 0):.2f} req/s")
        report.append(f"DB Throughput: {concurrent.get('db_ops_per_sec', 0):.2f} ops/s")
        report.append(f"Overall Throughput: {concurrent.get('overall_throughput', 0):.2f} ops/s")
        report.append("")
        
        # Overall Summary
        report.append("OVERALL SUMMARY")
        report.append("-" * 40)
        
        # Check all targets
        kafka_target_met = kafka_results.get('target_met', False)
        api_targets_met = all(r['target_met'] for r in api_results) if api_results else False
        db_targets_met = all(r['target_met'] for r in db_results) if db_results else False
        
        all_targets_met = kafka_target_met and api_targets_met and db_targets_met
        
        report.append(f"Kafka 1000 msg/s Target: {'✅ MET' if kafka_target_met else '❌ NOT MET'}")
        report.append(f"API Latency Targets: {'✅ MET' if api_targets_met else '❌ NOT MET'}")
        report.append(f"Database Performance Targets: {'✅ MET' if db_targets_met else '❌ NOT MET'}")
        report.append("")
        report.append(f"ALL TARGETS MET: {'✅ YES' if all_targets_met else '❌ NO'}")
        
        # Performance comparison to baseline
        report.append("")
        report.append("RESOURCE USAGE VS BASELINE")
        report.append("-" * 40)
        delta = performance_analyzer.compare_to_baseline()
        for metric, change in delta.items():
            sign = "+" if change > 0 else ""
            report.append(f"{metric}: {sign}{change:.2f}")
        
        report.append("")
        report.append("=" * 80)
        
        # Save and print report
        report_text = "\n".join(report)
        
        # Save to file
        report_file = f"load_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, "w") as f:
            f.write(report_text)
        
        # Print to console
        print(report_text)
        print(f"\nReport saved to: {report_file}")
        
        # Return summary
        return {
            'all_targets_met': all_targets_met,
            'kafka_target_met': kafka_target_met,
            'api_targets_met': api_targets_met,
            'db_targets_met': db_targets_met,
            'report_file': report_file
        }


async def main():
    """Main entry point for load testing."""
    tester = ComprehensiveLoadTester()
    
    try:
        await tester.run_full_test_suite()
        
        # Get Kafka profiler stats
        kafka_stats = kafka_profiler.get_stats()
        if kafka_stats:
            logger.info("\nKafka Profiler Statistics:")
            logger.info(f"  Messages Processed: {kafka_stats.get('messages_processed', 0)}")
            logger.info(f"  Avg Processing Time: {kafka_stats.get('avg_processing_time', 0)*1000:.2f}ms")
            logger.info(f"  Throughput: {kafka_stats.get('throughput', 0):.2f} msg/s")
        
    except Exception as e:
        logger.error(f"Load test failed: {e}")
        raise


if __name__ == "__main__":
    print("Starting comprehensive load tests...")
    print("This will test the system's ability to handle 1000 messages/second")
    print("-" * 80)
    
    asyncio.run(main())