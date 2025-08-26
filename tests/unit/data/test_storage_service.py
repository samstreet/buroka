"""
Tests for the data storage service.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta

from src.data.storage.service import DataStorageService, DataRetentionManager
from src.data.storage.sinks import InMemoryDataSink
from src.data.models.market_data import MarketDataType


class MockDataSink:
    """Mock data sink for testing."""
    
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.write_data_calls = []
        self.write_batch_calls = []
        self.health_check_result = True
        
    async def write_data(self, data, metadata=None):
        self.write_data_calls.append((data, metadata))
        if self.should_fail:
            return False
        return True
    
    async def write_batch(self, data_batch, metadata=None):
        self.write_batch_calls.append((data_batch, metadata))
        if self.should_fail:
            return False
        return True
    
    async def health_check(self):
        return self.health_check_result
    
    def get_stats(self):
        return {
            "total_writes": len(self.write_data_calls) + len(self.write_batch_calls),
            "failed_writes": 0
        }


class TestDataStorageService:
    """Test DataStorageService implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.primary_sink = MockDataSink()
        self.backup_sink = MockDataSink()
        
        self.service = DataStorageService(
            primary_sink=self.primary_sink,
            backup_sink=self.backup_sink,
            batch_size=5,  # Small batch size for testing
            batch_timeout=1,  # Short timeout for testing
            max_retries=2,
            retry_delay=0.1,
            deduplication_window=10
        )
    
    @pytest.mark.asyncio
    async def test_store_single_data_success(self):
        """Test successful single data storage."""
        test_data = {
            "symbol": "AAPL",
            "data_type": "quote",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": 150.0
        }
        
        result = await self.service.store_data(test_data)
        assert result is True
        
        # Check stats
        stats = await self.service.get_storage_stats()
        assert stats["total_writes"] == 1
    
    @pytest.mark.asyncio
    async def test_store_batch_data_success(self):
        """Test successful batch data storage."""
        test_batch = [
            {
                "symbol": "AAPL",
                "data_type": "quote",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "price": 150.0
            },
            {
                "symbol": "MSFT",
                "data_type": "quote",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "price": 300.0
            }
        ]
        
        result = await self.service.store_batch(test_batch)
        assert result is True
        
        # Check that data was written to primary sink
        assert len(self.primary_sink.write_batch_calls) == 1
        assert len(self.primary_sink.write_batch_calls[0][0]) == 2
        
        # Check stats
        stats = await self.service.get_storage_stats()
        assert stats["successful_writes"] == 2
        assert stats["batch_writes"] == 1
    
    @pytest.mark.asyncio
    async def test_deduplication(self):
        """Test data deduplication functionality."""
        test_data = {
            "symbol": "AAPL",
            "data_type": "quote",
            "timestamp": "2024-01-01T12:00:00Z",
            "price": 150.0
        }
        
        # Store same data twice
        result1 = await self.service.store_data(test_data)
        result2 = await self.service.store_data(test_data)
        
        assert result1 is True
        assert result2 is True  # Still returns success, but filtered as duplicate
        
        # Check stats
        stats = await self.service.get_storage_stats()
        assert stats["duplicates_filtered"] == 1
        assert stats["total_writes"] == 2
    
    @pytest.mark.asyncio
    async def test_batch_auto_flush(self):
        """Test automatic batch flushing when batch size reached."""
        # Store multiple records to trigger batch flush
        for i in range(6):  # More than batch_size (5)
            test_data = {
                "symbol": f"STOCK{i}",
                "data_type": "quote",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "price": 100.0 + i
            }
            await self.service.store_data(test_data)
        
        # Small delay to allow batch processing
        await asyncio.sleep(0.1)
        
        # Check that at least one batch was flushed
        stats = await self.service.get_storage_stats()
        assert stats["batch_writes"] >= 1
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test retry mechanism when primary sink fails."""
        # Make primary sink fail
        self.primary_sink.should_fail = True
        
        test_data = {
            "symbol": "AAPL",
            "data_type": "quote",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": 150.0
        }
        
        # Store batch to trigger retry logic
        result = await self.service.store_batch([test_data])
        
        # Should succeed via backup sink
        assert result is True
        
        # Check stats
        stats = await self.service.get_storage_stats()
        assert stats["backup_writes"] == 1
        assert stats["retry_attempts"] > 0
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test comprehensive health check."""
        health = await self.service.health_check()
        
        assert "overall" in health
        assert "components" in health
        assert health["overall"] == "healthy"
        assert "primary_sink" in health["components"]
        assert "backup_sink" in health["components"]
        assert "batch_processor" in health["components"]
    
    @pytest.mark.asyncio
    async def test_health_check_degraded(self):
        """Test health check when primary sink is unhealthy."""
        self.primary_sink.health_check_result = False
        
        health = await self.service.health_check()
        
        assert health["overall"] == "degraded"  # Should be degraded, not unhealthy (has backup)
        assert health["components"]["primary_sink"]["status"] == "unhealthy"
        assert health["components"]["backup_sink"]["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_flush_all_batches(self):
        """Test manual batch flushing."""
        # Add some data to batch buffer
        test_data = {
            "symbol": "AAPL",
            "data_type": "quote",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": 150.0
        }
        await self.service.store_data(test_data)
        
        # Manually flush batches
        result = await self.service.flush_all_batches()
        assert result is True
        
        # Check that batch was written
        stats = await self.service.get_storage_stats()
        assert stats["batch_buffer_size"] == 0
    
    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test graceful shutdown."""
        # Add some data
        test_data = {
            "symbol": "AAPL",
            "data_type": "quote",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": 150.0
        }
        await self.service.store_data(test_data)
        
        # Shutdown should flush remaining data
        await self.service.shutdown()
        
        # Verify background task was cancelled
        assert self.service._batch_task.cancelled() or self.service._batch_task.done()
    
    @pytest.mark.asyncio
    async def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, 'service'):
            await self.service.shutdown()


class TestDataRetentionManager:
    """Test DataRetentionManager implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.storage_service = MagicMock()
        self.retention_manager = DataRetentionManager(self.storage_service)
    
    @pytest.mark.asyncio
    async def test_apply_retention_policies(self):
        """Test applying retention policies."""
        # Mock cleanup_old_data to return some counts
        cleanup_results = [5, 10, 2, 0, 1, 3, 8]  # Results for each policy
        self.storage_service.cleanup_old_data.side_effect = cleanup_results
        
        result = await self.retention_manager.apply_retention_policies()
        
        # Should have results for all default policies
        assert len(result) == len(self.retention_manager.retention_policies)
        
        # Check that cleanup_old_data was called for each policy
        assert self.storage_service.cleanup_old_data.call_count == len(self.retention_manager.retention_policies)
        
        # Check specific policy results
        assert "intraday_1m" in result
        assert "daily" in result
        assert "quote" in result
    
    def test_add_retention_policy(self):
        """Test adding custom retention policy."""
        initial_count = len(self.retention_manager.retention_policies)
        
        self.retention_manager.add_retention_policy("custom_policy", timedelta(days=14))
        
        assert len(self.retention_manager.retention_policies) == initial_count + 1
        assert "custom_policy" in self.retention_manager.retention_policies
        assert self.retention_manager.retention_policies["custom_policy"] == timedelta(days=14)
    
    def test_remove_retention_policy(self):
        """Test removing retention policy."""
        # Add a policy first
        self.retention_manager.add_retention_policy("temp_policy", timedelta(days=7))
        
        # Remove it
        result = self.retention_manager.remove_retention_policy("temp_policy")
        
        assert result is True
        assert "temp_policy" not in self.retention_manager.retention_policies
    
    def test_remove_nonexistent_policy(self):
        """Test removing non-existent retention policy."""
        result = self.retention_manager.remove_retention_policy("nonexistent_policy")
        assert result is False


class TestStorageServiceIntegration:
    """Integration tests for storage service with real sinks."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.primary_sink = InMemoryDataSink(max_records=1000)
        self.backup_sink = InMemoryDataSink(max_records=500)
        
        self.service = DataStorageService(
            primary_sink=self.primary_sink,
            backup_sink=self.backup_sink,
            batch_size=10,
            batch_timeout=2,
            max_retries=3,
            deduplication_window=30
        )
    
    @pytest.mark.asyncio
    async def test_end_to_end_storage(self):
        """Test end-to-end storage workflow."""
        # Create sample market data
        sample_data = []
        for i in range(25):  # More than batch size
            data = {
                "symbol": f"STOCK{i % 5}",  # 5 different stocks
                "data_type": "quote",
                "timestamp": (datetime.now(timezone.utc) + timedelta(seconds=i)).isoformat(),
                "price": 100.0 + i,
                "volume": 1000 + i * 10
            }
            sample_data.append(data)
        
        # Store data in batches
        batch_size = 5
        for i in range(0, len(sample_data), batch_size):
            batch = sample_data[i:i + batch_size]
            result = await self.service.store_batch(batch)
            assert result is True
        
        # Flush any remaining batches
        await self.service.flush_all_batches()
        
        # Verify data was stored
        records = self.primary_sink.get_records()
        assert len(records) > 0
        
        # Verify statistics
        stats = await self.service.get_storage_stats()
        assert stats["successful_writes"] == len(sample_data)
        assert stats["success_rate"] == 100.0
        assert stats["batch_writes"] > 0
    
    @pytest.mark.asyncio
    async def test_deduplication_with_real_data(self):
        """Test deduplication with realistic market data."""
        base_timestamp = datetime.now(timezone.utc)
        
        # Create duplicate quote data (same symbol, timestamp, type)
        quote_data = {
            "symbol": "AAPL",
            "data_type": "quote",
            "timestamp": base_timestamp.isoformat(),
            "bid_price": 149.50,
            "ask_price": 150.00,
            "last_price": 149.75
        }
        
        # Store same quote multiple times
        for _ in range(5):
            await self.service.store_data(quote_data)
        
        # Create different quote (different timestamp)
        different_quote = {
            **quote_data,
            "timestamp": (base_timestamp + timedelta(seconds=1)).isoformat(),
            "last_price": 149.80
        }
        await self.service.store_data(different_quote)
        
        # Flush batches
        await self.service.flush_all_batches()
        
        # Check deduplication stats
        stats = await self.service.get_storage_stats()
        assert stats["duplicates_filtered"] == 4  # 4 out of 5 duplicates
        assert stats["total_writes"] == 6  # 5 + 1
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test storage service performance under load."""
        import time
        
        # Generate large dataset
        large_dataset = []
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        base_time = datetime.now(timezone.utc)
        
        for i in range(1000):  # 1000 records
            data = {
                "symbol": symbols[i % len(symbols)],
                "data_type": "quote",
                "timestamp": (base_time + timedelta(seconds=i)).isoformat(),
                "price": 100.0 + (i % 100),
                "volume": 1000 + i
            }
            large_dataset.append(data)
        
        # Measure storage time
        start_time = time.time()
        
        # Store in batches of 50
        for i in range(0, len(large_dataset), 50):
            batch = large_dataset[i:i + 50]
            result = await self.service.store_batch(batch)
            assert result is True
        
        # Flush remaining batches
        await self.service.flush_all_batches()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        assert total_time < 5.0  # Should complete in under 5 seconds
        
        # Verify all data was stored
        stats = await self.service.get_storage_stats()
        assert stats["successful_writes"] == len(large_dataset)
        assert stats["success_rate"] == 100.0
        
        # Check storage efficiency
        records = self.primary_sink.get_records()
        assert len(records) > 0
    
    @pytest.mark.asyncio
    async def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, 'service'):
            await self.service.shutdown()