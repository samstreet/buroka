"""
Tests for the data ingestion service.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.data.ingestion.service import (
    DataIngestionService, IngestionJob, JobStatus, 
    SimpleCircuitBreaker, IngestionScheduler
)
from src.data.ingestion.transformers import MarketDataTransformer
from src.data.ingestion.metrics import InMemoryMetricsCollector
from src.data.storage.sinks import InMemoryDataSink
from src.data.models.market_data import MarketDataResponse, MarketDataType, DataGranularity


class MockDataSource:
    """Mock data source for testing."""
    
    def __init__(self):
        self.validate_symbol_result = True
        self.get_quote_result = None
        self.get_daily_data_result = None
        
    def validate_symbol(self, symbol: str) -> bool:
        return self.validate_symbol_result
    
    async def get_quote(self, symbol: str) -> MarketDataResponse:
        if self.get_quote_result:
            return self.get_quote_result
        
        return MarketDataResponse(
            symbol=symbol,
            data_type=MarketDataType.QUOTE,
            timestamp=datetime.now(timezone.utc),
            data={
                "quote": {
                    "symbol": symbol,
                    "timestamp": datetime.now(timezone.utc),
                    "bid_price": 99.50,
                    "ask_price": 100.00,
                    "bid_size": 100,
                    "ask_size": 200,
                    "last_price": 99.75,
                    "last_size": 50,
                    "change": 0.25,
                    "change_percent": 0.25
                }
            },
            success=True
        )
    
    async def get_daily_data(self, symbol: str, outputsize: str = "compact") -> MarketDataResponse:
        if self.get_daily_data_result:
            return self.get_daily_data_result
            
        return MarketDataResponse(
            symbol=symbol,
            data_type=MarketDataType.DAILY,
            timestamp=datetime.now(timezone.utc),
            data={
                "ohlc_data": [{
                    "symbol": symbol,
                    "timestamp": datetime.now(timezone.utc),
                    "open_price": 100.00,
                    "high_price": 102.00,
                    "low_price": 99.00,
                    "close_price": 101.00,
                    "volume": 1000000,
                    "adjusted_close": None,
                    "granularity": "daily",
                    "data_type": "daily"
                }]
            },
            success=True
        )
    
    async def get_intraday_data(self, symbol: str, interval: str) -> MarketDataResponse:
        return await self.get_daily_data(symbol)  # Simplified for tests
    
    async def search_symbols(self, keywords: str) -> MarketDataResponse:
        return MarketDataResponse(
            symbol=keywords,
            data_type=MarketDataType.QUOTE,
            timestamp=datetime.now(timezone.utc),
            data={"results": []},
            success=True
        )


class TestSimpleCircuitBreaker:
    """Test SimpleCircuitBreaker implementation."""
    
    def test_initialization(self):
        """Test circuit breaker initialization."""
        cb = SimpleCircuitBreaker(failure_threshold=3, timeout=30)
        assert cb.get_state() == "closed"
        assert cb.get_failure_count() == 0
    
    @pytest.mark.asyncio
    async def test_successful_calls(self):
        """Test circuit breaker with successful calls."""
        cb = SimpleCircuitBreaker()
        
        async def successful_func():
            return "success"
        
        result = await cb.call(successful_func)
        assert result == "success"
        assert cb.get_state() == "closed"
        assert cb.get_failure_count() == 0
    
    @pytest.mark.asyncio
    async def test_failure_counting(self):
        """Test circuit breaker failure counting."""
        cb = SimpleCircuitBreaker(failure_threshold=3)
        
        async def failing_func():
            raise Exception("Test failure")
        
        # First two failures should keep circuit closed
        for _ in range(2):
            with pytest.raises(Exception):
                await cb.call(failing_func)
        
        assert cb.get_state() == "closed"
        assert cb.get_failure_count() == 2
        
        # Third failure should open the circuit
        with pytest.raises(Exception):
            await cb.call(failing_func)
        
        assert cb.get_state() == "open"
        assert cb.get_failure_count() == 3
    
    @pytest.mark.asyncio
    async def test_open_circuit_rejection(self):
        """Test that open circuit rejects calls."""
        cb = SimpleCircuitBreaker(failure_threshold=1)
        
        async def failing_func():
            raise Exception("Test failure")
        
        # Trigger circuit opening
        with pytest.raises(Exception):
            await cb.call(failing_func)
        
        assert cb.get_state() == "open"
        
        # Next call should be rejected
        with pytest.raises(Exception) as exc_info:
            await cb.call(failing_func)
        
        assert "Circuit breaker is open" in str(exc_info.value)
    
    def test_reset(self):
        """Test circuit breaker reset."""
        cb = SimpleCircuitBreaker()
        cb.failure_count = 5
        cb.state = "open"
        
        cb.reset()
        
        assert cb.get_state() == "closed"
        assert cb.get_failure_count() == 0


class TestIngestionScheduler:
    """Test IngestionScheduler implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scheduler = IngestionScheduler()
    
    @pytest.mark.asyncio
    async def test_schedule_ingestion(self):
        """Test scheduling ingestion job."""
        job_id = await self.scheduler.schedule_ingestion(
            "AAPL", MarketDataType.QUOTE, 300
        )
        
        assert job_id is not None
        assert len(job_id) > 0
        
        jobs = await self.scheduler.get_active_jobs()
        assert len(jobs) == 1
        assert jobs[0]["symbol"] == "AAPL"
        assert jobs[0]["data_type"] == "quote"
        assert jobs[0]["interval"] == 300
    
    @pytest.mark.asyncio
    async def test_cancel_ingestion(self):
        """Test cancelling ingestion job."""
        job_id = await self.scheduler.schedule_ingestion(
            "AAPL", MarketDataType.QUOTE, 300
        )
        
        success = await self.scheduler.cancel_ingestion(job_id)
        assert success is True
        
        jobs = await self.scheduler.get_active_jobs()
        assert len(jobs) == 0  # Cancelled jobs are not active
    
    @pytest.mark.asyncio
    async def test_pause_resume_ingestion(self):
        """Test pausing and resuming ingestion job."""
        job_id = await self.scheduler.schedule_ingestion(
            "AAPL", MarketDataType.QUOTE, 300
        )
        
        # Start the job first
        self.scheduler.jobs[job_id].status = JobStatus.RUNNING
        
        # Pause
        success = await self.scheduler.pause_ingestion(job_id)
        assert success is True
        assert self.scheduler.jobs[job_id].status == JobStatus.PAUSED
        
        # Resume
        success = await self.scheduler.resume_ingestion(job_id)
        assert success is True
        assert self.scheduler.jobs[job_id].status == JobStatus.RUNNING
    
    @pytest.mark.asyncio
    async def test_get_ready_jobs(self):
        """Test getting ready jobs."""
        job_id = await self.scheduler.schedule_ingestion(
            "AAPL", MarketDataType.QUOTE, 300
        )
        
        ready_jobs = await self.scheduler.get_ready_jobs()
        assert len(ready_jobs) == 1
        assert ready_jobs[0].job_id == job_id


class TestDataIngestionService:
    """Test DataIngestionService implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_source = MockDataSource()
        self.data_transformer = MarketDataTransformer()
        self.data_sink = InMemoryDataSink()
        self.metrics_collector = InMemoryMetricsCollector()
        self.scheduler = IngestionScheduler()
        self.circuit_breaker = SimpleCircuitBreaker()
        
        self.service = DataIngestionService(
            data_source=self.data_source,
            data_transformer=self.data_transformer,
            data_sink=self.data_sink,
            metrics_collector=self.metrics_collector,
            scheduler=self.scheduler,
            circuit_breaker=self.circuit_breaker
        )
    
    @pytest.mark.asyncio
    async def test_ingest_symbol_data_success(self):
        """Test successful symbol data ingestion."""
        result = await self.service.ingest_symbol_data("AAPL", MarketDataType.QUOTE)
        
        assert result["success"] is True
        assert result["symbol"] == "AAPL"
        assert len(result["records"]) > 0
        
        # Check metrics were recorded
        metrics = self.metrics_collector.get_metrics_summary()
        assert "ingestion_requests_total" in str(metrics["counters"])
        assert "ingestion_requests_success_total" in str(metrics["counters"])
    
    @pytest.mark.asyncio
    async def test_ingest_symbol_data_invalid_symbol(self):
        """Test ingestion with invalid symbol."""
        self.data_source.validate_symbol_result = False
        
        result = await self.service.ingest_symbol_data("INVALID", MarketDataType.QUOTE)
        
        assert result["success"] is False
        assert "Invalid symbol" in result["error"]
        
        # Check validation error metric
        metrics = self.metrics_collector.get_metrics_summary()
        assert "ingestion_validation_errors_total" in str(metrics["counters"])
    
    @pytest.mark.asyncio
    async def test_ingest_symbol_data_api_failure(self):
        """Test ingestion with API failure."""
        # Mock API failure
        self.data_source.get_quote_result = MarketDataResponse(
            symbol="AAPL",
            data_type=MarketDataType.QUOTE,
            timestamp=datetime.now(timezone.utc),
            data={},
            success=False,
            error_message="API Error"
        )
        
        result = await self.service.ingest_symbol_data("AAPL", MarketDataType.QUOTE)
        
        assert result["success"] is False
        assert "API Error" in result["error"]
        
        # Check failure metric
        metrics = self.metrics_collector.get_metrics_summary()
        assert "ingestion_requests_failed_total" in str(metrics["counters"])
    
    @pytest.mark.asyncio
    async def test_ingest_batch_data(self):
        """Test batch data ingestion."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        result = await self.service.ingest_batch_data(symbols, MarketDataType.QUOTE)
        
        assert result["success"] is True
        assert result["total_symbols"] == 3
        assert result["successful"] <= 3
        assert len(result["results"]) == 3
    
    @pytest.mark.asyncio
    async def test_start_continuous_ingestion(self):
        """Test starting continuous ingestion."""
        symbols = ["AAPL", "MSFT"]
        
        batch_job_id = await self.service.start_continuous_ingestion(symbols, interval=60)
        
        assert batch_job_id is not None
        
        # Check jobs were scheduled
        active_jobs = await self.scheduler.get_active_jobs()
        assert len(active_jobs) == 2
        
        # Check background task started
        assert self.service._background_task is not None
    
    @pytest.mark.asyncio
    async def test_get_ingestion_status(self):
        """Test getting ingestion status."""
        # Schedule some jobs first
        await self.service.start_continuous_ingestion(["AAPL"], interval=60)
        
        status = await self.service.get_ingestion_status()
        
        assert "active_jobs" in status
        assert "circuit_breaker_state" in status
        assert "metrics" in status
        assert status["active_jobs"] > 0
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check."""
        health = await self.service.health_check()
        
        assert "overall" in health
        assert "components" in health
        assert health["overall"] in ["healthy", "degraded", "unhealthy"]
        assert "data_sink" in health["components"]
        assert "circuit_breaker" in health["components"]
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration."""
        # Mock API to always fail
        async def failing_get_quote(symbol):
            raise Exception("API Error")
        
        self.data_source.get_quote = failing_get_quote
        
        # Make enough calls to trigger circuit breaker
        for _ in range(6):  # More than failure threshold
            try:
                await self.service.ingest_symbol_data("AAPL", MarketDataType.QUOTE)
            except:
                pass
        
        # Circuit breaker should be open
        assert self.circuit_breaker.get_state() == "open"
    
    @pytest.mark.asyncio
    async def test_data_sink_integration(self):
        """Test data sink integration."""
        result = await self.service.ingest_symbol_data("AAPL", MarketDataType.QUOTE)
        
        assert result["success"] is True
        
        # Check data was written to sink
        records = self.data_sink.get_records()
        assert len(records) > 0
        assert records[-1]["data"]["symbol"] == "AAPL"
    
    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test service shutdown."""
        # Start continuous ingestion
        await self.service.start_continuous_ingestion(["AAPL"], interval=60)
        
        # Shutdown
        await self.service.shutdown()
        
        # Background task should be cancelled
        assert self.service._background_task.cancelled() or self.service._background_task.done()
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, 'service') and self.service._background_task:
            self.service._background_task.cancel()