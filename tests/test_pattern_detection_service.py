import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import redis

from src.services.pattern_detection_service import (
    PatternDetectionService,
    PatternDetector,
    CandlestickDetector,
    ChartPatternDetector,
    VolumePatternDetector,
    DetectorPriority,
    NotificationChannel,
    PatternNotification
)
from src.data.models.pattern_models import (
    PatternType, PatternDirection, PatternStatus, Timeframe,
    PatternDetectionResult, Pattern
)
from src.data.storage.pattern_repository import PatternRepository


class MockDetector(PatternDetector):
    def __init__(self, name: str = "MockDetector"):
        super().__init__(name, PatternType.TECHNICAL, DetectorPriority.MEDIUM)
        
    async def detect(self, data: pd.DataFrame, symbol: str, timeframe: Timeframe) -> List[PatternDetectionResult]:
        return [
            PatternDetectionResult(
                pattern_type=self.pattern_type,
                pattern_name="MockPattern",
                direction=PatternDirection.BULLISH,
                confidence=0.8,
                timeframe=timeframe,
                start_time=datetime.now(),
                symbol=symbol
            )
        ]


class TestPatternDetectors:
    def setup_method(self):
        self.dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        self.data = pd.DataFrame({
            'timestamp': self.dates,
            'open': np.random.uniform(99, 101, 100),
            'high': np.random.uniform(101, 103, 100),
            'low': np.random.uniform(97, 99, 100),
            'close': np.random.uniform(99, 101, 100),
            'volume': np.random.uniform(900000, 1100000, 100)
        })
        
    @pytest.mark.asyncio
    async def test_candlestick_detector(self):
        detector = CandlestickDetector()
        
        self.data.loc[50, 'open'] = 100
        self.data.loc[50, 'high'] = 100.5
        self.data.loc[50, 'low'] = 98
        self.data.loc[50, 'close'] = 100.3
        
        results = await detector.detect(self.data, "AAPL", Timeframe.HOUR_1)
        
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, PatternDetectionResult)
            assert result.pattern_type == PatternType.CANDLESTICK
            assert result.symbol == "AAPL"
            assert result.timeframe == Timeframe.HOUR_1
            
    @pytest.mark.asyncio
    async def test_chart_pattern_detector(self):
        detector = ChartPatternDetector()
        
        results = await detector.detect(self.data, "MSFT", Timeframe.DAY_1)
        
        assert isinstance(results, list)
        for result in results:
            assert result.pattern_type == PatternType.CHART
            assert result.symbol == "MSFT"
            
    @pytest.mark.asyncio
    async def test_volume_pattern_detector(self):
        detector = VolumePatternDetector()
        
        self.data.loc[60:65, 'volume'] = 3000000
        
        results = await detector.detect(self.data, "GOOGL", Timeframe.MINUTE_15)
        
        assert isinstance(results, list)
        for result in results:
            assert result.pattern_type == PatternType.VOLUME
            assert result.symbol == "GOOGL"


class TestPatternDetectionService:
    @pytest.fixture
    def mock_repository(self):
        repo = Mock(spec=PatternRepository)
        repo.save_pattern = Mock(return_value=Mock(spec=Pattern, id="test_id"))
        repo.find_patterns = Mock(return_value=([], 0))
        repo.save_correlation = Mock()
        return repo
        
    @pytest.fixture
    def mock_redis(self):
        redis_client = Mock(spec=redis.Redis)
        redis_client.get = Mock(return_value=None)
        redis_client.setex = Mock()
        return redis_client
        
    @pytest.fixture
    def service(self, mock_repository, mock_redis):
        return PatternDetectionService(mock_repository, mock_redis)
        
    def test_register_detector(self, service):
        detector = MockDetector("TestDetector")
        service.register_detector(detector)
        
        assert "TestDetector" in service.detectors
        assert service.detectors["TestDetector"] == detector
        
    def test_unregister_detector(self, service):
        detector = MockDetector("TestDetector")
        service.register_detector(detector)
        service.unregister_detector("TestDetector")
        
        assert "TestDetector" not in service.detectors
        
    def test_register_notification_handler(self, service):
        handler = Mock()
        service.register_notification_handler(NotificationChannel.WEBSOCKET, handler)
        
        assert NotificationChannel.WEBSOCKET in service.notification_handlers
        assert service.notification_handlers[NotificationChannel.WEBSOCKET] == handler
        
    @pytest.mark.asyncio
    async def test_scan_for_patterns(self, service, mock_repository):
        detector = MockDetector()
        service.register_detector(detector)
        
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1h'),
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000000] * 10
        })
        
        results = await service.scan_for_patterns(data, "AAPL", Timeframe.HOUR_1)
        
        assert len(results) > 0
        assert mock_repository.save_pattern.called
        
    @pytest.mark.asyncio
    async def test_scan_with_caching(self, service, mock_redis):
        detector = MockDetector()
        service.register_detector(detector)
        
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1h'),
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000000] * 10
        })
        
        results = await service.scan_for_patterns(data, "AAPL", Timeframe.HOUR_1)
        
        assert mock_redis.setex.called
        cache_call = mock_redis.setex.call_args
        assert cache_call[0][1] == service.cache_ttl
        
    @pytest.mark.asyncio
    async def test_filter_and_rank_patterns(self, service):
        patterns = [
            PatternDetectionResult(
                pattern_type=PatternType.CANDLESTICK,
                pattern_name=f"Pattern_{i}",
                direction=PatternDirection.BULLISH,
                confidence=0.5 + i * 0.1,
                timeframe=Timeframe.HOUR_1,
                start_time=datetime.now(),
                symbol="AAPL",
                strength=0.7 + i * 0.05
            )
            for i in range(5)
        ]
        
        filtered = await service.filter_and_rank_patterns(patterns)
        
        assert len(filtered) <= len(patterns)
        assert all(p.confidence >= 0.6 for p in filtered)
        
        for i in range(len(filtered) - 1):
            assert filtered[i].confidence >= filtered[i + 1].confidence
            
    @pytest.mark.asyncio
    async def test_analyze_correlations(self, service, mock_repository):
        patterns = [
            PatternDetectionResult(
                pattern_type=PatternType.CANDLESTICK,
                pattern_name="Pattern1",
                direction=PatternDirection.BULLISH,
                confidence=0.8,
                timeframe=Timeframe.HOUR_1,
                start_time=datetime.now(),
                symbol="AAPL"
            ),
            PatternDetectionResult(
                pattern_type=PatternType.VOLUME,
                pattern_name="Pattern2",
                direction=PatternDirection.BULLISH,
                confidence=0.75,
                timeframe=Timeframe.HOUR_1,
                start_time=datetime.now() + timedelta(minutes=5),
                symbol="AAPL"
            )
        ]
        
        mock_repository.find_patterns = Mock(return_value=([Mock(id=f"id_{i}") for i in range(2)], 2))
        
        await service.analyze_correlations(patterns)
        
        assert mock_repository.save_correlation.called
        
    @pytest.mark.asyncio
    async def test_send_notifications(self, service):
        handler = AsyncMock()
        service.register_notification_handler(NotificationChannel.WEBSOCKET, handler)
        
        patterns = [
            PatternDetectionResult(
                pattern_type=PatternType.CANDLESTICK,
                pattern_name="HighConfidencePattern",
                direction=PatternDirection.BULLISH,
                confidence=0.95,
                timeframe=Timeframe.HOUR_1,
                start_time=datetime.now(),
                symbol="AAPL"
            )
        ]
        
        await service.send_notifications(patterns)
        
        handler.assert_called_once()
        notification = handler.call_args[0][0]
        assert isinstance(notification, PatternNotification)
        assert notification.priority == DetectorPriority.CRITICAL
        
    @pytest.mark.asyncio
    async def test_real_time_scanning(self, service):
        data_source = AsyncMock(return_value=pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1h'),
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000000] * 10
        }))
        
        await service.start_real_time_scanning("AAPL", Timeframe.HOUR_1, data_source, interval_seconds=0.1)
        
        task_key = f"AAPL_{Timeframe.HOUR_1.value}"
        assert task_key in service.scanning_tasks
        
        await asyncio.sleep(0.2)
        
        await service.stop_real_time_scanning("AAPL", Timeframe.HOUR_1)
        assert task_key not in service.scanning_tasks
        
    @pytest.mark.asyncio
    async def test_get_active_patterns(self, service, mock_repository):
        mock_patterns = [Mock(spec=Pattern) for _ in range(3)]
        mock_repository.find_patterns = Mock(return_value=(mock_patterns, 3))
        
        patterns = await service.get_active_patterns("AAPL", Timeframe.HOUR_1)
        
        assert len(patterns) == 3
        filter_call = mock_repository.find_patterns.call_args[0][0]
        assert filter_call.symbols == ["AAPL"]
        assert filter_call.timeframes == [Timeframe.HOUR_1]
        assert PatternStatus.ACTIVE in filter_call.statuses
        
    @pytest.mark.asyncio
    async def test_get_pattern_performance(self, service, mock_repository):
        mock_patterns = [
            Mock(spec=Pattern, actual_outcome="success", actual_return=0.05, 
                 pattern_type="candlestick", direction="bullish"),
            Mock(spec=Pattern, actual_outcome="success", actual_return=0.03,
                 pattern_type="chart", direction="bullish"),
            Mock(spec=Pattern, actual_outcome="failure", actual_return=-0.02,
                 pattern_type="candlestick", direction="bearish")
        ]
        mock_repository.find_patterns = Mock(return_value=(mock_patterns, 3))
        
        performance = await service.get_pattern_performance("AAPL", PatternType.CANDLESTICK, days_back=30)
        
        assert performance['total_patterns'] == 3
        assert performance['successful'] == 2
        assert performance['failed'] == 1
        assert performance['success_rate'] == 2/3
        assert performance['avg_return'] == pytest.approx(0.02, rel=1e-3)
        assert performance['best_return'] == 0.05
        assert performance['worst_return'] == -0.02
        
    def test_notification_priority_determination(self, service):
        high_confidence = PatternDetectionResult(
            pattern_type=PatternType.CANDLESTICK,
            pattern_name="Pattern",
            direction=PatternDirection.BULLISH,
            confidence=0.92,
            timeframe=Timeframe.HOUR_1,
            start_time=datetime.now(),
            symbol="AAPL"
        )
        
        priority = service._determine_notification_priority(high_confidence)
        assert priority == DetectorPriority.CRITICAL
        
        low_confidence = PatternDetectionResult(
            pattern_type=PatternType.CANDLESTICK,
            pattern_name="Pattern",
            direction=PatternDirection.BULLISH,
            confidence=0.65,
            timeframe=Timeframe.HOUR_1,
            start_time=datetime.now(),
            symbol="AAPL"
        )
        
        priority = service._determine_notification_priority(low_confidence)
        assert priority == DetectorPriority.LOW
        
    def test_correlation_detection(self, service):
        p1 = PatternDetectionResult(
            pattern_type=PatternType.CANDLESTICK,
            pattern_name="Pattern1",
            direction=PatternDirection.BULLISH,
            confidence=0.8,
            timeframe=Timeframe.HOUR_1,
            start_time=datetime.now(),
            symbol="AAPL"
        )
        
        p2 = PatternDetectionResult(
            pattern_type=PatternType.VOLUME,
            pattern_name="Pattern2",
            direction=PatternDirection.BULLISH,
            confidence=0.75,
            timeframe=Timeframe.HOUR_1,
            start_time=datetime.now() + timedelta(minutes=30),
            symbol="AAPL"
        )
        
        assert service._are_patterns_correlated(p1, p2) == True
        
        p3 = PatternDetectionResult(
            pattern_type=PatternType.VOLUME,
            pattern_name="Pattern3",
            direction=PatternDirection.BEARISH,
            confidence=0.75,
            timeframe=Timeframe.HOUR_1,
            start_time=datetime.now() + timedelta(hours=12),
            symbol="AAPL"
        )
        
        assert service._are_patterns_correlated(p1, p3) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])