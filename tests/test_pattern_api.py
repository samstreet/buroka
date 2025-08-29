import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, WebSocket
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import json
import asyncio

from src.api.routers.patterns import router, get_repository, get_detection_service, websocket_manager
from src.data.models.pattern_models import (
    Pattern, PatternType, PatternDirection, PatternStatus, Timeframe,
    PatternDetectionResult, PatternStatistics
)
from src.data.storage.pattern_repository import PatternRepository
from src.services.pattern_detection_service import PatternDetectionService


app = FastAPI()
app.include_router(router)


@pytest.fixture
def mock_repository():
    repo = Mock(spec=PatternRepository)
    
    mock_patterns = [
        Mock(
            spec=Pattern,
            id="pattern1",
            symbol="AAPL",
            pattern_type="candlestick",
            pattern_name="Hammer",
            direction="bullish",
            status="active",
            timeframe="1h",
            detected_at=datetime.utcnow(),
            start_time=datetime.utcnow() - timedelta(hours=2),
            end_time=None,
            duration_seconds=None,
            confidence=0.85,
            strength=0.75,
            quality_score=0.8,
            entry_price=150.0,
            target_price=155.0,
            stop_loss=148.0,
            risk_reward_ratio=2.5,
            actual_outcome=None,
            actual_return=None,
            pattern_data={"test": "data"},
            metadata={}
        ),
        Mock(
            spec=Pattern,
            id="pattern2",
            symbol="AAPL",
            pattern_type="chart",
            pattern_name="Triangle",
            direction="neutral",
            status="completed",
            timeframe="4h",
            detected_at=datetime.utcnow() - timedelta(days=1),
            start_time=datetime.utcnow() - timedelta(days=2),
            end_time=datetime.utcnow() - timedelta(hours=12),
            duration_seconds=129600,
            confidence=0.72,
            strength=0.65,
            quality_score=0.7,
            entry_price=148.0,
            target_price=152.0,
            stop_loss=146.0,
            risk_reward_ratio=2.0,
            actual_outcome="success",
            actual_return=0.025,
            pattern_data={},
            metadata={}
        )
    ]
    
    repo.find_patterns = Mock(return_value=(mock_patterns, 2))
    repo.get_pattern_statistics = Mock(return_value=[
        Mock(
            spec=PatternStatistics,
            pattern_type="candlestick",
            pattern_name="Hammer",
            symbol="AAPL",
            timeframe="1h",
            total_occurrences=100,
            successful_occurrences=65,
            failed_occurrences=35,
            success_rate=0.65,
            avg_confidence=0.78,
            avg_return=0.032,
            avg_duration=7200,
            avg_risk_reward=2.3,
            best_return=0.15,
            worst_return=-0.08,
            std_dev_return=0.045,
            last_occurrence=datetime.utcnow(),
            last_success=datetime.utcnow() - timedelta(hours=6),
            last_failure=datetime.utcnow() - timedelta(days=1)
        )
    ])
    
    return repo


@pytest.fixture
def mock_detection_service(mock_repository):
    service = Mock(spec=PatternDetectionService)
    service.repository = mock_repository
    service.get_pattern_performance = Mock(return_value={
        'total_patterns': 50,
        'successful': 30,
        'failed': 15,
        'success_rate': 0.6,
        'avg_return': 0.025,
        'best_return': 0.12,
        'worst_return': -0.05,
        'patterns_by_type': {'candlestick': 25, 'chart': 25},
        'patterns_by_direction': {'bullish': 30, 'bearish': 20}
    })
    return service


@pytest.fixture
def client(mock_repository, mock_detection_service):
    app.dependency_overrides[get_repository] = lambda: mock_repository
    app.dependency_overrides[get_detection_service] = lambda: mock_detection_service
    
    client = TestClient(app)
    yield client
    
    app.dependency_overrides.clear()


class TestPatternEndpoints:
    
    def test_get_patterns_by_symbol(self, client, mock_repository):
        response = client.get("/api/v1/patterns/AAPL")
        
        assert response.status_code == 200
        patterns = response.json()
        assert len(patterns) == 2
        assert patterns[0]['symbol'] == "AAPL"
        assert patterns[0]['confidence'] == 0.85
        
        mock_repository.find_patterns.assert_called_once()
        
    def test_get_patterns_with_filters(self, client, mock_repository):
        response = client.get(
            "/api/v1/patterns/AAPL",
            params={
                "pattern_type": "candlestick",
                "timeframe": "1h",
                "min_confidence": 0.7,
                "status": "active",
                "days_back": 7,
                "limit": 10
            }
        )
        
        assert response.status_code == 200
        patterns = response.json()
        assert isinstance(patterns, list)
        
        call_args = mock_repository.find_patterns.call_args[0][0]
        assert call_args.symbols == ["AAPL"]
        assert call_args.min_confidence == 0.7
        assert call_args.limit == 10
        
    def test_search_patterns(self, client, mock_repository):
        search_request = {
            "symbols": ["AAPL", "MSFT"],
            "pattern_types": ["candlestick", "chart"],
            "directions": ["bullish"],
            "min_confidence": 0.75,
            "limit": 50,
            "offset": 0
        }
        
        response = client.post("/api/v1/patterns/search", json=search_request)
        
        assert response.status_code == 200
        result = response.json()
        assert 'patterns' in result
        assert 'total_count' in result
        assert 'has_more' in result
        assert result['total_count'] == 2
        
    def test_get_pattern_performance(self, client, mock_detection_service):
        response = client.get(
            "/api/v1/patterns/AAPL/performance",
            params={
                "pattern_type": "candlestick",
                "days": 30
            }
        )
        
        assert response.status_code == 200
        performance = response.json()
        assert performance['symbol'] == "AAPL"
        assert performance['total_patterns'] == 50
        assert performance['success_rate'] == 0.6
        assert 'daily_distribution' in performance
        
    def test_get_pattern_statistics(self, client, mock_repository):
        response = client.get(
            "/api/v1/patterns/statistics/candlestick/Hammer",
            params={"symbol": "AAPL", "timeframe": "1h"}
        )
        
        assert response.status_code == 200
        stats = response.json()
        assert stats['pattern_name'] == "Hammer"
        assert stats['success_rate'] == 0.65
        assert stats['total_occurrences'] == 100
        
    def test_pattern_subscription(self, client):
        subscription_request = {
            "symbols": ["AAPL", "MSFT"],
            "pattern_types": ["candlestick"],
            "min_confidence": 0.8,
            "notification_channels": ["websocket", "webhook"],
            "webhook_url": "https://example.com/webhook"
        }
        
        response = client.post("/api/v1/patterns/subscribe", json=subscription_request)
        
        assert response.status_code == 200
        result = response.json()
        assert 'subscription_id' in result
        assert result['status'] == 'active'
        assert result['symbols'] == ["AAPL", "MSFT"]
        
    def test_get_pattern_visualization(self, client, mock_repository):
        mock_pattern = Mock(
            spec=Pattern,
            id="pattern1",
            symbol="AAPL",
            pattern_type="candlestick",
            pattern_name="Hammer",
            direction="bullish",
            entry_price=150.0,
            target_price=155.0,
            stop_loss=148.0,
            pattern_data={
                'price_points': [
                    {'time': '2024-01-01T10:00:00', 'value': 150.0},
                    {'time': '2024-01-01T11:00:00', 'value': 151.0}
                ],
                'pattern_points': [
                    {'time': '2024-01-01T10:30:00', 'price': 150.5, 'label': 'Entry'}
                ],
                'indicators': {'rsi': [50, 55, 60]},
                'support_resistance': [148.0, 152.0, 156.0]
            }
        )
        
        mock_repository.find_patterns = Mock(return_value=([mock_pattern], 1))
        
        response = client.get("/api/v1/patterns/visualization/pattern1")
        
        assert response.status_code == 200
        visualization = response.json()
        assert visualization['pattern_id'] == "pattern1"
        assert len(visualization['chart_data']) == 2
        assert len(visualization['annotations']) >= 4
        assert 'indicators' in visualization
        assert len(visualization['support_resistance_levels']) == 3
        
    def test_discover_trending_patterns(self, client, mock_repository):
        mock_patterns = [
            Mock(
                spec=Pattern,
                pattern_type="candlestick",
                pattern_name="Hammer",
                symbol="AAPL",
                confidence=0.8,
                direction="bullish",
                detected_at=datetime.utcnow()
            ) for _ in range(5)
        ] + [
            Mock(
                spec=Pattern,
                pattern_type="chart",
                pattern_name="Triangle",
                symbol="MSFT",
                confidence=0.75,
                direction="neutral",
                detected_at=datetime.utcnow()
            ) for _ in range(3)
        ]
        
        mock_repository.find_patterns = Mock(return_value=(mock_patterns, 8))
        
        response = client.get(
            "/api/v1/patterns/discovery/trending",
            params={"hours": 24, "min_occurrences": 3}
        )
        
        assert response.status_code == 200
        trending = response.json()
        assert len(trending) >= 1
        assert trending[0]['occurrences'] >= 3
        assert 'avg_confidence' in trending[0]
        assert 'dominant_direction' in trending[0]


class TestWebSocketManager:
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        manager = websocket_manager
        mock_websocket = Mock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        
        await manager.connect(mock_websocket, "client1")
        
        assert "client1" in manager.active_connections
        assert mock_websocket in manager.active_connections["client1"]
        mock_websocket.accept.assert_called_once()
        
        manager.disconnect(mock_websocket, "client1")
        assert "client1" not in manager.active_connections
        
    @pytest.mark.asyncio
    async def test_broadcast_pattern(self):
        manager = websocket_manager
        mock_websocket = Mock(spec=WebSocket)
        mock_websocket.send_json = AsyncMock()
        
        manager.subscriptions[mock_websocket] = {
            'symbols': ['AAPL'],
            'pattern_types': ['candlestick'],
            'min_confidence': 0.7
        }
        
        pattern = PatternDetectionResult(
            pattern_type=PatternType.CANDLESTICK,
            pattern_name="Hammer",
            direction=PatternDirection.BULLISH,
            confidence=0.8,
            timeframe=Timeframe.HOUR_1,
            start_time=datetime.now(),
            symbol="AAPL"
        )
        
        await manager.broadcast_pattern(pattern)
        
        mock_websocket.send_json.assert_called_once()
        sent_data = mock_websocket.send_json.call_args[0][0]
        assert sent_data['pattern_name'] == "Hammer"
        assert sent_data['symbol'] == "AAPL"
        
        del manager.subscriptions[mock_websocket]
        
    def test_matches_subscription(self):
        manager = websocket_manager
        
        pattern = PatternDetectionResult(
            pattern_type=PatternType.CANDLESTICK,
            pattern_name="Hammer",
            direction=PatternDirection.BULLISH,
            confidence=0.8,
            timeframe=Timeframe.HOUR_1,
            start_time=datetime.now(),
            symbol="AAPL"
        )
        
        subscription = {
            'symbols': ['AAPL', 'MSFT'],
            'pattern_types': ['candlestick'],
            'min_confidence': 0.7
        }
        
        assert manager._matches_subscription(pattern, subscription) == True
        
        subscription['symbols'] = ['MSFT']
        assert manager._matches_subscription(pattern, subscription) == False
        
        subscription['symbols'] = ['AAPL']
        subscription['min_confidence'] = 0.9
        assert manager._matches_subscription(pattern, subscription) == False


class TestPatternSearchRequest:
    
    def test_pattern_search_validation(self):
        from src.api.routers.patterns import PatternSearchRequest
        
        request = PatternSearchRequest(
            symbols=["AAPL"],
            pattern_types=["candlestick"],
            min_confidence=0.5,
            max_confidence=0.9,
            limit=100
        )
        
        assert request.pattern_types == ["CANDLESTICK"]
        assert request.min_confidence == 0.5
        assert request.limit == 100
        
    def test_invalid_confidence_range(self):
        from src.api.routers.patterns import PatternSearchRequest
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            PatternSearchRequest(
                min_confidence=1.5
            )
            
        with pytest.raises(ValidationError):
            PatternSearchRequest(
                limit=1001
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])