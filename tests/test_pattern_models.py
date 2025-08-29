import pytest
from datetime import datetime, timedelta
from typing import List
import uuid

from src.data.models.pattern_models import (
    Pattern, PatternPerformance, PatternCorrelation, PatternStatistics, Tag,
    PatternType, PatternDirection, PatternStatus, Timeframe,
    PatternDetectionResult, PatternFilter, PatternRetentionPolicy,
    PatternClassification
)
from src.data.storage.pattern_repository import PatternRepository


class TestPatternModels:
    def test_pattern_detection_result(self):
        result = PatternDetectionResult(
            pattern_type=PatternType.CANDLESTICK,
            pattern_name="Hammer",
            direction=PatternDirection.BULLISH,
            confidence=0.85,
            timeframe=Timeframe.HOUR_1,
            start_time=datetime.now() - timedelta(hours=2),
            end_time=datetime.now(),
            symbol="AAPL",
            entry_price=150.0,
            target_price=155.0,
            stop_loss=148.0,
            metadata={"volume": 1000000}
        )
        
        assert result.risk_reward_ratio == 2.5
        assert result.duration.total_seconds() == 7200
        
        result_dict = result.to_dict()
        assert result_dict['pattern_type'] == 'candlestick'
        assert result_dict['confidence'] == 0.85
        assert result_dict['risk_reward_ratio'] == 2.5
    
    def test_timeframe_seconds_conversion(self):
        assert Timeframe.MINUTE_1.seconds == 60
        assert Timeframe.HOUR_1.seconds == 3600
        assert Timeframe.DAY_1.seconds == 86400
        assert Timeframe.WEEK_1.seconds == 604800
    
    def test_pattern_filter(self):
        filter_params = PatternFilter(
            pattern_types=[PatternType.CANDLESTICK, PatternType.CHART],
            directions=[PatternDirection.BULLISH],
            min_confidence=0.7,
            symbols=["AAPL", "GOOGL"],
            limit=50
        )
        
        query_params = filter_params.to_query_params()
        assert query_params['pattern_types'] == ['candlestick', 'chart']
        assert query_params['directions'] == ['bullish']
        assert query_params['min_confidence'] == 0.7
        assert query_params['limit'] == 50
    
    def test_pattern_classification(self):
        classification = PatternClassification(
            primary_direction=PatternDirection.BULLISH,
            trend_alignment=True,
            volume_confirmation=True,
            momentum_confirmation=True,
            risk_level="low",
            confidence_factors={'pattern': 0.8, 'volume': 0.9}
        )
        
        assert classification.overall_score > 0.7
        assert classification.get_trading_bias() == "strong_buy"
        
        classification.risk_level = "high"
        classification.trend_alignment = False
        assert classification.get_trading_bias() != "strong_buy"
    
    def test_pattern_retention_policy(self):
        pattern = Pattern(
            id=uuid.uuid4(),
            symbol="AAPL",
            pattern_type=PatternType.CANDLESTICK.value,
            pattern_name="Hammer",
            direction=PatternDirection.BULLISH.value,
            confidence=0.85,
            timeframe=Timeframe.HOUR_1.value,
            start_time=datetime.now() - timedelta(hours=2),
            detected_at=datetime.now() - timedelta(days=10),
            created_at=datetime.now() - timedelta(days=10)
        )
        
        retention_days = PatternRetentionPolicy.get_retention_days(pattern)
        assert retention_days == 180
        
        pattern.confidence = 0.5
        retention_days = PatternRetentionPolicy.get_retention_days(pattern)
        assert retention_days == 30
        
        pattern.created_at = datetime.now() - timedelta(days=200)
        assert PatternRetentionPolicy.should_archive(pattern) == True


class TestPatternRepository:
    @pytest.fixture
    def repository(self):
        return PatternRepository("sqlite:///:memory:")
    
    def test_save_pattern(self, repository):
        detection_result = PatternDetectionResult(
            pattern_type=PatternType.CHART,
            pattern_name="Head and Shoulders",
            direction=PatternDirection.BEARISH,
            confidence=0.75,
            timeframe=Timeframe.DAY_1,
            start_time=datetime.now() - timedelta(days=5),
            symbol="MSFT",
            entry_price=300.0,
            target_price=280.0,
            stop_loss=305.0,
            tags=["reversal", "high_confidence"]
        )
        
        pattern = repository.save_pattern(detection_result)
        
        assert pattern.id is not None
        assert pattern.pattern_type == PatternType.CHART.value
        assert pattern.confidence == 0.75
        assert pattern.risk_reward_ratio == 4.0
        assert len(pattern.tags) == 2
    
    def test_update_pattern_status(self, repository):
        detection_result = PatternDetectionResult(
            pattern_type=PatternType.BREAKOUT,
            pattern_name="Triangle Breakout",
            direction=PatternDirection.BULLISH,
            confidence=0.8,
            timeframe=Timeframe.HOUR_4,
            start_time=datetime.now() - timedelta(hours=8),
            symbol="TSLA"
        )
        
        pattern = repository.save_pattern(detection_result)
        
        success = repository.update_pattern_status(
            pattern.id,
            PatternStatus.COMPLETED,
            outcome="success",
            actual_return=0.05
        )
        
        assert success == True
        
        patterns, _ = repository.find_patterns(PatternFilter(
            statuses=[PatternStatus.COMPLETED]
        ))
        
        assert len(patterns) == 1
        assert patterns[0].actual_return == 0.05
    
    def test_save_performance(self, repository):
        detection_result = PatternDetectionResult(
            pattern_type=PatternType.CANDLESTICK,
            pattern_name="Engulfing",
            direction=PatternDirection.BULLISH,
            confidence=0.9,
            timeframe=Timeframe.MINUTE_15,
            start_time=datetime.now() - timedelta(hours=1),
            symbol="SPY"
        )
        
        pattern = repository.save_pattern(detection_result)
        
        performance = repository.save_performance(pattern.id, {
            'hit_target': True,
            'max_favorable_excursion': 0.08,
            'time_to_target': 3600,
            'volume_ratio': 2.5,
            'follow_through_rate': 0.85
        })
        
        assert performance.hit_target == True
        assert performance.max_favorable_excursion == 0.08
        assert performance.performance_score > 0.7
    
    def test_find_patterns(self, repository):
        for i in range(10):
            detection_result = PatternDetectionResult(
                pattern_type=PatternType.CANDLESTICK if i % 2 == 0 else PatternType.CHART,
                pattern_name=f"Pattern_{i}",
                direction=PatternDirection.BULLISH if i % 3 == 0 else PatternDirection.BEARISH,
                confidence=0.5 + i * 0.05,
                timeframe=Timeframe.HOUR_1,
                start_time=datetime.now() - timedelta(hours=i),
                symbol="AAPL" if i < 5 else "GOOGL"
            )
            repository.save_pattern(detection_result)
        
        filter_params = PatternFilter(
            pattern_types=[PatternType.CANDLESTICK],
            min_confidence=0.6,
            limit=5
        )
        
        patterns, total = repository.find_patterns(filter_params)
        
        assert len(patterns) <= 5
        assert all(p.pattern_type == PatternType.CANDLESTICK.value for p in patterns)
        assert all(p.confidence >= 0.6 for p in patterns)
    
    def test_save_correlation(self, repository):
        pattern1 = repository.save_pattern(PatternDetectionResult(
            pattern_type=PatternType.CHART,
            pattern_name="Pattern1",
            direction=PatternDirection.BULLISH,
            confidence=0.8,
            timeframe=Timeframe.HOUR_1,
            start_time=datetime.now(),
            symbol="AAPL"
        ))
        
        pattern2 = repository.save_pattern(PatternDetectionResult(
            pattern_type=PatternType.VOLUME,
            pattern_name="Pattern2",
            direction=PatternDirection.BULLISH,
            confidence=0.75,
            timeframe=Timeframe.HOUR_1,
            start_time=datetime.now(),
            symbol="AAPL"
        ))
        
        correlation = repository.save_correlation(
            pattern1.id,
            pattern2.id,
            correlation_type="co_occurrence",
            correlation_strength=0.85,
            temporal_relationship="concurrent"
        )
        
        assert correlation.correlation_strength == 0.85
        
        correlations = repository.find_correlated_patterns(pattern1.id, min_correlation=0.8)
        assert len(correlations) == 1
        assert correlations[0].correlated_pattern_id == pattern2.id
    
    def test_get_pattern_statistics(self, repository):
        for i in range(5):
            detection_result = PatternDetectionResult(
                pattern_type=PatternType.CANDLESTICK,
                pattern_name="Hammer",
                direction=PatternDirection.BULLISH,
                confidence=0.8,
                timeframe=Timeframe.HOUR_1,
                start_time=datetime.now() - timedelta(hours=i),
                symbol="AAPL"
            )
            pattern = repository.save_pattern(detection_result)
            
            repository.update_pattern_status(
                pattern.id,
                PatternStatus.COMPLETED,
                outcome="success" if i % 2 == 0 else "failure",
                actual_return=0.05 if i % 2 == 0 else -0.02
            )
        
        stats = repository.get_pattern_statistics(
            pattern_type=PatternType.CANDLESTICK,
            pattern_name="Hammer",
            symbol="AAPL"
        )
        
        assert len(stats) > 0
        stat = stats[0]
        assert stat.total_occurrences == 5
        assert stat.successful_occurrences == 3
        assert stat.success_rate == 0.6
    
    def test_classify_pattern(self, repository):
        pattern = Pattern(
            id=uuid.uuid4(),
            symbol="AAPL",
            pattern_type=PatternType.BREAKOUT.value,
            pattern_name="Triangle Breakout",
            direction=PatternDirection.BULLISH.value,
            confidence=0.85,
            strength=0.75,
            timeframe=Timeframe.HOUR_1.value,
            start_time=datetime.now(),
            detected_at=datetime.now(),
            risk_reward_ratio=3.0
        )
        
        market_data = {
            'trend_direction': PatternDirection.BULLISH.value,
            'volume_confirmation': True,
            'momentum_indicators': {'rsi': 55, 'macd': 'bullish'},
            'market_phase': 'accumulation'
        }
        
        classification = repository.classify_pattern(pattern, market_data)
        
        assert classification.primary_direction == PatternDirection.BULLISH
        assert classification.trend_alignment == True
        assert classification.volume_confirmation == True
        assert classification.momentum_confirmation == True
        assert classification.risk_level == 'low'
        assert classification.suggested_action == 'enter_long'
        assert classification.overall_score > 0.7
    
    def test_cleanup_expired_patterns(self, repository):
        expired_pattern = PatternDetectionResult(
            pattern_type=PatternType.CANDLESTICK,
            pattern_name="Expired",
            direction=PatternDirection.NEUTRAL,
            confidence=0.5,
            timeframe=Timeframe.MINUTE_1,
            start_time=datetime.now() - timedelta(days=100),
            symbol="OLD"
        )
        
        pattern = repository.save_pattern(expired_pattern)
        
        with repository.get_session() as session:
            db_pattern = session.query(Pattern).filter_by(id=pattern.id).first()
            db_pattern.expires_at = datetime.now() - timedelta(days=1)
        
        deleted = repository.cleanup_expired_patterns()
        assert deleted == 1
        
        patterns, _ = repository.find_patterns(PatternFilter())
        assert len(patterns) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])