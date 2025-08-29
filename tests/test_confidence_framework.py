import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

from src.core.confidence.confidence_framework import (
    ConfidenceFramework,
    AdaptiveConfidenceFramework,
    ConfidenceScore,
    ConfidenceComponent,
    MarketCondition
)


class TestConfidenceScore:
    def test_confidence_level_classification(self):
        score = ConfidenceScore(
            overall_score=0.92,
            components={},
            weights={}
        )
        assert score.confidence_level == "VERY_HIGH"
        assert score.recommendation == "STRONG_SIGNAL"
        
        score.overall_score = 0.45
        assert score.confidence_level == "LOW"
        assert score.recommendation == "NO_ACTION"
        
    def test_component_contribution(self):
        score = ConfidenceScore(
            overall_score=0.75,
            components={
                ConfidenceComponent.STATISTICAL_SIGNIFICANCE: 0.8,
                ConfidenceComponent.HISTORICAL_SUCCESS: 0.7
            },
            weights={
                ConfidenceComponent.STATISTICAL_SIGNIFICANCE: 0.3,
                ConfidenceComponent.HISTORICAL_SUCCESS: 0.25
            }
        )
        
        contrib = score.get_component_contribution(ConfidenceComponent.STATISTICAL_SIGNIFICANCE)
        assert contrib == pytest.approx(0.24, rel=1e-3)
        
        contrib = score.get_component_contribution(ConfidenceComponent.HISTORICAL_SUCCESS)
        assert contrib == pytest.approx(0.175, rel=1e-3)


class TestConfidenceFramework:
    def setup_method(self):
        self.framework = ConfidenceFramework()
        self.dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        self.historical_data = pd.DataFrame({
            'timestamp': self.dates,
            'open': np.random.uniform(99, 101, 100),
            'high': np.random.uniform(101, 103, 100),
            'low': np.random.uniform(97, 99, 100),
            'close': np.random.uniform(99, 101, 100),
            'volume': np.random.uniform(900000, 1100000, 100)
        })
        
    def test_default_weights(self):
        assert len(self.framework.weights) == 5
        assert pytest.approx(sum(self.framework.weights.values()), rel=1e-2) == 1.0
        assert self.framework.weights[ConfidenceComponent.STATISTICAL_SIGNIFICANCE] == 0.30
        
    def test_custom_weights(self):
        custom_weights = {
            ConfidenceComponent.STATISTICAL_SIGNIFICANCE: 0.4,
            ConfidenceComponent.HISTORICAL_SUCCESS: 0.3,
            ConfidenceComponent.VOLUME_CONFIRMATION: 0.3
        }
        framework = ConfidenceFramework(custom_weights)
        assert pytest.approx(sum(framework.weights.values()), rel=1e-2) == 1.0
        
    def test_calculate_confidence_basic(self):
        pattern_data = {
            'pattern_type': 'breakout',
            'pattern_name': 'triangle_breakout',
            'direction': 'bullish',
            'historical_success_rate': 0.75,
            'sample_size': 50,
            'avg_return': 0.035,
            'win_loss_ratio': 1.8
        }
        
        market_data = {
            'volume': 1500000,
            'avg_volume': 1000000,
            'trend_strength': 0.6,
            'volatility': 0.15
        }
        
        score = self.framework.calculate_confidence(pattern_data, market_data, self.historical_data)
        
        assert isinstance(score, ConfidenceScore)
        assert 0 <= score.overall_score <= 1
        assert len(score.components) == 5
        assert all(0 <= s <= 1 for s in score.components.values())
        
    def test_statistical_significance_calculation(self):
        pattern_data = {
            'price_points': [100, 102, 104, 103, 105],
            'pattern_points': [
                {'time': '2024-01-01', 'price': 100},
                {'time': '2024-01-02', 'price': 102},
                {'time': '2024-01-03', 'price': 104},
                {'time': '2024-01-04', 'price': 103},
                {'time': '2024-01-05', 'price': 105}
            ],
            'r_squared': 0.85
        }
        
        significance = self.framework._calculate_statistical_significance(pattern_data, self.historical_data)
        
        assert 0 <= significance <= 1
        assert significance > 0.5
        
    def test_historical_success_calculation(self):
        pattern_data = {
            'historical_success_rate': 0.68,
            'sample_size': 75,
            'recent_performance': 0.72,
            'avg_return': 0.042,
            'win_loss_ratio': 2.1
        }
        
        success = self.framework._calculate_historical_success(pattern_data)
        
        assert 0 <= success <= 1
        assert success > 0.6
        
    def test_volume_confirmation_calculation(self):
        pattern_data = {
            'pattern_type': 'breakout',
            'direction': 'bullish',
            'volume_trend': 'increasing'
        }
        
        market_data = {
            'volume': 2000000,
            'avg_volume': 1000000,
            'volume_profile': {
                'at_poc': True,
                'above_vwap': True
            }
        }
        
        volume_conf = self.framework._calculate_volume_confirmation(pattern_data, market_data)
        
        assert 0 <= volume_conf <= 1
        assert volume_conf > 0.8
        
    def test_timeframe_alignment_calculation(self):
        pattern_data = {
            'direction': 'bullish',
            'higher_timeframe_trend': 'bullish',
            'confluence_zones': ['support1', 'ma200', 'fib_618']
        }
        
        market_data = {
            'timeframe_signals': {
                '1m': {'direction': 'bullish'},
                '5m': {'direction': 'bullish'},
                '15m': {'direction': 'neutral'},
                '1h': {'direction': 'bullish'},
                '4h': {'direction': 'bullish'}
            }
        }
        
        alignment = self.framework._calculate_timeframe_alignment(pattern_data, market_data)
        
        assert 0 <= alignment <= 1
        assert alignment > 0.7
        
    def test_market_condition_adjustment(self):
        pattern_data = {
            'pattern_type': 'continuation',
            'direction': 'bullish',
            'market_phase': 'markup'
        }
        
        market_data = {
            'trend_strength': 0.8,
            'trend_direction': 'bullish',
            'volatility': 0.12
        }
        
        adjustment = self.framework._calculate_market_condition_adjustment(pattern_data, market_data)
        
        assert 0 <= adjustment <= 1
        assert adjustment > 0.6
        
    def test_market_condition_identification(self):
        market_data = {
            'trend_strength': 0.75,
            'volatility': 0.15
        }
        condition = self.framework._identify_market_condition(market_data)
        assert condition == MarketCondition.STRONG_TREND
        
        market_data = {
            'trend_strength': 0.2,
            'volatility': 0.35
        }
        condition = self.framework._identify_market_condition(market_data)
        assert condition == MarketCondition.VOLATILE
        
    def test_consistency_bonus(self):
        components = {
            ConfidenceComponent.STATISTICAL_SIGNIFICANCE: 0.75,
            ConfidenceComponent.HISTORICAL_SUCCESS: 0.78,
            ConfidenceComponent.VOLUME_CONFIRMATION: 0.73,
            ConfidenceComponent.TIMEFRAME_ALIGNMENT: 0.76,
            ConfidenceComponent.MARKET_CONDITION: 0.74
        }
        
        bonus = self.framework._calculate_consistency_bonus(components)
        assert bonus > 0
        
        components = {
            ConfidenceComponent.STATISTICAL_SIGNIFICANCE: 0.9,
            ConfidenceComponent.HISTORICAL_SUCCESS: 0.3,
            ConfidenceComponent.VOLUME_CONFIRMATION: 0.8,
            ConfidenceComponent.TIMEFRAME_ALIGNMENT: 0.4,
            ConfidenceComponent.MARKET_CONDITION: 0.7
        }
        
        bonus = self.framework._calculate_consistency_bonus(components)
        assert bonus <= 0
        
    def test_update_weights(self):
        performance_data = {
            'component_performance': {
                ConfidenceComponent.STATISTICAL_SIGNIFICANCE: 0.35,
                ConfidenceComponent.HISTORICAL_SUCCESS: 0.20,
                ConfidenceComponent.VOLUME_CONFIRMATION: 0.25,
                ConfidenceComponent.TIMEFRAME_ALIGNMENT: 0.15,
                ConfidenceComponent.MARKET_CONDITION: 0.05
            }
        }
        
        original_weights = self.framework.weights.copy()
        self.framework.update_weights(performance_data)
        
        assert pytest.approx(sum(self.framework.weights.values()), rel=1e-2) == 1.0
        
        assert self.framework.weights[ConfidenceComponent.STATISTICAL_SIGNIFICANCE] > \
               original_weights[ConfidenceComponent.STATISTICAL_SIGNIFICANCE]
        
    def test_confidence_explanation(self):
        pattern_data = {
            'pattern_type': 'breakout',
            'historical_success_rate': 0.8,
            'sample_size': 100
        }
        
        score = self.framework.calculate_confidence(pattern_data)
        explanation = self.framework.get_confidence_explanation(score)
        
        assert 'overall_confidence' in explanation
        assert 'confidence_level' in explanation
        assert 'recommendation' in explanation
        assert 'component_breakdown' in explanation
        assert 'key_factors' in explanation
        assert 'warnings' in explanation
        
        assert len(explanation['component_breakdown']) == len(score.components)
        
        for breakdown in explanation['component_breakdown']:
            assert 'component' in breakdown
            assert 'score' in breakdown
            assert 'weight' in breakdown
            assert 'contribution' in breakdown
            assert 'percentage' in breakdown


class TestAdaptiveConfidenceFramework:
    def setup_method(self):
        self.adaptive = AdaptiveConfidenceFramework(learning_rate=0.1)
        
    def test_initialization(self):
        assert self.adaptive.learning_rate == 0.1
        assert len(self.adaptive.performance_history) == 0
        assert len(self.adaptive.weight_history) == 0
        
    def test_learn_from_outcome(self):
        score = ConfidenceScore(
            overall_score=0.75,
            components={
                ConfidenceComponent.STATISTICAL_SIGNIFICANCE: 0.8,
                ConfidenceComponent.HISTORICAL_SUCCESS: 0.7,
                ConfidenceComponent.VOLUME_CONFIRMATION: 0.75,
                ConfidenceComponent.TIMEFRAME_ALIGNMENT: 0.73,
                ConfidenceComponent.MARKET_CONDITION: 0.77
            },
            weights=self.adaptive.weights.copy()
        )
        
        original_weights = self.adaptive.weights.copy()
        
        self.adaptive.learn_from_outcome(score, actual_outcome=True, actual_return=0.05)
        
        assert len(self.adaptive.performance_history) == 1
        assert len(self.adaptive.weight_history) == 1
        
        for component in ConfidenceComponent:
            if component in self.adaptive.weights:
                assert self.adaptive.weights[component] != original_weights.get(component, 0)
        
        assert pytest.approx(sum(self.adaptive.weights.values()), rel=1e-2) == 1.0
        
    def test_performance_evaluation(self):
        for i in range(110):
            score = ConfidenceScore(
                overall_score=0.6 + (i % 40) / 100,
                components={
                    ConfidenceComponent.STATISTICAL_SIGNIFICANCE: 0.7 + (i % 30) / 100,
                    ConfidenceComponent.HISTORICAL_SUCCESS: 0.6 + (i % 35) / 100,
                    ConfidenceComponent.VOLUME_CONFIRMATION: 0.65 + (i % 25) / 100,
                    ConfidenceComponent.TIMEFRAME_ALIGNMENT: 0.68 + (i % 28) / 100,
                    ConfidenceComponent.MARKET_CONDITION: 0.62 + (i % 32) / 100
                },
                weights=self.adaptive.weights.copy()
            )
            
            outcome = score.overall_score > 0.65
            returns = 0.03 if outcome else -0.01
            
            self.adaptive.learn_from_outcome(score, outcome, returns)
        
        assert len(self.adaptive.performance_history) == 110
        assert len(self.adaptive.weight_history) == 110
        
    def test_learning_metrics(self):
        for i in range(50):
            score = ConfidenceScore(
                overall_score=0.5 + i / 100,
                components={comp: 0.5 + i / 100 for comp in ConfidenceComponent},
                weights=self.adaptive.weights.copy()
            )
            
            outcome = i % 2 == 0
            returns = 0.02 if outcome else -0.01
            
            self.adaptive.learn_from_outcome(score, outcome, returns)
        
        metrics = self.adaptive.get_learning_metrics()
        
        assert 'total_predictions' in metrics
        assert 'accuracy' in metrics
        assert 'avg_confidence' in metrics
        assert 'avg_return' in metrics
        assert 'confidence_correlation' in metrics
        assert 'weight_evolution' in metrics
        
        assert metrics['total_predictions'] == 50
        assert 0 <= metrics['accuracy'] <= 1
        
    def test_weight_adaptation(self):
        initial_weights = self.adaptive.weights.copy()
        
        for i in range(200):
            components = {}
            for comp in ConfidenceComponent:
                if comp == ConfidenceComponent.STATISTICAL_SIGNIFICANCE:
                    components[comp] = 0.9
                elif comp == ConfidenceComponent.HISTORICAL_SUCCESS:
                    components[comp] = 0.3
                else:
                    components[comp] = 0.6
            
            score = ConfidenceScore(
                overall_score=0.65,
                components=components,
                weights=self.adaptive.weights.copy()
            )
            
            outcome = components[ConfidenceComponent.STATISTICAL_SIGNIFICANCE] > 0.7
            returns = 0.03 if outcome else -0.02
            
            self.adaptive.learn_from_outcome(score, outcome, returns)
        
        final_weights = self.adaptive.weights
        
        assert final_weights[ConfidenceComponent.STATISTICAL_SIGNIFICANCE] > \
               initial_weights[ConfidenceComponent.STATISTICAL_SIGNIFICANCE]
        
        assert final_weights[ConfidenceComponent.HISTORICAL_SUCCESS] < \
               initial_weights[ConfidenceComponent.HISTORICAL_SUCCESS]


class TestIntegration:
    def test_full_confidence_workflow(self):
        framework = ConfidenceFramework()
        
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
        historical_data = pd.DataFrame({
            'timestamp': dates,
            'close': 100 + np.cumsum(np.random.randn(200) * 0.5),
            'volume': np.random.uniform(900000, 1100000, 200)
        })
        
        pattern_data = {
            'pattern_type': 'head_and_shoulders',
            'pattern_name': 'Head and Shoulders',
            'direction': 'bearish',
            'price_points': [100, 105, 103, 108, 104, 106, 102],
            'pattern_points': [
                {'time': '2024-01-01', 'price': 100, 'label': 'left_shoulder'},
                {'time': '2024-01-02', 'price': 108, 'label': 'head'},
                {'time': '2024-01-03', 'price': 102, 'label': 'right_shoulder'}
            ],
            'historical_success_rate': 0.72,
            'sample_size': 85,
            'avg_return': -0.038,
            'win_loss_ratio': 1.9,
            'r_squared': 0.78,
            'higher_timeframe_trend': 'bearish',
            'confluence_zones': ['resistance1', 'ma50'],
            'volume_trend': 'increasing'
        }
        
        market_data = {
            'volume': 1800000,
            'avg_volume': 1000000,
            'trend_strength': 0.65,
            'trend_direction': 'bearish',
            'volatility': 0.18,
            'timeframe_signals': {
                '15m': {'direction': 'bearish'},
                '1h': {'direction': 'bearish'},
                '4h': {'direction': 'bearish'},
                '1d': {'direction': 'neutral'}
            },
            'volume_profile': {
                'at_poc': False,
                'above_vwap': False
            }
        }
        
        confidence = framework.calculate_confidence(pattern_data, market_data, historical_data)
        
        assert confidence.overall_score > 0.6
        assert confidence.confidence_level in ["HIGH", "MEDIUM", "VERY_HIGH"]
        assert confidence.recommendation in ["STRONG_SIGNAL", "MODERATE_SIGNAL"]
        
        explanation = framework.get_confidence_explanation(confidence)
        assert len(explanation['component_breakdown']) == 5
        assert len(explanation['key_factors']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])