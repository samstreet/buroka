import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.validation.pattern_validator import (
    PatternValidator,
    ValidationResult,
    DegradationAnalysis,
    ABTestResult,
    ValidationMethod,
    DegradationStatus
)
from src.core.confidence.confidence_framework import ConfidenceFramework


class MockPatternDetector:
    def __init__(self, success_rate: float = 0.7):
        self.success_rate = success_rate
        
    def __call__(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        patterns = []
        num_patterns = max(1, len(data) // 50)
        
        for i in range(num_patterns):
            patterns.append({
                'type': 'candlestick',
                'name': f'pattern_{i}',
                'direction': 'bullish' if i % 2 == 0 else 'bearish',
                'entry_index': i * 50 + 10,
                'exit_index': min(i * 50 + 30, len(data) - 1),
                'confidence': 0.5 + (i % 5) * 0.1,
                'success_rate': self.success_rate
            })
        
        return patterns


class TestValidationResult:
    def test_validation_result_properties(self):
        result = ValidationResult(
            method=ValidationMethod.OUT_OF_SAMPLE,
            total_patterns=100,
            true_positives=60,
            false_positives=10,
            true_negatives=20,
            false_negatives=10,
            accuracy=0.8,
            precision=0.857,
            recall=0.857,
            f1_score=0.857,
            sharpe_ratio=1.2,
            max_drawdown=-0.15,
            profit_factor=2.5,
            win_rate=0.65,
            avg_return=0.025,
            validation_period=(datetime(2024, 1, 1), datetime(2024, 2, 1))
        )
        
        cm = result.confusion_matrix
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 20  # True negatives
        assert cm[1, 1] == 60  # True positives
        
        mcc = result.matthews_correlation
        assert -1 <= mcc <= 1
        assert mcc > 0  # Should be positive for good classifier


class TestPatternValidator:
    def setup_method(self):
        self.validator = PatternValidator()
        self.dates = pd.date_range(start='2024-01-01', periods=500, freq='1h')
        self.data = pd.DataFrame({
            'timestamp': self.dates,
            'open': 100 + np.cumsum(np.random.randn(500) * 0.5),
            'high': 102 + np.cumsum(np.random.randn(500) * 0.5),
            'low': 98 + np.cumsum(np.random.randn(500) * 0.5),
            'close': 100 + np.cumsum(np.random.randn(500) * 0.5),
            'volume': np.random.uniform(900000, 1100000, 500)
        })
        self.data.set_index('timestamp', inplace=True)
        
    def test_validate_out_of_sample(self):
        train_data = self.data.iloc[:300]
        test_data = self.data.iloc[300:]
        detector = MockPatternDetector(success_rate=0.7)
        
        result = self.validator.validate_out_of_sample(
            train_data, test_data, detector, threshold=0.6
        )
        
        assert isinstance(result, ValidationResult)
        assert result.method == ValidationMethod.OUT_OF_SAMPLE
        assert result.total_patterns > 0
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.precision <= 1
        assert 0 <= result.recall <= 1
        assert 0 <= result.f1_score <= 1
        assert result.validation_period[0] == test_data.index[0]
        assert result.validation_period[1] == test_data.index[-1]
        
    def test_rolling_window_backtest(self):
        detector = MockPatternDetector(success_rate=0.65)
        
        results = self.validator.rolling_window_backtest(
            self.data,
            detector,
            window_size=100,
            step_size=50,
            validation_size=20
        )
        
        assert len(results) > 0
        assert all(isinstance(r, ValidationResult) for r in results)
        assert all(r.method == ValidationMethod.OUT_OF_SAMPLE for r in results)
        
        # Check that windows are progressing
        for i in range(len(results) - 1):
            assert results[i].validation_period[0] < results[i+1].validation_period[0]
            
    def test_walk_forward_analysis(self):
        detector = MockPatternDetector(success_rate=0.72)
        
        results = self.validator.walk_forward_analysis(
            self.data,
            detector,
            initial_train_size=200,
            test_size=50,
            retrain_frequency=25
        )
        
        assert len(results) > 0
        assert all(isinstance(r, ValidationResult) for r in results)
        
        # Check that test periods don't overlap
        for i in range(len(results) - 1):
            assert results[i].validation_period[1] <= results[i+1].validation_period[0]
            
    def test_cross_validate(self):
        detector = MockPatternDetector(success_rate=0.68)
        
        results = self.validator.cross_validate(
            self.data,
            detector,
            n_splits=3
        )
        
        assert len(results) == 3
        assert all(isinstance(r, ValidationResult) for r in results)
        
        # Check that each split has increasing data
        for i in range(len(results) - 1):
            assert results[i].metadata['train_size'] < results[i+1].metadata['train_size']
            
    def test_detect_pattern_degradation(self):
        # Create historical performance data with degradation
        historical = []
        for i in range(30):
            success_rate = 0.75 if i < 20 else 0.55  # Degradation in last 10
            historical.append({
                'pattern_type': 'candlestick',
                'pattern_name': 'hammer',
                'success': np.random.random() < success_rate,
                'return': np.random.normal(0.02 if i < 20 else -0.01, 0.05)
            })
        
        analysis = self.validator.detect_pattern_degradation(
            historical,
            lookback_periods=10,
            significance_level=0.05
        )
        
        assert isinstance(analysis, DegradationAnalysis)
        assert analysis.status in DegradationStatus
        assert 0 <= analysis.current_performance <= 1
        assert 0 <= analysis.historical_avg <= 1
        assert analysis.recommendation != ""
        
    def test_degradation_insufficient_data(self):
        historical = [
            {'success': True, 'return': 0.02},
            {'success': False, 'return': -0.01}
        ]
        
        analysis = self.validator.detect_pattern_degradation(
            historical,
            lookback_periods=10
        )
        
        assert analysis.status == DegradationStatus.INSUFFICIENT_DATA
        assert "Insufficient data" in analysis.recommendation
        
    def test_ab_test_patterns(self):
        # Create results for two pattern variants
        variant_a = [
            {'return': np.random.normal(0.02, 0.01), 'success': True}
            for _ in range(50)
        ]
        
        variant_b = [
            {'return': np.random.normal(0.03, 0.01), 'success': True}
            for _ in range(50)
        ]
        
        result = self.validator.ab_test_patterns(
            variant_a,
            variant_b,
            metric='return',
            confidence_level=0.95
        )
        
        assert isinstance(result, ABTestResult)
        assert result.sample_size_a == 50
        assert result.sample_size_b == 50
        assert 0 <= result.p_value <= 1
        assert result.confidence_level == 0.95
        assert result.recommendation != ""
        
        # B should show improvement
        assert result.improvement > 0
        
    def test_ab_test_insufficient_data(self):
        result = self.validator.ab_test_patterns(
            [],
            [],
            metric='return'
        )
        
        assert result.is_significant == False
        assert "Insufficient data" in result.recommendation
        
    def test_analyze_false_positives_negatives(self):
        predictions = [
            {'signal': True, 'pattern_name': 'hammer', 'confidence': 0.8, 'timestamp': datetime.now()},
            {'signal': True, 'pattern_name': 'doji', 'confidence': 0.7, 'timestamp': datetime.now()},
            {'signal': False, 'pattern_name': 'engulfing', 'confidence': 0.4, 'timestamp': datetime.now()},
            {'signal': False, 'pattern_name': 'hammer', 'confidence': 0.3, 'timestamp': datetime.now()},
            {'signal': True, 'pattern_name': 'hammer', 'confidence': 0.9, 'timestamp': datetime.now()}
        ]
        
        actuals = [
            {'success': False},  # False positive
            {'success': True},   # True positive
            {'success': False},  # True negative
            {'success': True},   # False negative
            {'success': True}    # True positive
        ]
        
        analysis = self.validator.analyze_false_positives_negatives(predictions, actuals)
        
        assert 'total_false_positives' in analysis
        assert 'total_false_negatives' in analysis
        assert 'false_positive_rate' in analysis
        assert 'false_negative_rate' in analysis
        assert 'most_common_fp_patterns' in analysis
        assert 'most_common_fn_patterns' in analysis
        assert 'avg_fp_confidence' in analysis
        assert 'avg_fn_confidence' in analysis
        assert 'recommendations' in analysis
        
        assert analysis['total_false_positives'] == 1
        assert analysis['total_false_negatives'] == 1
        
    def test_generate_performance_report(self):
        # Create multiple validation results
        results = []
        for i in range(5):
            results.append(ValidationResult(
                method=ValidationMethod.OUT_OF_SAMPLE,
                total_patterns=100,
                true_positives=60 + i,
                false_positives=10 - i//2,
                true_negatives=20 + i//2,
                false_negatives=10 - i//2,
                accuracy=0.75 + i * 0.02,
                precision=0.8 + i * 0.01,
                recall=0.8 + i * 0.01,
                f1_score=0.8 + i * 0.01,
                sharpe_ratio=1.0 + i * 0.1,
                max_drawdown=-0.2 + i * 0.02,
                profit_factor=2.0 + i * 0.2,
                win_rate=0.6 + i * 0.02,
                avg_return=0.02 + i * 0.002,
                validation_period=(datetime(2024, 1, 1), datetime(2024, 1, 31))
            ))
        
        report = self.validator.generate_performance_report(results, pattern_type='candlestick')
        
        assert 'total_validations' in report
        assert report['total_validations'] == 5
        assert report['pattern_type'] == 'candlestick'
        assert 'avg_accuracy' in report
        assert 'avg_precision' in report
        assert 'avg_recall' in report
        assert 'avg_f1_score' in report
        assert 'avg_sharpe_ratio' in report
        assert 'best_validation' in report
        assert 'worst_validation' in report
        assert 'stability_score' in report
        assert 'recommendations' in report
        
        assert 0 <= report['stability_score'] <= 1
        assert isinstance(report['recommendations'], list)
        
    def test_empty_performance_report(self):
        report = self.validator.generate_performance_report([])
        assert 'error' in report
        
    def test_calculate_sharpe_ratio(self):
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02])
        sharpe = self.validator._calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        
        # Test with zero returns
        zero_returns = np.array([])
        sharpe_zero = self.validator._calculate_sharpe_ratio(zero_returns)
        assert sharpe_zero == 0
        
    def test_calculate_max_drawdown(self):
        returns = np.array([0.05, -0.02, -0.03, 0.04, -0.01, 0.02])
        drawdown = self.validator._calculate_max_drawdown(returns)
        assert drawdown <= 0
        
        # Test with empty returns
        empty_returns = np.array([])
        dd_empty = self.validator._calculate_max_drawdown(empty_returns)
        assert dd_empty == 0
        
    def test_calculate_profit_factor(self):
        returns = np.array([0.05, -0.02, 0.03, -0.01, 0.04])
        pf = self.validator._calculate_profit_factor(returns)
        assert pf > 0
        
        # Test with all positive returns
        positive_returns = np.array([0.01, 0.02, 0.03])
        pf_positive = self.validator._calculate_profit_factor(positive_returns)
        assert pf_positive == float('inf')
        
        # Test with all negative returns
        negative_returns = np.array([-0.01, -0.02, -0.03])
        pf_negative = self.validator._calculate_profit_factor(negative_returns)
        assert pf_negative == 0


class TestIntegration:
    def test_full_validation_workflow(self):
        # Create validator with custom confidence framework
        confidence_framework = ConfidenceFramework()
        validator = PatternValidator(confidence_framework)
        
        # Generate synthetic data
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1h')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.cumsum(np.random.randn(1000) * 0.5),
            'high': 102 + np.cumsum(np.random.randn(1000) * 0.5),
            'low': 98 + np.cumsum(np.random.randn(1000) * 0.5),
            'close': 100 + np.cumsum(np.random.randn(1000) * 0.5),
            'volume': np.random.uniform(900000, 1100000, 1000)
        })
        data.set_index('timestamp', inplace=True)
        
        # Create pattern detector
        detector = MockPatternDetector(success_rate=0.7)
        
        # Run rolling window backtest
        backtest_results = validator.rolling_window_backtest(
            data[:800],
            detector,
            window_size=200,
            step_size=100,
            validation_size=50
        )
        
        assert len(backtest_results) > 0
        
        # Generate performance report
        report = validator.generate_performance_report(backtest_results, pattern_type='test')
        
        assert report['avg_accuracy'] > 0
        assert len(report['recommendations']) > 0
        
        # Test pattern degradation
        historical_performance = []
        for result in backtest_results:
            for i in range(10):
                historical_performance.append({
                    'pattern_type': 'test',
                    'pattern_name': 'test_pattern',
                    'success': np.random.random() < result.accuracy,
                    'return': np.random.normal(result.avg_return, 0.01)
                })
        
        degradation = validator.detect_pattern_degradation(historical_performance)
        assert degradation.status in DegradationStatus
        
        # Run A/B test
        variant_a = historical_performance[:len(historical_performance)//2]
        variant_b = historical_performance[len(historical_performance)//2:]
        
        ab_result = validator.ab_test_patterns(variant_a, variant_b)
        assert ab_result.recommendation != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])