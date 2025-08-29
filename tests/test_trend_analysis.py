"""
Tests for trend analysis algorithms.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.core.analysis.trend_analysis import (
    HodrickPrescottFilter,
    MannKendallTest,
    TrendReversalDetector,
    SupportResistanceIdentifier,
    TrendlineDetector,
    TrendStrengthAnalyzer,
    TrendResult,
    SupportResistanceLevel
)


@pytest.fixture
def trending_data():
    """Generate trending price data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Uptrend with noise
    uptrend = 100 + np.arange(100) * 0.5 + np.random.randn(100) * 2
    
    # Downtrend with noise
    downtrend = 150 - np.arange(100) * 0.5 + np.random.randn(100) * 2
    
    # Sideways with noise
    sideways = 100 + np.sin(np.arange(100) * 0.1) * 5 + np.random.randn(100) * 2
    
    return {
        'dates': dates,
        'uptrend': pd.Series(uptrend, index=dates),
        'downtrend': pd.Series(downtrend, index=dates),
        'sideways': pd.Series(sideways, index=dates)
    }


@pytest.fixture
def ohlc_data():
    """Generate OHLC price data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Generate realistic OHLC data
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    
    high = close + np.abs(np.random.randn(100) * 1)
    low = close - np.abs(np.random.randn(100) * 1)
    open_price = close + np.random.randn(100) * 0.5
    
    return {
        'dates': dates,
        'open': pd.Series(open_price, index=dates),
        'high': pd.Series(high, index=dates),
        'low': pd.Series(low, index=dates),
        'close': pd.Series(close, index=dates)
    }


class TestHodrickPrescottFilter:
    """Test Hodrick-Prescott filter."""
    
    def test_hp_filter_decomposition(self, trending_data):
        """Test HP filter decomposition."""
        hp_filter = HodrickPrescottFilter(lambda_param=1600)
        
        result = hp_filter.decompose(trending_data['uptrend'])
        
        assert 'trend' in result
        assert 'cycle' in result
        assert 'noise' in result
        assert 'trend_strength' in result
        
        # Trend should be smoother than original
        trend_volatility = result['trend'].std()
        original_volatility = result['original'].std()
        assert trend_volatility < original_volatility
        
        # Components should sum to original (approximately)
        reconstructed = result['trend'] + result['cycle']
        diff = abs(reconstructed - result['original']).mean()
        assert diff < 0.01  # Small reconstruction error
    
    def test_hp_filter_trend_extraction(self, trending_data):
        """Test trend extraction quality."""
        hp_filter = HodrickPrescottFilter()
        
        # Test on clear uptrend
        uptrend_result = hp_filter.decompose(trending_data['uptrend'])
        uptrend_slope = np.polyfit(range(len(uptrend_result['trend'])), 
                                  uptrend_result['trend'].values, 1)[0]
        assert uptrend_slope > 0  # Should detect upward trend
        
        # Test on clear downtrend
        downtrend_result = hp_filter.decompose(trending_data['downtrend'])
        downtrend_slope = np.polyfit(range(len(downtrend_result['trend'])), 
                                    downtrend_result['trend'].values, 1)[0]
        assert downtrend_slope < 0  # Should detect downward trend
    
    def test_hp_filter_parameter_sensitivity(self, trending_data):
        """Test sensitivity to lambda parameter."""
        series = trending_data['uptrend']
        
        # Low lambda (less smoothing)
        hp_low = HodrickPrescottFilter(lambda_param=100)
        result_low = hp_low.decompose(series)
        
        # High lambda (more smoothing)
        hp_high = HodrickPrescottFilter(lambda_param=10000)
        result_high = hp_high.decompose(series)
        
        # Higher lambda should produce smoother trend
        assert result_high['trend'].std() < result_low['trend'].std()


class TestMannKendallTest:
    """Test Mann-Kendall trend test."""
    
    def test_mk_test_uptrend(self, trending_data):
        """Test Mann-Kendall on uptrend."""
        mk_test = MannKendallTest(alpha=0.05)
        result = mk_test.test(trending_data['uptrend'])
        
        assert result['trend'] == 'increasing'
        assert result['significant'] == True
        assert result['p_value'] < 0.05
        assert result['z_score'] > 0
        assert result['sen_slope'] > 0
    
    def test_mk_test_downtrend(self, trending_data):
        """Test Mann-Kendall on downtrend."""
        mk_test = MannKendallTest(alpha=0.05)
        result = mk_test.test(trending_data['downtrend'])
        
        assert result['trend'] == 'decreasing'
        assert result['significant'] == True
        assert result['p_value'] < 0.05
        assert result['z_score'] < 0
        assert result['sen_slope'] < 0
    
    def test_mk_test_no_trend(self, trending_data):
        """Test Mann-Kendall on sideways market."""
        mk_test = MannKendallTest(alpha=0.05)
        result = mk_test.test(trending_data['sideways'])
        
        # Sideways market should have no significant trend
        assert result['trend'] == 'no_trend'
        assert result['significant'] == False
        assert result['p_value'] > 0.05
    
    def test_mk_test_ties_handling(self):
        """Test Mann-Kendall with tied values."""
        # Create data with ties
        data = pd.Series([1, 2, 2, 3, 3, 3, 4, 5, 5])
        
        mk_test = MannKendallTest()
        result = mk_test.test(data)
        
        # Should still detect increasing trend despite ties
        assert result['trend'] == 'increasing'
        assert result['sen_slope'] > 0


class TestTrendReversalDetector:
    """Test trend reversal detection."""
    
    def test_reversal_detection(self):
        """Test basic reversal detection."""
        # Create data with clear reversal
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        
        # Uptrend then downtrend
        prices = []
        for i in range(30):
            prices.append(100 + i)
        for i in range(30):
            prices.append(130 - i)
        
        prices_series = pd.Series(prices, index=dates)
        
        detector = TrendReversalDetector(lookback=10, confirmation_bars=3)
        reversals = detector.detect_reversals(prices_series)
        
        assert len(reversals) > 0
        
        # Should detect top reversal around index 30
        top_reversals = [r for r in reversals if r['type'] == 'top_reversal']
        assert len(top_reversals) > 0
        
        # Check reversal is around the peak
        for reversal in top_reversals:
            assert 25 <= reversal['index'] <= 35
    
    def test_double_top_pattern(self):
        """Test double top pattern detection."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        # Create double top pattern
        prices = []
        # First peak
        for i in range(15):
            prices.append(100 + i)
        for i in range(10):
            prices.append(115 - i)
        # Second peak
        for i in range(15):
            prices.append(105 + i)
        for i in range(10):
            prices.append(120 - i * 2)
        
        prices_series = pd.Series(prices, index=dates)
        
        detector = TrendReversalDetector()
        reversals = detector.detect_reversals(prices_series)
        
        # Should detect double top pattern
        double_tops = [r for r in reversals if r.get('pattern') == 'double_top']
        assert len(double_tops) > 0
    
    def test_reversal_confirmation(self, ohlc_data):
        """Test reversal confirmation requirements."""
        detector = TrendReversalDetector(confirmation_bars=5)
        reversals = detector.detect_reversals(ohlc_data['close'])
        
        # All detected reversals should be confirmed
        for reversal in reversals:
            assert reversal['confirmed'] == True
            assert reversal['confidence'] > 0


class TestSupportResistanceIdentifier:
    """Test support and resistance identification."""
    
    def test_level_identification(self, ohlc_data):
        """Test basic level identification."""
        identifier = SupportResistanceIdentifier(lookback=50, min_touches=2)
        
        levels = identifier.identify_levels(
            ohlc_data['high'],
            ohlc_data['low'],
            ohlc_data['close']
        )
        
        assert len(levels) > 0
        
        # Check level properties
        for level in levels:
            assert isinstance(level, SupportResistanceLevel)
            assert level.level > 0
            assert level.level_type in ['support', 'resistance']
            assert level.strength >= 2  # Min touches
            assert 0 <= level.confidence <= 1
    
    def test_support_resistance_classification(self, ohlc_data):
        """Test classification of support vs resistance."""
        identifier = SupportResistanceIdentifier()
        
        levels = identifier.identify_levels(
            ohlc_data['high'],
            ohlc_data['low'],
            ohlc_data['close']
        )
        
        current_price = ohlc_data['close'].iloc[-1]
        
        # Levels below current price should be support
        support_levels = [l for l in levels if l.level_type == 'support']
        for level in support_levels:
            assert level.level < current_price * 1.02  # Allow small tolerance
        
        # Levels above current price should be resistance
        resistance_levels = [l for l in levels if l.level_type == 'resistance']
        for level in resistance_levels:
            assert level.level > current_price * 0.98  # Allow small tolerance
    
    def test_fibonacci_levels(self, ohlc_data):
        """Test Fibonacci level calculation."""
        identifier = SupportResistanceIdentifier()
        
        # Test internal Fibonacci calculation
        fib_levels = identifier._find_fibonacci_levels(
            ohlc_data['high'],
            ohlc_data['low']
        )
        
        assert len(fib_levels) == 5  # Should have 5 Fibonacci levels
        
        # Check Fibonacci ratios
        high = ohlc_data['high'].max()
        low = ohlc_data['low'].min()
        price_range = high - low
        
        # Verify some key Fibonacci levels
        level_values = [l.level for l in fib_levels]
        
        # 50% retracement should be present
        fifty_percent = high - (price_range * 0.5)
        assert any(abs(l - fifty_percent) < 1 for l in level_values)
    
    def test_psychological_levels(self):
        """Test psychological (round number) levels."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Create data around round numbers
        prices = []
        for i in range(100):
            base = 100 if i < 50 else 200
            prices.append(base + np.random.randn() * 2)
        
        close = pd.Series(prices, index=dates)
        high = close + 1
        low = close - 1
        
        identifier = SupportResistanceIdentifier(min_touches=5)
        levels = identifier.identify_levels(high, low, close)
        
        # Should identify 100 and 200 as psychological levels
        level_values = [l.level for l in levels]
        
        # Check for round numbers (within tolerance)
        has_100_level = any(95 < l < 105 for l in level_values)
        has_200_level = any(195 < l < 205 for l in level_values)
        
        assert has_100_level or has_200_level


class TestTrendlineDetector:
    """Test trendline detection."""
    
    def test_trendline_detection(self, trending_data):
        """Test basic trendline detection."""
        detector = TrendlineDetector(min_points=3)
        
        trendlines = detector.detect_trendlines(trending_data['uptrend'])
        
        assert len(trendlines) > 0
        
        # Check trendline properties
        for line in trendlines:
            assert 'slope' in line
            assert 'intercept' in line
            assert 'r_squared' in line
            assert 'touches' in line
            assert line['touches'] >= 3  # Min points requirement
    
    def test_uptrend_line(self, trending_data):
        """Test uptrend line detection."""
        detector = TrendlineDetector()
        
        trendlines = detector.detect_trendlines(trending_data['uptrend'])
        
        # Should detect upward sloping lines
        upward_lines = [l for l in trendlines if l['slope'] > 0]
        assert len(upward_lines) > 0
        
        # Check line quality
        for line in upward_lines:
            assert line['r_squared'] > 0.5  # Reasonable fit
    
    def test_channel_detection(self):
        """Test channel (parallel lines) detection."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        # Create channel pattern
        prices = []
        for i in range(50):
            base = 100 + i * 0.5  # Upward slope
            oscillation = 5 * np.sin(i * 0.5)  # Channel width
            prices.append(base + oscillation)
        
        prices_series = pd.Series(prices, index=dates)
        
        detector = TrendlineDetector()
        trendlines = detector.detect_trendlines(prices_series)
        
        # Should detect channel lines
        channels = [l for l in trendlines if l['type'] == 'channel']
        
        # Channels should have upper and lower lines
        for channel in channels:
            assert 'upper_slope' in channel
            assert 'lower_slope' in channel
            assert 'width' in channel


class TestTrendStrengthAnalyzer:
    """Test trend strength measurement."""
    
    def test_strength_measurement(self, trending_data):
        """Test comprehensive strength measurement."""
        analyzer = TrendStrengthAnalyzer()
        
        # Test on strong uptrend
        uptrend_strength = analyzer.measure_strength(trending_data['uptrend'])
        
        assert 'overall_strength' in uptrend_strength
        assert 'trend_quality' in uptrend_strength
        assert 'adx' in uptrend_strength
        assert 'linear_regression' in uptrend_strength
        assert 'mann_kendall' in uptrend_strength
        
        # Strong uptrend should have high strength
        assert uptrend_strength['overall_strength'] > 40
        assert uptrend_strength['mann_kendall']['trend'] == 'increasing'
    
    def test_adx_calculation(self, trending_data):
        """Test ADX calculation and interpretation."""
        analyzer = TrendStrengthAnalyzer()
        
        # Test on different trend types
        uptrend_strength = analyzer.measure_strength(trending_data['uptrend'])
        sideways_strength = analyzer.measure_strength(trending_data['sideways'])
        
        # Uptrend should have higher ADX than sideways
        assert uptrend_strength['adx'] > sideways_strength['adx']
        
        # Check interpretation
        assert uptrend_strength['adx_interpretation'] in [
            'weak_trend', 'strong_trend', 'very_strong_trend', 'extremely_strong_trend'
        ]
    
    def test_ma_alignment(self, trending_data):
        """Test moving average alignment."""
        analyzer = TrendStrengthAnalyzer()
        
        strength = analyzer.measure_strength(trending_data['uptrend'], period=20)
        
        assert 'ma_alignment' in strength
        assert 'aligned' in strength['ma_alignment']
        assert 'strength' in strength['ma_alignment']
        assert 'mas' in strength['ma_alignment']
    
    def test_trend_quality_assessment(self):
        """Test trend quality categories."""
        analyzer = TrendStrengthAnalyzer()
        
        # Create data with different trend strengths
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Very strong trend
        strong_trend = pd.Series(np.arange(100), index=dates)
        strong_result = analyzer.measure_strength(strong_trend)
        assert strong_result['trend_quality'] in ['strong_trend', 'very_strong_trend']
        
        # No trend (random walk)
        no_trend = pd.Series(np.random.randn(100).cumsum(), index=dates)
        no_trend_result = analyzer.measure_strength(no_trend)
        assert no_trend_result['trend_quality'] in ['no_trend', 'weak_trend', 'moderate_trend']


class TestIntegration:
    """Integration tests for trend analysis."""
    
    def test_complete_trend_analysis(self, ohlc_data):
        """Test complete trend analysis workflow."""
        # HP Filter
        hp_filter = HodrickPrescottFilter()
        hp_result = hp_filter.decompose(ohlc_data['close'])
        
        # Mann-Kendall test
        mk_test = MannKendallTest()
        mk_result = mk_test.test(hp_result['trend'])
        
        # Reversal detection
        reversal_detector = TrendReversalDetector()
        reversals = reversal_detector.detect_reversals(ohlc_data['close'])
        
        # Support/Resistance
        sr_identifier = SupportResistanceIdentifier()
        levels = sr_identifier.identify_levels(
            ohlc_data['high'],
            ohlc_data['low'],
            ohlc_data['close']
        )
        
        # Trendlines
        trendline_detector = TrendlineDetector()
        trendlines = trendline_detector.detect_trendlines(ohlc_data['close'])
        
        # Trend strength
        strength_analyzer = TrendStrengthAnalyzer()
        strength = strength_analyzer.measure_strength(ohlc_data['close'])
        
        # All components should work together
        assert hp_result is not None
        assert mk_result is not None
        assert isinstance(reversals, list)
        assert isinstance(levels, list)
        assert isinstance(trendlines, list)
        assert isinstance(strength, dict)
        
        # Results should be consistent
        if mk_result['trend'] == 'increasing':
            assert hp_result['trend'].iloc[-1] > hp_result['trend'].iloc[0]
        elif mk_result['trend'] == 'decreasing':
            assert hp_result['trend'].iloc[-1] < hp_result['trend'].iloc[0]