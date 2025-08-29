"""
Tests for volume-based technical indicators.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.core.indicators.volume import (
    OnBalanceVolume,
    VolumeRateOfChange,
    VolumeWeightedAveragePrice,
    AccumulationDistributionLine,
    VolumeSpikeDetector,
    VolumeAnalyzer,
    VolumeIndicatorConfig
)


@pytest.fixture
def sample_price_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    high = close + np.abs(np.random.randn(100) * 1)
    low = close - np.abs(np.random.randn(100) * 1)
    volume = np.random.randint(1000000, 5000000, 100).astype(float)
    
    # Add some volume spikes
    spike_indices = [20, 45, 70, 85]
    for idx in spike_indices:
        volume[idx] *= 3  # Triple the volume for spikes
    
    return {
        'high': pd.Series(high, index=dates),
        'low': pd.Series(low, index=dates),
        'close': pd.Series(close, index=dates),
        'volume': pd.Series(volume, index=dates)
    }


@pytest.fixture
def trend_data():
    """Generate trending price data."""
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    
    # Uptrend
    close_up = 100 + np.arange(50) * 0.5 + np.random.randn(50) * 0.5
    volume_up = np.linspace(1000000, 2000000, 50) + np.random.randn(50) * 100000
    
    # Downtrend
    close_down = 125 - np.arange(50) * 0.5 + np.random.randn(50) * 0.5
    volume_down = np.linspace(2000000, 1000000, 50) + np.random.randn(50) * 100000
    
    return {
        'uptrend': {
            'close': pd.Series(close_up, index=dates),
            'volume': pd.Series(np.abs(volume_up), index=dates)
        },
        'downtrend': {
            'close': pd.Series(close_down, index=dates),
            'volume': pd.Series(np.abs(volume_down), index=dates)
        }
    }


class TestOnBalanceVolume:
    """Test On-Balance Volume indicator."""
    
    def test_obv_calculation(self, sample_price_data):
        """Test basic OBV calculation."""
        obv = OnBalanceVolume()
        result = obv.calculate(
            sample_price_data['close'],
            sample_price_data['volume']
        )
        
        assert result is not None
        assert len(result.values) == len(sample_price_data['close'])
        assert result.indicator_name == "OBV"
        
        # Check that OBV changes with price
        obv_values = result.values.values
        price_changes = sample_price_data['close'].diff()
        
        for i in range(1, len(obv_values)):
            if price_changes.iloc[i] > 0:
                # OBV should increase
                assert obv_values[i] > obv_values[i-1]
            elif price_changes.iloc[i] < 0:
                # OBV should decrease
                assert obv_values[i] < obv_values[i-1]
    
    def test_obv_divergence_detection(self, trend_data):
        """Test OBV divergence detection."""
        obv = OnBalanceVolume()
        
        # Create divergence scenario: price up, volume down
        prices = trend_data['uptrend']['close']
        volumes = trend_data['downtrend']['volume']
        
        result = obv.calculate(prices, volumes)
        
        assert 'divergence' in result.metadata
        assert 'trend' in result.metadata
    
    def test_obv_signal_generation(self, sample_price_data):
        """Test OBV signal generation."""
        obv = OnBalanceVolume()
        result = obv.calculate(
            sample_price_data['close'],
            sample_price_data['volume']
        )
        
        assert result.signal is not None
        assert len(result.signal) == len(sample_price_data['close'])
        
        # Check signal values are valid
        unique_signals = result.signal.unique()
        assert all(s in [-1, 0, 1] for s in unique_signals)


class TestVolumeRateOfChange:
    """Test Volume Rate of Change indicator."""
    
    def test_vroc_calculation(self, sample_price_data):
        """Test basic VROC calculation."""
        vroc = VolumeRateOfChange()
        result = vroc.calculate(sample_price_data['volume'])
        
        assert result is not None
        assert len(result.values) == len(sample_price_data['volume'])
        assert result.indicator_name == "VROC"
        
        # Check VROC formula
        period = 14
        for i in range(period, len(result.values)):
            if not pd.isna(result.values.iloc[i]):
                expected = ((sample_price_data['volume'].iloc[i] - 
                           sample_price_data['volume'].iloc[i-period]) / 
                          sample_price_data['volume'].iloc[i-period]) * 100
                assert abs(result.values.iloc[i] - expected) < 0.01
    
    def test_vroc_custom_period(self, sample_price_data):
        """Test VROC with custom period."""
        vroc = VolumeRateOfChange()
        result = vroc.calculate(sample_price_data['volume'], period=20)
        
        assert result is not None
        assert result.metadata['period'] == 20
    
    def test_vroc_extremes_detection(self, sample_price_data):
        """Test VROC extreme levels detection."""
        vroc = VolumeRateOfChange()
        result = vroc.calculate(sample_price_data['volume'])
        
        assert 'extreme_levels' in result.metadata
        assert 'vroc_trend' in result.metadata


class TestVolumeWeightedAveragePrice:
    """Test VWAP indicator."""
    
    def test_vwap_calculation(self, sample_price_data):
        """Test basic VWAP calculation."""
        vwap = VolumeWeightedAveragePrice()
        result = vwap.calculate(
            sample_price_data['high'],
            sample_price_data['low'],
            sample_price_data['close'],
            sample_price_data['volume']
        )
        
        assert result is not None
        assert len(result.values) == len(sample_price_data['close'])
        assert result.indicator_name == "VWAP"
        
        # VWAP should be between high and low
        for i in range(len(result.values)):
            if not pd.isna(result.values.iloc[i]):
                assert result.values.iloc[i] >= sample_price_data['low'].iloc[i]
                assert result.values.iloc[i] <= sample_price_data['high'].iloc[i]
    
    def test_vwap_bands(self, sample_price_data):
        """Test VWAP bands calculation."""
        vwap = VolumeWeightedAveragePrice()
        result = vwap.calculate(
            sample_price_data['high'],
            sample_price_data['low'],
            sample_price_data['close'],
            sample_price_data['volume']
        )
        
        assert 'upper_band_1' in result.metadata
        assert 'lower_band_1' in result.metadata
        assert 'upper_band_2' in result.metadata
        assert 'lower_band_2' in result.metadata
        
        # Bands should be symmetric around VWAP
        vwap_val = result.values.iloc[-1]
        upper_1 = result.metadata['upper_band_1'].iloc[-1]
        lower_1 = result.metadata['lower_band_1'].iloc[-1]
        
        assert abs((upper_1 - vwap_val) - (vwap_val - lower_1)) < 0.01
    
    def test_vwap_anchors(self, sample_price_data):
        """Test different VWAP anchor points."""
        vwap = VolumeWeightedAveragePrice()
        
        anchors = ['session', 'week', 'month']
        results = {}
        
        for anchor in anchors:
            result = vwap.calculate(
                sample_price_data['high'],
                sample_price_data['low'],
                sample_price_data['close'],
                sample_price_data['volume'],
                anchor=anchor
            )
            results[anchor] = result
            assert result.metadata['anchor'] == anchor
        
        # Different anchors should produce different results
        assert not np.array_equal(
            results['session'].values.values,
            results['month'].values.values
        )


class TestAccumulationDistributionLine:
    """Test Accumulation/Distribution Line indicator."""
    
    def test_ad_line_calculation(self, sample_price_data):
        """Test basic A/D Line calculation."""
        ad = AccumulationDistributionLine()
        result = ad.calculate(
            sample_price_data['high'],
            sample_price_data['low'],
            sample_price_data['close'],
            sample_price_data['volume']
        )
        
        assert result is not None
        assert len(result.values) == len(sample_price_data['close'])
        assert result.indicator_name == "A/D Line"
        
        # Check metadata
        assert 'money_flow_multiplier' in result.metadata
        assert 'money_flow_volume' in result.metadata
        assert 'signal_line' in result.metadata
    
    def test_ad_line_accumulation(self, trend_data):
        """Test A/D Line during accumulation phase."""
        ad = AccumulationDistributionLine()
        
        # Create accumulation scenario
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        high = trend_data['uptrend']['close'] * 1.01
        low = trend_data['uptrend']['close'] * 0.99
        
        result = ad.calculate(
            pd.Series(high.values, index=dates),
            pd.Series(low.values, index=dates),
            trend_data['uptrend']['close'],
            trend_data['uptrend']['volume']
        )
        
        # A/D Line should generally increase during uptrend
        ad_values = result.values.values
        positive_changes = sum(1 for i in range(1, len(ad_values)) 
                             if ad_values[i] > ad_values[i-1])
        
        assert positive_changes > len(ad_values) * 0.6  # More than 60% positive
    
    def test_ad_line_divergence(self, sample_price_data):
        """Test A/D Line divergence detection."""
        ad = AccumulationDistributionLine()
        result = ad.calculate(
            sample_price_data['high'],
            sample_price_data['low'],
            sample_price_data['close'],
            sample_price_data['volume']
        )
        
        assert 'divergence' in result.metadata
        assert 'trend' in result.metadata
        assert result.metadata['trend'] in ['accumulation', 'distribution', 'neutral']


class TestVolumeSpikeDetector:
    """Test Volume Spike Detection."""
    
    def test_spike_detection(self, sample_price_data):
        """Test basic spike detection."""
        detector = VolumeSpikeDetector()
        result = detector.calculate(
            sample_price_data['volume'],
            sample_price_data['close']
        )
        
        assert result is not None
        assert len(result.values) == len(sample_price_data['volume'])
        assert result.indicator_name == "Volume Spike Detector"
        
        # Should detect at least some spikes (we added 4)
        spike_count = result.metadata['spike_count']
        assert spike_count > 0
        assert spike_count <= 10  # Shouldn't detect too many
    
    def test_spike_classification(self, sample_price_data):
        """Test spike classification."""
        detector = VolumeSpikeDetector()
        result = detector.calculate(
            sample_price_data['volume'],
            sample_price_data['close']
        )
        
        spike_events = result.metadata['spike_events']
        
        for event in spike_events:
            assert 'type' in event
            assert 'spike_ratio' in event
            assert 'z_score' in event
            assert event['spike_ratio'] > 1  # Should be above average
    
    def test_spike_pattern_analysis(self, sample_price_data):
        """Test spike pattern analysis."""
        detector = VolumeSpikeDetector()
        result = detector.calculate(
            sample_price_data['volume'],
            sample_price_data['close']
        )
        
        analysis = result.metadata['spike_analysis']
        
        assert 'pattern' in analysis
        assert 'frequency' in analysis
        assert 'trend' in analysis
        assert 'avg_spike_ratio' in analysis
        assert 'max_spike_ratio' in analysis
    
    def test_custom_threshold(self):
        """Test spike detection with custom threshold."""
        config = VolumeIndicatorConfig(
            spike_threshold=4.0,  # Higher threshold
            spike_lookback=30
        )
        detector = VolumeSpikeDetector(config)
        
        # Generate data with known spike
        volume = pd.Series([1000000] * 50)
        volume.iloc[25] = 5000000  # 5x spike
        
        result = detector.calculate(volume)
        
        # Should detect the spike
        assert result.metadata['spike_count'] >= 1
        
        # With lower threshold
        config.spike_threshold = 2.0
        detector = VolumeSpikeDetector(config)
        result = detector.calculate(volume)
        
        # Should still detect the spike
        assert result.metadata['spike_count'] >= 1


class TestVolumeAnalyzer:
    """Test comprehensive volume analysis."""
    
    def test_comprehensive_analysis(self, sample_price_data):
        """Test full volume analysis."""
        analyzer = VolumeAnalyzer()
        results = analyzer.analyze(
            sample_price_data['high'],
            sample_price_data['low'],
            sample_price_data['close'],
            sample_price_data['volume']
        )
        
        # Should calculate all indicators
        expected_indicators = ['obv', 'vroc', 'vwap', 'ad_line', 'volume_spikes']
        for indicator in expected_indicators:
            assert indicator in results
            assert results[indicator] is not None
        
        # Should have summary
        assert 'summary' in results
        assert 'composite_signal' in results['summary']
    
    def test_composite_signal_generation(self, sample_price_data):
        """Test composite signal generation."""
        analyzer = VolumeAnalyzer()
        results = analyzer.analyze(
            sample_price_data['high'],
            sample_price_data['low'],
            sample_price_data['close'],
            sample_price_data['volume']
        )
        
        composite_signal = results['summary']['composite_signal']
        
        valid_signals = ['strong_buy', 'buy', 'neutral', 'sell', 'strong_sell']
        assert composite_signal in valid_signals
    
    def test_error_handling(self):
        """Test error handling in comprehensive analysis."""
        analyzer = VolumeAnalyzer()
        
        # Test with insufficient data
        small_data = pd.Series([100, 101, 102])
        small_volume = pd.Series([1000, 1100, 1200])
        
        results = analyzer.analyze(
            small_data,
            small_data,
            small_data,
            small_volume
        )
        
        # Should handle errors gracefully
        assert 'summary' in results
        assert len(results) > 0  # Should have at least some results


class TestIntegration:
    """Integration tests for volume indicators."""
    
    def test_indicator_consistency(self, sample_price_data):
        """Test that indicators give consistent signals."""
        config = VolumeIndicatorConfig()
        
        # Calculate all indicators
        obv = OnBalanceVolume(config)
        ad = AccumulationDistributionLine(config)
        
        obv_result = obv.calculate(
            sample_price_data['close'],
            sample_price_data['volume']
        )
        
        ad_result = ad.calculate(
            sample_price_data['high'],
            sample_price_data['low'],
            sample_price_data['close'],
            sample_price_data['volume']
        )
        
        # Both should detect similar trends
        obv_trend = obv_result.metadata['trend']
        ad_trend = ad_result.metadata['trend']
        
        # Trends should not be completely opposite
        if obv_trend == 'bullish':
            assert ad_trend != 'distribution'
        if obv_trend == 'bearish':
            assert ad_trend != 'accumulation'
    
    def test_real_market_scenario(self):
        """Test with realistic market scenario."""
        # Simulate a pump and dump scenario
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Normal trading for 40 days
        close = [100] * 40
        volume = [1000000] * 40
        
        # Pump phase (20 days)
        for i in range(20):
            close.append(100 + i * 2)  # Price increases
            volume.append(1000000 * (1 + i * 0.2))  # Volume increases
        
        # Dump phase (40 days)
        for i in range(40):
            close.append(140 - i * 3)  # Price crashes
            volume.append(3000000 - i * 50000)  # Volume decreases
        
        close_series = pd.Series(close, index=dates)
        volume_series = pd.Series(volume, index=dates)
        
        # Detect volume spike during pump
        detector = VolumeSpikeDetector()
        result = detector.calculate(volume_series, close_series)
        
        # Should detect spikes during pump phase
        spike_events = result.metadata['spike_events']
        pump_spikes = [s for s in spike_events if 40 <= s['index'] < 60]
        
        assert len(pump_spikes) > 0  # Should detect pump phase spikes