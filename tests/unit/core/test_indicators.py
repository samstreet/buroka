"""
Unit tests for technical indicators - TDD approach
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the indicators we'll be creating
from src.core.indicators.base import TechnicalIndicator
from src.core.indicators.moving_averages import SimpleMovingAverage, ExponentialMovingAverage, WeightedMovingAverage
from src.core.indicators.momentum import RSI, MACD, StochasticOscillator
from src.core.indicators.volatility import BollingerBands


class TestTechnicalIndicatorBase:
    """Test base technical indicator functionality."""
    
    def test_indicator_interface(self):
        """Test: Base indicator should enforce interface."""
        
        # Should not be able to instantiate abstract base class
        with pytest.raises(TypeError):
            TechnicalIndicator()
    
    def create_sample_data(self, length=100):
        """Create sample price data for testing."""
        dates = [datetime.now() - timedelta(days=i) for i in range(length)]
        dates.reverse()
        
        # Generate realistic price data
        np.random.seed(42)  # Reproducible results
        prices = []
        current_price = 100.0
        
        for _ in range(length):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            current_price *= (1 + change)
            prices.append(current_price)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': [p * np.random.uniform(0.99, 1.01) for p in prices],
            'high': [p * np.random.uniform(1.01, 1.05) for p in prices],
            'low': [p * np.random.uniform(0.95, 0.99) for p in prices],
            'close': prices,
            'volume': [np.random.randint(10000, 100000) for _ in range(length)]
        })


class TestSimpleMovingAverage:
    """Test Simple Moving Average indicator."""
    
    def test_sma_calculation(self):
        """Test: Should calculate SMA correctly."""
        data = pd.DataFrame({
            'close': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        
        sma = SimpleMovingAverage(period=5)
        result = sma.calculate(data)
        
        # First 4 values should be NaN, 5th should be average of first 5
        assert pd.isna(result.iloc[3])
        assert result.iloc[4] == 30.0  # (10+20+30+40+50)/5
        assert result.iloc[9] == 80.0  # (60+70+80+90+100)/5
    
    def test_sma_insufficient_data(self):
        """Test: Should handle insufficient data gracefully."""
        data = pd.DataFrame({'close': [10, 20, 30]})
        
        sma = SimpleMovingAverage(period=5)
        result = sma.calculate(data)
        
        # All values should be NaN when insufficient data
        assert all(pd.isna(result))
    
    def test_sma_validation(self):
        """Test: Should validate parameters."""
        with pytest.raises(ValueError):
            SimpleMovingAverage(period=0)
        
        with pytest.raises(ValueError):
            SimpleMovingAverage(period=-1)


class TestExponentialMovingAverage:
    """Test Exponential Moving Average indicator."""
    
    def test_ema_calculation(self):
        """Test: Should calculate EMA correctly."""
        data = pd.DataFrame({
            'close': [22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29]
        })
        
        ema = ExponentialMovingAverage(period=5)
        result = ema.calculate(data)
        
        # EMA should not be NaN after sufficient periods
        assert not pd.isna(result.iloc[4])
        assert not pd.isna(result.iloc[9])
        
        # EMA should be more responsive than SMA
        sma = SimpleMovingAverage(period=5)
        sma_result = sma.calculate(data)
        
        # Last EMA value should be different from SMA
        assert abs(result.iloc[9] - sma_result.iloc[9]) > 0.01


class TestRSI:
    """Test Relative Strength Index indicator."""
    
    def test_rsi_calculation(self):
        """Test: Should calculate RSI correctly."""
        # Create data with clear up/down trend
        prices = [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89,
                 46.03, 46.83, 47.69, 46.49, 46.26, 47.09, 47.37, 47.20, 47.72, 47.90]
        
        data = pd.DataFrame({'close': prices})
        
        rsi = RSI(period=14)
        result = rsi.calculate(data)
        
        # RSI should be between 0 and 100
        valid_values = result.dropna()
        assert all(valid_values >= 0)
        assert all(valid_values <= 100)
        
        # Should have NaN for first 14 values
        assert sum(pd.isna(result)) == 14
    
    def test_rsi_boundary_conditions(self):
        """Test: Should handle boundary conditions."""
        # All increasing prices should approach 100
        data = pd.DataFrame({'close': range(1, 21)})
        rsi = RSI(period=14)
        result = rsi.calculate(data)
        
        assert result.iloc[-1] > 90  # Should be very high
        
        # All decreasing prices should approach 0
        data = pd.DataFrame({'close': range(20, 0, -1)})
        result = rsi.calculate(data)
        
        assert result.iloc[-1] < 10  # Should be very low


class TestMACD:
    """Test MACD indicator."""
    
    def test_macd_calculation(self):
        """Test: Should calculate MACD correctly."""
        # Create sample data
        data = pd.DataFrame({
            'close': [22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29,
                     22.15, 22.39, 22.38, 22.61, 23.36, 24.05, 23.75, 23.83, 23.95, 23.63,
                     23.82, 23.87, 23.65, 23.19, 23.10, 23.33, 22.68, 23.10, 22.40, 22.17]
        })
        
        macd = MACD(fast_period=12, slow_period=26, signal_period=9)
        result = macd.calculate(data)
        
        # Should return DataFrame with MACD, Signal, and Histogram
        assert 'macd' in result.columns
        assert 'signal' in result.columns
        assert 'histogram' in result.columns
        
        # Histogram should be MACD - Signal
        valid_rows = ~result['macd'].isna() & ~result['signal'].isna()
        expected_histogram = result.loc[valid_rows, 'macd'] - result.loc[valid_rows, 'signal']
        actual_histogram = result.loc[valid_rows, 'histogram']
        
        np.testing.assert_array_almost_equal(actual_histogram, expected_histogram, decimal=6)


class TestBollingerBands:
    """Test Bollinger Bands indicator."""
    
    def test_bollinger_bands_calculation(self):
        """Test: Should calculate Bollinger Bands correctly."""
        data = pd.DataFrame({
            'close': [20, 21, 22, 23, 24, 25, 24, 23, 22, 21, 20, 19, 18, 19, 20, 21, 22, 23, 24, 25]
        })
        
        bb = BollingerBands(period=10, std_dev=2)
        result = bb.calculate(data)
        
        # Should return DataFrame with upper, middle, lower bands
        assert 'upper_band' in result.columns
        assert 'middle_band' in result.columns
        assert 'lower_band' in result.columns
        
        # Middle band should be SMA
        sma = SimpleMovingAverage(period=10)
        expected_middle = sma.calculate(data)
        
        valid_rows = ~result['middle_band'].isna()
        np.testing.assert_array_almost_equal(
            result.loc[valid_rows, 'middle_band'], 
            expected_middle.loc[valid_rows], 
            decimal=6
        )
        
        # Upper band should be above middle, lower should be below
        for idx in result.index:
            if not pd.isna(result.loc[idx, 'upper_band']):
                assert result.loc[idx, 'upper_band'] > result.loc[idx, 'middle_band']
                assert result.loc[idx, 'lower_band'] < result.loc[idx, 'middle_band']


class TestStochasticOscillator:
    """Test Stochastic Oscillator indicator."""
    
    def test_stochastic_calculation(self):
        """Test: Should calculate Stochastic correctly."""
        # Create sample OHLC data
        data = pd.DataFrame({
            'high': [110, 115, 120, 118, 125, 122, 128, 130, 135, 132],
            'low': [105, 108, 115, 112, 118, 119, 125, 127, 130, 128],
            'close': [108, 112, 118, 115, 122, 120, 127, 129, 132, 130]
        })
        
        stoch = StochasticOscillator(k_period=5, d_period=3)
        result = stoch.calculate(data)
        
        # Should return DataFrame with %K and %D
        assert 'percent_k' in result.columns
        assert 'percent_d' in result.columns
        
        # Values should be between 0 and 100
        valid_k = result['percent_k'].dropna()
        valid_d = result['percent_d'].dropna()
        
        assert all(valid_k >= 0) and all(valid_k <= 100)
        assert all(valid_d >= 0) and all(valid_d <= 100)