"""
Momentum technical indicators
"""

import pandas as pd
import numpy as np
from typing import Union

from .base import OscillatorIndicator, safe_division, validate_series_length
from .moving_averages import SimpleMovingAverage, ExponentialMovingAverage


class RSI(OscillatorIndicator):
    """
    Relative Strength Index (RSI) oscillator.
    
    Measures the speed and magnitude of price changes.
    Values range from 0 to 100, with readings above 70
    typically considered overbought and below 30 oversold.
    """
    
    def get_required_columns(self) -> list[str]:
        """RSI requires close prices."""
        return ['close']
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate RSI.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with RSI values (0-100)
        """
        self.validate_data(data)
        
        close_prices = data['close']
        
        # Calculate price changes
        delta = close_prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses using Wilder's smoothing
        # (equivalent to EMA with alpha = 1/period)
        alpha = 1.0 / self.period
        
        avg_gains = gains.ewm(alpha=alpha, adjust=False).mean()
        avg_losses = losses.ewm(alpha=alpha, adjust=False).mean()
        
        # Calculate relative strength
        rs = safe_division(avg_gains, avg_losses)
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        # Set initial values to NaN
        rsi.iloc[:self.period] = np.nan
        
        return rsi


class MACD(OscillatorIndicator):
    """
    Moving Average Convergence Divergence (MACD) indicator.
    
    Shows the relationship between two moving averages of a security's price.
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize MACD.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
        """
        # Set attributes before calling super().__init__
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # Use slow_period as the main period for base class
        super().__init__(slow_period, f"MACD({fast_period},{slow_period},{signal_period})")
        
        self.parameters.update({
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period
        })
    
    def _validate_parameters(self) -> None:
        """Validate MACD parameters."""
        if self.fast_period <= 0 or self.slow_period <= 0 or self.signal_period <= 0:
            raise ValueError("All periods must be positive")
        
        if self.fast_period >= self.slow_period:
            raise ValueError("Fast period must be less than slow period")
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD, Signal line, and Histogram.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with 'macd', 'signal', and 'histogram' columns
        """
        self.validate_data(data)
        
        close_prices = data['close']
        
        # Calculate fast and slow EMAs
        fast_ema = ExponentialMovingAverage(self.fast_period)
        slow_ema = ExponentialMovingAverage(self.slow_period)
        
        fast_values = fast_ema.calculate(data)
        slow_values = slow_ema.calculate(data)
        
        # Calculate MACD line
        macd_line = fast_values - slow_values
        
        # Calculate signal line (EMA of MACD line)
        signal_data = pd.DataFrame({'close': macd_line})
        signal_ema = ExponentialMovingAverage(self.signal_period)
        signal_line = signal_ema.calculate(signal_data)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Create result DataFrame
        result = pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }, index=data.index)
        
        return result


class StochasticOscillator(OscillatorIndicator):
    """
    Stochastic Oscillator (%K and %D).
    
    Compares a particular closing price to a range of prices
    over a certain period of time.
    """
    
    def __init__(self, k_period: int = 14, d_period: int = 3, smooth_k: int = 3):
        """
        Initialize Stochastic Oscillator.
        
        Args:
            k_period: Period for %K calculation
            d_period: Period for %D smoothing
            smooth_k: Period for %K smoothing (0 for fast stochastic)
        """
        super().__init__(k_period, f"Stochastic({k_period},{d_period},{smooth_k})")
        
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
        
        self.parameters.update({
            'k_period': k_period,
            'd_period': d_period,
            'smooth_k': smooth_k
        })
    
    def _validate_parameters(self) -> None:
        """Validate Stochastic parameters."""
        if self.k_period <= 0 or self.d_period <= 0:
            raise ValueError("K and D periods must be positive")
        
        if self.smooth_k < 0:
            raise ValueError("Smooth K period must be non-negative")
    
    def get_required_columns(self) -> list[str]:
        """Stochastic requires high, low, and close prices."""
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic %K and %D.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with 'percent_k' and 'percent_d' columns
        """
        self.validate_data(data)
        
        high_prices = data['high']
        low_prices = data['low']
        close_prices = data['close']
        
        # Calculate lowest low and highest high over K period
        lowest_low = low_prices.rolling(window=self.k_period, min_periods=self.k_period).min()
        highest_high = high_prices.rolling(window=self.k_period, min_periods=self.k_period).max()
        
        # Calculate raw %K
        raw_k = 100 * safe_division(close_prices - lowest_low, highest_high - lowest_low)
        
        # Smooth %K if requested
        if self.smooth_k > 1:
            percent_k = raw_k.rolling(window=self.smooth_k, min_periods=self.smooth_k).mean()
        else:
            percent_k = raw_k
        
        # Calculate %D (moving average of %K)
        percent_d = percent_k.rolling(window=self.d_period, min_periods=self.d_period).mean()
        
        # Create result DataFrame
        result = pd.DataFrame({
            'percent_k': percent_k,
            'percent_d': percent_d
        }, index=data.index)
        
        return result


class WilliamsPercentR(OscillatorIndicator):
    """
    Williams %R oscillator.
    
    Momentum indicator that measures overbought and oversold levels.
    Values range from -100 to 0.
    """
    
    def get_required_columns(self) -> list[str]:
        """Williams %R requires high, low, and close prices."""
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Williams %R.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with Williams %R values (-100 to 0)
        """
        self.validate_data(data)
        
        high_prices = data['high']
        low_prices = data['low']
        close_prices = data['close']
        
        # Calculate highest high and lowest low over period
        highest_high = high_prices.rolling(window=self.period, min_periods=self.period).max()
        lowest_low = low_prices.rolling(window=self.period, min_periods=self.period).min()
        
        # Calculate Williams %R
        williams_r = -100 * safe_division(highest_high - close_prices, highest_high - lowest_low)
        
        return williams_r


class RateOfChange(OscillatorIndicator):
    """
    Rate of Change (ROC) momentum oscillator.
    
    Measures the percentage change in price from n periods ago.
    """
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Rate of Change.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with ROC values (percentage)
        """
        self.validate_data(data)
        
        close_prices = data['close']
        
        # Calculate rate of change
        roc = ((close_prices - close_prices.shift(self.period)) / close_prices.shift(self.period)) * 100
        
        return roc


class CommodityChannelIndex(OscillatorIndicator):
    """
    Commodity Channel Index (CCI).
    
    Measures the variation of a security's price from its statistical average.
    """
    
    def __init__(self, period: int = 20, constant: float = 0.015):
        """
        Initialize CCI.
        
        Args:
            period: Number of periods
            constant: Scaling constant (typically 0.015)
        """
        super().__init__(period, f"CCI({period})")
        self.constant = constant
        self.parameters['constant'] = constant
    
    def get_required_columns(self) -> list[str]:
        """CCI requires high, low, and close prices."""
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Commodity Channel Index.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with CCI values
        """
        self.validate_data(data)
        
        high_prices = data['high']
        low_prices = data['low']
        close_prices = data['close']
        
        # Calculate typical price
        typical_price = (high_prices + low_prices + close_prices) / 3
        
        # Calculate moving average of typical price
        sma_tp = typical_price.rolling(window=self.period, min_periods=self.period).mean()
        
        # Calculate mean absolute deviation
        mad = typical_price.rolling(window=self.period, min_periods=self.period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        
        # Calculate CCI
        cci = (typical_price - sma_tp) / (self.constant * mad)
        
        return cci