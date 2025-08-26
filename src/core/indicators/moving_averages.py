"""
Moving Average technical indicators
"""

import pandas as pd
import numpy as np
from typing import Union

from .base import MovingAverageIndicator, validate_series_length


class SimpleMovingAverage(MovingAverageIndicator):
    """
    Simple Moving Average (SMA) indicator.
    
    The SMA is calculated as the arithmetic mean of closing prices
    over a specified number of periods.
    """
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with SMA values
        """
        self.validate_data(data)
        
        close_prices = data['close']
        
        # Calculate rolling mean
        sma = close_prices.rolling(window=self.period, min_periods=self.period).mean()
        
        return sma


class ExponentialMovingAverage(MovingAverageIndicator):
    """
    Exponential Moving Average (EMA) indicator.
    
    The EMA gives more weight to recent prices, making it more
    responsive to new information than SMA.
    """
    
    def __init__(self, period: int, alpha: float = None):
        """
        Initialize EMA.
        
        Args:
            period: Number of periods
            alpha: Smoothing factor (if None, calculated as 2/(period+1))
        """
        if alpha is None:
            alpha = 2.0 / (period + 1)
        
        self.alpha = alpha
        super().__init__(period)
        self.parameters['alpha'] = alpha
    
    def _validate_parameters(self) -> None:
        """Validate EMA parameters."""
        super()._validate_parameters()
        
        if not (0 < self.alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1")
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with EMA values
        """
        self.validate_data(data)
        
        close_prices = data['close']
        
        # Calculate EMA using pandas ewm
        ema = close_prices.ewm(alpha=self.alpha, adjust=False).mean()
        
        # Set first (period-1) values to NaN to match SMA behavior
        ema.iloc[:self.period-1] = np.nan
        
        return ema


class WeightedMovingAverage(MovingAverageIndicator):
    """
    Weighted Moving Average (WMA) indicator.
    
    The WMA assigns different weights to each data point,
    with more recent prices having higher weights.
    """
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Weighted Moving Average.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with WMA values
        """
        self.validate_data(data)
        
        close_prices = data['close']
        
        # Create weights: 1, 2, 3, ..., period
        weights = np.arange(1, self.period + 1)
        
        def calculate_wma(series):
            if len(series) < self.period:
                return np.nan
            
            # Get last 'period' values
            values = series.iloc[-self.period:]
            
            # Calculate weighted average
            return np.sum(values * weights) / np.sum(weights)
        
        # Apply rolling calculation
        wma = close_prices.rolling(window=self.period, min_periods=self.period).apply(
            lambda x: calculate_wma(pd.Series(x)), raw=False
        )
        
        return wma


class AdaptiveMovingAverage(MovingAverageIndicator):
    """
    Adaptive Moving Average (KAMA - Kaufman's Adaptive Moving Average).
    
    Adjusts the smoothing constant based on market volatility.
    """
    
    def __init__(self, period: int, fast_period: int = 2, slow_period: int = 30):
        """
        Initialize KAMA.
        
        Args:
            period: Period for efficiency ratio calculation
            fast_period: Fast smoothing constant period
            slow_period: Slow smoothing constant period
        """
        super().__init__(period)
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        
        self.parameters.update({
            'fast_period': fast_period,
            'slow_period': slow_period
        })
        
        # Calculate smoothing constants
        self.fast_sc = 2.0 / (fast_period + 1)
        self.slow_sc = 2.0 / (slow_period + 1)
    
    def _validate_parameters(self) -> None:
        """Validate KAMA parameters."""
        super()._validate_parameters()
        
        if self.fast_period >= self.slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        if self.fast_period <= 0 or self.slow_period <= 0:
            raise ValueError("Fast and slow periods must be positive")
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Kaufman's Adaptive Moving Average.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with KAMA values
        """
        self.validate_data(data)
        
        close_prices = data['close']
        
        # Calculate direction (change over period)
        direction = abs(close_prices - close_prices.shift(self.period))
        
        # Calculate volatility (sum of absolute changes)
        volatility = abs(close_prices - close_prices.shift(1)).rolling(window=self.period).sum()
        
        # Calculate efficiency ratio
        efficiency_ratio = direction / volatility
        
        # Calculate smoothing constant
        smoothing_constant = (efficiency_ratio * (self.fast_sc - self.slow_sc) + self.slow_sc) ** 2
        
        # Initialize KAMA series
        kama = pd.Series(index=close_prices.index, dtype=float)
        kama.iloc[0] = close_prices.iloc[0]
        
        # Calculate KAMA iteratively
        for i in range(1, len(close_prices)):
            if pd.isna(smoothing_constant.iloc[i]):
                kama.iloc[i] = np.nan
            else:
                kama.iloc[i] = kama.iloc[i-1] + smoothing_constant.iloc[i] * (close_prices.iloc[i] - kama.iloc[i-1])
        
        # Set initial values to NaN
        kama.iloc[:self.period] = np.nan
        
        return kama


class TripleExponentialMovingAverage(MovingAverageIndicator):
    """
    Triple Exponential Moving Average (TEMA).
    
    Reduces lag by applying exponential smoothing three times.
    """
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Triple Exponential Moving Average.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with TEMA values
        """
        self.validate_data(data)
        
        close_prices = data['close']
        
        # Calculate first EMA
        ema1 = close_prices.ewm(span=self.period, adjust=False).mean()
        
        # Calculate second EMA of first EMA
        ema2 = ema1.ewm(span=self.period, adjust=False).mean()
        
        # Calculate third EMA of second EMA
        ema3 = ema2.ewm(span=self.period, adjust=False).mean()
        
        # Calculate TEMA: 3*EMA1 - 3*EMA2 + EMA3
        tema = 3 * ema1 - 3 * ema2 + ema3
        
        # Set initial values to NaN to match period requirements
        tema.iloc[:self.period*2] = np.nan
        
        return tema


class MovingAverageFactory:
    """Factory class for creating moving average indicators."""
    
    _indicators = {
        'sma': SimpleMovingAverage,
        'ema': ExponentialMovingAverage,
        'wma': WeightedMovingAverage,
        'kama': AdaptiveMovingAverage,
        'tema': TripleExponentialMovingAverage
    }
    
    @classmethod
    def create(cls, indicator_type: str, **kwargs) -> MovingAverageIndicator:
        """
        Create a moving average indicator.
        
        Args:
            indicator_type: Type of moving average ('sma', 'ema', 'wma', 'kama', 'tema')
            **kwargs: Parameters for the indicator
            
        Returns:
            MovingAverageIndicator instance
            
        Raises:
            ValueError: If indicator type is not supported
        """
        indicator_type = indicator_type.lower()
        
        if indicator_type not in cls._indicators:
            available = ', '.join(cls._indicators.keys())
            raise ValueError(f"Unknown indicator type '{indicator_type}'. Available: {available}")
        
        indicator_class = cls._indicators[indicator_type]
        return indicator_class(**kwargs)
    
    @classmethod
    def get_available_indicators(cls) -> list[str]:
        """Get list of available moving average indicators."""
        return list(cls._indicators.keys())