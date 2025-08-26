"""
Volatility technical indicators
"""

import pandas as pd
import numpy as np
from typing import Union

from .base import VolatilityIndicator, TechnicalIndicator
from .moving_averages import SimpleMovingAverage


class BollingerBands(VolatilityIndicator):
    """
    Bollinger Bands volatility indicator.
    
    Consists of a middle band (SMA) and upper/lower bands
    that are standard deviations away from the middle band.
    """
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with 'upper_band', 'middle_band', 'lower_band' columns
        """
        self.validate_data(data)
        
        close_prices = data['close']
        
        # Calculate middle band (SMA)
        sma = SimpleMovingAverage(self.period)
        middle_band = sma.calculate(data)
        
        # Calculate rolling standard deviation
        rolling_std = close_prices.rolling(window=self.period, min_periods=self.period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * self.std_dev)
        lower_band = middle_band - (rolling_std * self.std_dev)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band
        }, index=data.index)
        
        return result


class AverageTrueRange(TechnicalIndicator):
    """
    Average True Range (ATR) volatility indicator.
    
    Measures market volatility by decomposing the entire range
    of an asset price for that period.
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize ATR.
        
        Args:
            period: Number of periods for ATR calculation
        """
        super().__init__(f"ATR({period})", {'period': period})
        self.period = period
    
    def _validate_parameters(self) -> None:
        """Validate ATR parameters."""
        if self.period <= 0:
            raise ValueError("Period must be positive")
    
    def get_required_columns(self) -> list[str]:
        """ATR requires high, low, and close prices."""
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with ATR values
        """
        self.validate_data(data)
        
        high_prices = data['high']
        low_prices = data['low']
        close_prices = data['close']
        
        # Calculate True Range components
        tr1 = high_prices - low_prices  # Current high - current low
        tr2 = abs(high_prices - close_prices.shift(1))  # Current high - previous close
        tr3 = abs(low_prices - close_prices.shift(1))   # Current low - previous close
        
        # True Range is the maximum of the three
        true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Calculate ATR using Wilder's smoothing (equivalent to EMA)
        alpha = 1.0 / self.period
        atr = true_range.ewm(alpha=alpha, adjust=False).mean()
        
        # Set initial values to NaN
        atr.iloc[:self.period] = np.nan
        
        return atr


class KeltnerChannels(TechnicalIndicator):
    """
    Keltner Channels volatility indicator.
    
    Similar to Bollinger Bands but uses ATR instead of standard deviation.
    """
    
    def __init__(self, period: int = 20, atr_period: int = 10, multiplier: float = 2.0):
        """
        Initialize Keltner Channels.
        
        Args:
            period: Period for the middle line (EMA)
            atr_period: Period for ATR calculation
            multiplier: ATR multiplier for bands
        """
        super().__init__(f"KeltnerChannels({period},{atr_period},{multiplier})", {
            'period': period,
            'atr_period': atr_period,
            'multiplier': multiplier
        })
        self.period = period
        self.atr_period = atr_period
        self.multiplier = multiplier
    
    def _validate_parameters(self) -> None:
        """Validate Keltner Channels parameters."""
        if self.period <= 0 or self.atr_period <= 0:
            raise ValueError("Periods must be positive")
        
        if self.multiplier <= 0:
            raise ValueError("Multiplier must be positive")
    
    def get_required_columns(self) -> list[str]:
        """Keltner Channels require high, low, and close prices."""
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Keltner Channels.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with 'upper_channel', 'middle_line', 'lower_channel' columns
        """
        self.validate_data(data)
        
        close_prices = data['close']
        
        # Calculate middle line (EMA of close prices)
        middle_line = close_prices.ewm(span=self.period, adjust=False).mean()
        
        # Calculate ATR
        atr_indicator = AverageTrueRange(self.atr_period)
        atr_values = atr_indicator.calculate(data)
        
        # Calculate upper and lower channels
        upper_channel = middle_line + (atr_values * self.multiplier)
        lower_channel = middle_line - (atr_values * self.multiplier)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'upper_channel': upper_channel,
            'middle_line': middle_line,
            'lower_channel': lower_channel
        }, index=data.index)
        
        return result


class StandardDeviation(TechnicalIndicator):
    """
    Standard Deviation volatility indicator.
    
    Measures the dispersion of price from its average over a given period.
    """
    
    def __init__(self, period: int = 20):
        """
        Initialize Standard Deviation indicator.
        
        Args:
            period: Number of periods for calculation
        """
        super().__init__(f"StdDev({period})", {'period': period})
        self.period = period
    
    def _validate_parameters(self) -> None:
        """Validate parameters."""
        if self.period <= 1:
            raise ValueError("Period must be greater than 1")
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Standard Deviation.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with standard deviation values
        """
        self.validate_data(data)
        
        close_prices = data['close']
        
        # Calculate rolling standard deviation
        std_dev = close_prices.rolling(window=self.period, min_periods=self.period).std()
        
        return std_dev


class ChaikinVolatility(TechnicalIndicator):
    """
    Chaikin Volatility indicator.
    
    Measures the rate of change of the trading range (high-low spread).
    """
    
    def __init__(self, period: int = 10, roc_period: int = 10):
        """
        Initialize Chaikin Volatility.
        
        Args:
            period: Period for exponential moving average of H-L spread
            roc_period: Period for rate of change calculation
        """
        super().__init__(f"ChaikinVolatility({period},{roc_period})", {
            'period': period,
            'roc_period': roc_period
        })
        self.period = period
        self.roc_period = roc_period
    
    def _validate_parameters(self) -> None:
        """Validate parameters."""
        if self.period <= 0 or self.roc_period <= 0:
            raise ValueError("Periods must be positive")
    
    def get_required_columns(self) -> list[str]:
        """Chaikin Volatility requires high and low prices."""
        return ['high', 'low']
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Chaikin Volatility.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with Chaikin Volatility values
        """
        self.validate_data(data)
        
        high_prices = data['high']
        low_prices = data['low']
        
        # Calculate high-low spread
        hl_spread = high_prices - low_prices
        
        # Calculate exponential moving average of H-L spread
        ema_spread = hl_spread.ewm(span=self.period, adjust=False).mean()
        
        # Calculate rate of change of the EMA
        chaikin_volatility = ((ema_spread - ema_spread.shift(self.roc_period)) / 
                             ema_spread.shift(self.roc_period)) * 100
        
        return chaikin_volatility


class HistoricalVolatility(TechnicalIndicator):
    """
    Historical Volatility indicator.
    
    Measures the standard deviation of logarithmic price returns
    over a specified period, annualized.
    """
    
    def __init__(self, period: int = 20, trading_days: int = 252):
        """
        Initialize Historical Volatility.
        
        Args:
            period: Number of periods for calculation
            trading_days: Number of trading days per year for annualization
        """
        super().__init__(f"HistoricalVolatility({period})", {
            'period': period,
            'trading_days': trading_days
        })
        self.period = period
        self.trading_days = trading_days
    
    def _validate_parameters(self) -> None:
        """Validate parameters."""
        if self.period <= 1:
            raise ValueError("Period must be greater than 1")
        
        if self.trading_days <= 0:
            raise ValueError("Trading days must be positive")
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Historical Volatility.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with annualized historical volatility values
        """
        self.validate_data(data)
        
        close_prices = data['close']
        
        # Calculate logarithmic returns
        log_returns = np.log(close_prices / close_prices.shift(1))
        
        # Calculate rolling standard deviation of returns
        rolling_std = log_returns.rolling(window=self.period, min_periods=self.period).std()
        
        # Annualize the volatility
        historical_volatility = rolling_std * np.sqrt(self.trading_days) * 100
        
        return historical_volatility