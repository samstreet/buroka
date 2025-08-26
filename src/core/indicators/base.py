"""
Base classes for technical indicators following SOLID principles
"""

from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicator(ABC):
    """
    Abstract base class for all technical indicators.
    
    Follows the Single Responsibility Principle - each indicator
    is responsible for one specific calculation.
    """
    
    def __init__(self, name: str, parameters: Dict[str, Any]):
        """
        Initialize technical indicator.
        
        Args:
            name: Human-readable name of the indicator
            parameters: Dictionary of indicator parameters
        """
        self.name = name
        self.parameters = parameters
        self.created_at = datetime.now()
        self._validate_parameters()
    
    @abstractmethod
    def _validate_parameters(self) -> None:
        """Validate indicator parameters. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate the technical indicator.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series or DataFrame with indicator values
        """
        pass
    
    def get_required_columns(self) -> list[str]:
        """Get list of required columns for this indicator."""
        return ['close']  # Default to close price
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data has required columns.
        
        Args:
            data: Input DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = self.get_required_columns()
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) == 0:
            raise ValueError("Input data cannot be empty")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get indicator metadata."""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'created_at': self.created_at.isoformat(),
            'required_columns': self.get_required_columns()
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.parameters})"


class MovingAverageIndicator(TechnicalIndicator):
    """Base class for moving average indicators."""
    
    def __init__(self, period: int, name: str = None):
        """
        Initialize moving average indicator.
        
        Args:
            period: Number of periods for the moving average
            name: Optional custom name
        """
        if name is None:
            name = f"{self.__class__.__name__}({period})"
        
        self.period = period  # Set this before calling super().__init__
        parameters = {'period': period}
        super().__init__(name, parameters)
    
    def _validate_parameters(self) -> None:
        """Validate moving average parameters."""
        if self.period <= 0:
            raise ValueError("Period must be positive")
        
        if not isinstance(self.period, int):
            raise ValueError("Period must be an integer")


class OscillatorIndicator(TechnicalIndicator):
    """Base class for oscillator indicators (RSI, Stochastic, etc.)."""
    
    def __init__(self, period: int, name: str = None):
        """
        Initialize oscillator indicator.
        
        Args:
            period: Number of periods for calculation
            name: Optional custom name
        """
        if name is None:
            name = f"{self.__class__.__name__}({period})"
        
        self.period = period  # Set this before calling super().__init__
        parameters = {'period': period}
        super().__init__(name, parameters)
    
    def _validate_parameters(self) -> None:
        """Validate oscillator parameters."""
        if self.period <= 0:
            raise ValueError("Period must be positive")
        
        if not isinstance(self.period, int):
            raise ValueError("Period must be an integer")


class VolatilityIndicator(TechnicalIndicator):
    """Base class for volatility indicators."""
    
    def __init__(self, period: int, std_dev: float = 2.0, name: str = None):
        """
        Initialize volatility indicator.
        
        Args:
            period: Number of periods for calculation
            std_dev: Standard deviation multiplier
            name: Optional custom name
        """
        if name is None:
            name = f"{self.__class__.__name__}({period}, {std_dev})"
        
        self.period = period  # Set these before calling super().__init__
        self.std_dev = std_dev
        parameters = {'period': period, 'std_dev': std_dev}
        super().__init__(name, parameters)
    
    def _validate_parameters(self) -> None:
        """Validate volatility indicator parameters."""
        if self.period <= 0:
            raise ValueError("Period must be positive")
        
        if self.std_dev <= 0:
            raise ValueError("Standard deviation multiplier must be positive")


class IndicatorCalculationError(Exception):
    """Custom exception for indicator calculation errors."""
    pass


def safe_division(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Perform safe division avoiding division by zero.
    
    Args:
        numerator: Numerator series
        denominator: Denominator series
        
    Returns:
        Series with safe division results
    """
    return numerator / denominator.replace(0, np.nan)


def validate_series_length(series: pd.Series, min_length: int, name: str) -> None:
    """
    Validate that a series has minimum required length.
    
    Args:
        series: Series to validate
        min_length: Minimum required length
        name: Name of the series for error messages
        
    Raises:
        IndicatorCalculationError: If series is too short
    """
    if len(series) < min_length:
        raise IndicatorCalculationError(
            f"{name} requires at least {min_length} data points, got {len(series)}"
        )