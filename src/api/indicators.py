"""
API endpoints for technical indicators
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from src.core.indicators.moving_averages import MovingAverageFactory, SimpleMovingAverage, ExponentialMovingAverage
from src.core.indicators.momentum import RSI, MACD, StochasticOscillator
from src.core.indicators.volatility import BollingerBands, AverageTrueRange

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/indicators", tags=["Technical Indicators"])


def generate_sample_data(symbol: str = "TEST", days: int = 100) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    dates.reverse()
    
    np.random.seed(42)  # Reproducible results
    prices = []
    current_price = 100.0
    
    for _ in range(days):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        current_price *= (1 + change)
        prices.append(current_price)
    
    # Generate OHLC from close prices
    data = []
    for i, close in enumerate(prices):
        high = close * np.random.uniform(1.01, 1.05)
        low = close * np.random.uniform(0.95, 0.99)
        open_price = close * np.random.uniform(0.98, 1.02)
        volume = np.random.randint(10000, 100000)
        
        data.append({
            'timestamp': dates[i],
            'symbol': symbol,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    return pd.DataFrame(data)


@router.get("/test")
async def test_indicators() -> Dict[str, Any]:
    """Test endpoint to verify indicators are working."""
    try:
        # Generate sample data
        data = generate_sample_data(days=50)
        
        # Test Simple Moving Average
        sma = SimpleMovingAverage(period=10)
        sma_result = sma.calculate(data)
        
        # Test RSI
        rsi = RSI(period=14)
        rsi_result = rsi.calculate(data)
        
        # Test Bollinger Bands
        bb = BollingerBands(period=20, std_dev=2.0)
        bb_result = bb.calculate(data)
        
        return {
            "status": "success",
            "message": "Technical indicators working correctly",
            "sample_data_points": len(data),
            "indicators_tested": {
                "sma_10": {
                    "latest_value": float(sma_result.dropna().iloc[-1]) if not sma_result.dropna().empty else None,
                    "valid_values": int(sma_result.notna().sum())
                },
                "rsi_14": {
                    "latest_value": float(rsi_result.dropna().iloc[-1]) if not rsi_result.dropna().empty else None,
                    "valid_values": int(rsi_result.notna().sum())
                },
                "bollinger_bands": {
                    "upper_band": float(bb_result['upper_band'].dropna().iloc[-1]) if not bb_result['upper_band'].dropna().empty else None,
                    "middle_band": float(bb_result['middle_band'].dropna().iloc[-1]) if not bb_result['middle_band'].dropna().empty else None,
                    "lower_band": float(bb_result['lower_band'].dropna().iloc[-1]) if not bb_result['lower_band'].dropna().empty else None,
                    "valid_values": int(bb_result['upper_band'].notna().sum())
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error testing indicators: {e}")
        raise HTTPException(status_code=500, detail=f"Indicator test failed: {str(e)}")


@router.get("/sma/{symbol}")
async def calculate_sma(
    symbol: str,
    period: int = Query(20, description="Number of periods for SMA"),
    use_sample_data: bool = Query(True, description="Use sample data for testing")
) -> Dict[str, Any]:
    """Calculate Simple Moving Average for a symbol."""
    try:
        if use_sample_data:
            data = generate_sample_data(symbol=symbol, days=max(100, period * 3))
        else:
            # TODO: Fetch real data from database
            raise HTTPException(status_code=501, detail="Real data fetching not implemented yet")
        
        sma = SimpleMovingAverage(period=period)
        result = sma.calculate(data)
        
        # Convert to list for JSON response
        values = []
        for idx, value in result.items():
            if pd.notna(value):
                values.append({
                    "timestamp": data.loc[idx, 'timestamp'].isoformat(),
                    "close_price": float(data.loc[idx, 'close']),
                    "sma_value": float(value)
                })
        
        return {
            "symbol": symbol,
            "indicator": "SMA",
            "parameters": sma.get_metadata(),
            "data_points": len(values),
            "values": values[-20:]  # Return last 20 values
        }
        
    except Exception as e:
        logger.error(f"Error calculating SMA for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"SMA calculation failed: {str(e)}")


@router.get("/rsi/{symbol}")
async def calculate_rsi(
    symbol: str,
    period: int = Query(14, description="Number of periods for RSI"),
    use_sample_data: bool = Query(True, description="Use sample data for testing")
) -> Dict[str, Any]:
    """Calculate RSI for a symbol."""
    try:
        if use_sample_data:
            data = generate_sample_data(symbol=symbol, days=max(100, period * 3))
        else:
            raise HTTPException(status_code=501, detail="Real data fetching not implemented yet")
        
        rsi = RSI(period=period)
        result = rsi.calculate(data)
        
        # Convert to list for JSON response
        values = []
        for idx, value in result.items():
            if pd.notna(value):
                values.append({
                    "timestamp": data.loc[idx, 'timestamp'].isoformat(),
                    "close_price": float(data.loc[idx, 'close']),
                    "rsi_value": float(value)
                })
        
        return {
            "symbol": symbol,
            "indicator": "RSI",
            "parameters": rsi.get_metadata(),
            "data_points": len(values),
            "values": values[-20:],  # Return last 20 values
            "interpretation": {
                "current_rsi": float(result.dropna().iloc[-1]) if not result.dropna().empty else None,
                "overbought_threshold": 70,
                "oversold_threshold": 30,
                "signal": "overbought" if result.dropna().iloc[-1] > 70 else "oversold" if result.dropna().iloc[-1] < 30 else "neutral"
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating RSI for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"RSI calculation failed: {str(e)}")


@router.get("/macd/{symbol}")
async def calculate_macd(
    symbol: str,
    fast_period: int = Query(12, description="Fast EMA period"),
    slow_period: int = Query(26, description="Slow EMA period"),
    signal_period: int = Query(9, description="Signal line period"),
    use_sample_data: bool = Query(True, description="Use sample data for testing")
) -> Dict[str, Any]:
    """Calculate MACD for a symbol."""
    try:
        if use_sample_data:
            data = generate_sample_data(symbol=symbol, days=max(100, slow_period * 3))
        else:
            raise HTTPException(status_code=501, detail="Real data fetching not implemented yet")
        
        macd = MACD(fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)
        result = macd.calculate(data)
        
        # Convert to list for JSON response
        values = []
        for idx in result.index:
            if pd.notna(result.loc[idx, 'macd']):
                values.append({
                    "timestamp": data.loc[idx, 'timestamp'].isoformat(),
                    "close_price": float(data.loc[idx, 'close']),
                    "macd": float(result.loc[idx, 'macd']),
                    "signal": float(result.loc[idx, 'signal']) if pd.notna(result.loc[idx, 'signal']) else None,
                    "histogram": float(result.loc[idx, 'histogram']) if pd.notna(result.loc[idx, 'histogram']) else None
                })
        
        return {
            "symbol": symbol,
            "indicator": "MACD",
            "parameters": macd.get_metadata(),
            "data_points": len(values),
            "values": values[-20:]  # Return last 20 values
        }
        
    except Exception as e:
        logger.error(f"Error calculating MACD for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"MACD calculation failed: {str(e)}")


@router.get("/bollinger/{symbol}")
async def calculate_bollinger_bands(
    symbol: str,
    period: int = Query(20, description="Number of periods for moving average"),
    std_dev: float = Query(2.0, description="Standard deviation multiplier"),
    use_sample_data: bool = Query(True, description="Use sample data for testing")
) -> Dict[str, Any]:
    """Calculate Bollinger Bands for a symbol."""
    try:
        if use_sample_data:
            data = generate_sample_data(symbol=symbol, days=max(100, period * 3))
        else:
            raise HTTPException(status_code=501, detail="Real data fetching not implemented yet")
        
        bb = BollingerBands(period=period, std_dev=std_dev)
        result = bb.calculate(data)
        
        # Convert to list for JSON response
        values = []
        for idx in result.index:
            if pd.notna(result.loc[idx, 'upper_band']):
                values.append({
                    "timestamp": data.loc[idx, 'timestamp'].isoformat(),
                    "close_price": float(data.loc[idx, 'close']),
                    "upper_band": float(result.loc[idx, 'upper_band']),
                    "middle_band": float(result.loc[idx, 'middle_band']),
                    "lower_band": float(result.loc[idx, 'lower_band'])
                })
        
        return {
            "symbol": symbol,
            "indicator": "Bollinger Bands",
            "parameters": bb.get_metadata(),
            "data_points": len(values),
            "values": values[-20:]  # Return last 20 values
        }
        
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Bollinger Bands calculation failed: {str(e)}")


@router.get("/available")
async def get_available_indicators() -> Dict[str, Any]:
    """Get list of available technical indicators."""
    return {
        "moving_averages": {
            "sma": "Simple Moving Average",
            "ema": "Exponential Moving Average",
            "wma": "Weighted Moving Average",
            "kama": "Kaufman's Adaptive Moving Average",
            "tema": "Triple Exponential Moving Average"
        },
        "momentum": {
            "rsi": "Relative Strength Index",
            "macd": "Moving Average Convergence Divergence",
            "stochastic": "Stochastic Oscillator",
            "williams_r": "Williams %R",
            "roc": "Rate of Change",
            "cci": "Commodity Channel Index"
        },
        "volatility": {
            "bollinger_bands": "Bollinger Bands",
            "atr": "Average True Range",
            "keltner_channels": "Keltner Channels",
            "standard_deviation": "Standard Deviation",
            "chaikin_volatility": "Chaikin Volatility",
            "historical_volatility": "Historical Volatility"
        }
    }