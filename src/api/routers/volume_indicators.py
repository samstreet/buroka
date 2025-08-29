"""
API endpoints for volume-based technical indicators.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from src.api.auth.dependencies import get_current_user
from src.api.models.common import ResponseModel, ErrorResponse
from src.core.indicators.volume import (
    OnBalanceVolume,
    VolumeRateOfChange,
    VolumeWeightedAveragePrice,
    AccumulationDistributionLine,
    VolumeSpikeDetector,
    VolumeAnalyzer,
    VolumeIndicatorConfig
)
from src.data.storage.optimized_clients import OptimizedInfluxDBClient
from src.utils.redis_client import get_redis_client
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/volume", tags=["Volume Indicators"])


class VolumeIndicatorRequest(BaseModel):
    """Request model for volume indicator calculation."""
    symbol: str = Field(..., description="Stock symbol")
    start_date: Optional[datetime] = Field(None, description="Start date for analysis")
    end_date: Optional[datetime] = Field(None, description="End date for analysis")
    indicator_type: str = Field(..., description="Type of volume indicator")
    config: Optional[Dict[str, Any]] = Field(None, description="Indicator configuration")


class VolumeIndicatorResponse(BaseModel):
    """Response model for volume indicators."""
    symbol: str
    indicator: str
    values: List[float]
    timestamps: List[datetime]
    signal: List[float]
    metadata: Dict[str, Any]
    analysis: Optional[Dict[str, Any]] = None


class VolumeSpikeAlert(BaseModel):
    """Model for volume spike alerts."""
    symbol: str
    timestamp: datetime
    volume: float
    average_volume: float
    spike_ratio: float
    spike_type: str
    price_action: str
    z_score: float


@router.get("/indicators/{symbol}/obv", response_model=VolumeIndicatorResponse)
async def get_obv(
    symbol: str,
    period: int = Query(100, description="Number of periods to analyze"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Calculate On-Balance Volume (OBV) for a symbol.
    """
    try:
        # Get market data
        market_data = await _get_market_data(symbol, start_date, end_date, period)
        
        if market_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Calculate OBV
        obv = OnBalanceVolume()
        result = obv.calculate(market_data['close'], market_data['volume'])
        
        # Prepare response
        return VolumeIndicatorResponse(
            symbol=symbol,
            indicator="OBV",
            values=result.values.tolist(),
            timestamps=result.values.index.tolist(),
            signal=result.signal.tolist(),
            metadata={
                'trend': result.metadata.get('trend'),
                'divergence': len(result.metadata.get('divergence', [])),
                'last_obv': result.metadata.get('last_obv'),
                'obv_change_pct': result.metadata.get('obv_change_pct')
            }
        )
        
    except Exception as e:
        logger.error(f"Error calculating OBV for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indicators/{symbol}/vroc", response_model=VolumeIndicatorResponse)
async def get_vroc(
    symbol: str,
    period: int = Query(14, description="VROC calculation period"),
    data_points: int = Query(100, description="Number of data points"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Calculate Volume Rate of Change (VROC) for a symbol.
    """
    try:
        # Get market data
        market_data = await _get_market_data(symbol, start_date, end_date, data_points)
        
        if market_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Calculate VROC
        vroc = VolumeRateOfChange()
        result = vroc.calculate(market_data['volume'], period=period)
        
        # Prepare response
        return VolumeIndicatorResponse(
            symbol=symbol,
            indicator="VROC",
            values=result.values.tolist(),
            timestamps=result.values.index.tolist(),
            signal=result.signal.tolist(),
            metadata={
                'current_vroc': result.metadata.get('current_vroc'),
                'vroc_trend': result.metadata.get('vroc_trend'),
                'extreme_levels': result.metadata.get('extreme_levels', []),
                'period': period
            }
        )
        
    except Exception as e:
        logger.error(f"Error calculating VROC for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indicators/{symbol}/vwap", response_model=VolumeIndicatorResponse)
async def get_vwap(
    symbol: str,
    anchor: str = Query("session", description="VWAP anchor point (session/week/month)"),
    data_points: int = Query(100, description="Number of data points"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Calculate Volume-Weighted Average Price (VWAP) for a symbol.
    """
    try:
        # Get market data
        market_data = await _get_market_data(symbol, start_date, end_date, data_points)
        
        if market_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Calculate VWAP
        vwap = VolumeWeightedAveragePrice()
        result = vwap.calculate(
            market_data['high'],
            market_data['low'],
            market_data['close'],
            market_data['volume'],
            anchor=anchor
        )
        
        # Prepare response with bands
        metadata = result.metadata.copy()
        metadata['upper_band_1'] = metadata['upper_band_1'].tolist() if 'upper_band_1' in metadata else []
        metadata['lower_band_1'] = metadata['lower_band_1'].tolist() if 'lower_band_1' in metadata else []
        metadata['upper_band_2'] = metadata['upper_band_2'].tolist() if 'upper_band_2' in metadata else []
        metadata['lower_band_2'] = metadata['lower_band_2'].tolist() if 'lower_band_2' in metadata else []
        
        return VolumeIndicatorResponse(
            symbol=symbol,
            indicator="VWAP",
            values=result.values.tolist(),
            timestamps=result.values.index.tolist(),
            signal=result.signal.tolist(),
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error calculating VWAP for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indicators/{symbol}/ad-line", response_model=VolumeIndicatorResponse)
async def get_ad_line(
    symbol: str,
    data_points: int = Query(100, description="Number of data points"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Calculate Accumulation/Distribution Line for a symbol.
    """
    try:
        # Get market data
        market_data = await _get_market_data(symbol, start_date, end_date, data_points)
        
        if market_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Calculate A/D Line
        ad = AccumulationDistributionLine()
        result = ad.calculate(
            market_data['high'],
            market_data['low'],
            market_data['close'],
            market_data['volume']
        )
        
        # Prepare response
        metadata = {
            'current_ad': result.metadata.get('current_ad'),
            'trend': result.metadata.get('trend'),
            'divergence_count': len(result.metadata.get('divergence', [])),
            'signal_line': result.metadata.get('signal_line', pd.Series()).tolist()
        }
        
        return VolumeIndicatorResponse(
            symbol=symbol,
            indicator="A/D Line",
            values=result.values.tolist(),
            timestamps=result.values.index.tolist(),
            signal=result.signal.tolist(),
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error calculating A/D Line for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spikes/{symbol}", response_model=List[VolumeSpikeAlert])
async def get_volume_spikes(
    symbol: str,
    threshold: float = Query(2.0, description="Spike threshold multiplier"),
    lookback: int = Query(20, description="Lookback period for average"),
    data_points: int = Query(100, description="Number of data points"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Detect volume spikes for a symbol.
    """
    try:
        # Get market data
        market_data = await _get_market_data(symbol, start_date, end_date, data_points)
        
        if market_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Configure and run spike detector
        config = VolumeIndicatorConfig(
            spike_threshold=threshold,
            spike_lookback=lookback
        )
        detector = VolumeSpikeDetector(config)
        result = detector.calculate(market_data['volume'], market_data['close'])
        
        # Convert spike events to alerts
        alerts = []
        for event in result.metadata.get('spike_events', []):
            spike_type, price_action = event['type'].split('_')
            
            alerts.append(VolumeSpikeAlert(
                symbol=symbol,
                timestamp=event['timestamp'],
                volume=event['volume'],
                average_volume=event['average_volume'],
                spike_ratio=event['spike_ratio'],
                spike_type=spike_type,
                price_action=price_action,
                z_score=event['z_score']
            ))
        
        return alerts
        
    except Exception as e:
        logger.error(f"Error detecting volume spikes for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_volume(
    request: VolumeIndicatorRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Perform comprehensive volume analysis for a symbol.
    """
    try:
        # Get market data
        market_data = await _get_market_data(
            request.symbol,
            request.start_date,
            request.end_date,
            100
        )
        
        if market_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Configure analyzer
        config = VolumeIndicatorConfig()
        if request.config:
            for key, value in request.config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Run analysis
        analyzer = VolumeAnalyzer(config)
        results = analyzer.analyze(
            market_data['high'],
            market_data['low'],
            market_data['close'],
            market_data['volume']
        )
        
        # Prepare response
        response = {
            'symbol': request.symbol,
            'analysis_period': {
                'start': market_data.index[0].isoformat(),
                'end': market_data.index[-1].isoformat(),
                'data_points': len(market_data)
            },
            'indicators': {},
            'composite_signal': results['summary']['composite_signal'],
            'timestamp': results['summary']['timestamp']
        }
        
        # Add indicator results
        for name, result in results.items():
            if name != 'summary' and hasattr(result, 'values'):
                response['indicators'][name] = {
                    'current_value': result.values.iloc[-1] if not result.values.empty else None,
                    'signal': result.signal.iloc[-1] if not result.signal.empty else 0,
                    'metadata': {
                        k: v for k, v in result.metadata.items()
                        if not isinstance(v, (pd.Series, pd.DataFrame))
                    }
                }
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing volume for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare/{symbols}", response_model=Dict[str, Any])
async def compare_volume_indicators(
    symbols: str,
    indicator: str = Query("obv", description="Indicator to compare (obv/vroc/vwap/ad_line)"),
    period: int = Query(50, description="Analysis period"),
    current_user: dict = Depends(get_current_user)
):
    """
    Compare volume indicators across multiple symbols.
    """
    try:
        symbol_list = symbols.split(',')
        if len(symbol_list) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 symbols allowed")
        
        comparison = {
            'indicator': indicator,
            'symbols': {},
            'analysis': {}
        }
        
        for symbol in symbol_list:
            # Get market data
            market_data = await _get_market_data(symbol, None, None, period)
            
            if market_data.empty:
                continue
            
            # Calculate requested indicator
            if indicator == "obv":
                calc = OnBalanceVolume()
                result = calc.calculate(market_data['close'], market_data['volume'])
            elif indicator == "vroc":
                calc = VolumeRateOfChange()
                result = calc.calculate(market_data['volume'])
            elif indicator == "vwap":
                calc = VolumeWeightedAveragePrice()
                result = calc.calculate(
                    market_data['high'],
                    market_data['low'],
                    market_data['close'],
                    market_data['volume']
                )
            elif indicator == "ad_line":
                calc = AccumulationDistributionLine()
                result = calc.calculate(
                    market_data['high'],
                    market_data['low'],
                    market_data['close'],
                    market_data['volume']
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unknown indicator: {indicator}")
            
            # Store results
            comparison['symbols'][symbol] = {
                'current_value': result.values.iloc[-1] if not result.values.empty else None,
                'signal': result.signal.iloc[-1] if not result.signal.empty else 0,
                'trend': result.metadata.get('trend', 'unknown')
            }
        
        # Analyze comparison
        signals = [v['signal'] for v in comparison['symbols'].values()]
        comparison['analysis'] = {
            'bullish_count': sum(1 for s in signals if s > 0),
            'bearish_count': sum(1 for s in signals if s < 0),
            'neutral_count': sum(1 for s in signals if s == 0),
            'consensus': 'bullish' if sum(signals) > 0 else 'bearish' if sum(signals) < 0 else 'neutral'
        }
        
        return comparison
        
    except Exception as e:
        logger.error(f"Error comparing volume indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _get_market_data(
    symbol: str,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    limit: int
) -> pd.DataFrame:
    """
    Helper function to get market data from storage.
    """
    try:
        # Check cache first
        redis_client = await get_redis_client()
        cache_key = f"market_data:{symbol}:{start_date}:{end_date}:{limit}"
        
        cached_data = await redis_client.get(cache_key)
        if cached_data:
            return pd.read_json(cached_data)
        
        # Get from database
        client = OptimizedInfluxDBClient()
        
        # Build query
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=limit)
        
        query = f"""
        SELECT 
            LAST(open_price) as open,
            MAX(high_price) as high,
            MIN(low_price) as low,
            LAST(close_price) as close,
            SUM(volume) as volume
        FROM market_data
        WHERE symbol = '{symbol}'
            AND time >= '{start_date.isoformat()}'
            AND time <= '{end_date.isoformat()}'
        GROUP BY time(1d)
        ORDER BY time DESC
        LIMIT {limit}
        """
        
        result = await client.query(query)
        
        if not result:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(result)
        df.set_index('time', inplace=True)
        
        # Cache for 5 minutes
        await redis_client.setex(
            cache_key,
            300,
            df.to_json()
        )
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        return pd.DataFrame()