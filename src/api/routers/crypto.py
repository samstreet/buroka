"""
Cryptocurrency API endpoints for Binance data

This module provides REST API endpoints for:
- Getting crypto market data
- Real-time price updates
- Historical kline data
- Market statistics
- Popular trading pairs
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
import asyncio
import logging

from ...services.binance_service import get_binance_service, BinanceService, CryptoTicker, CryptoKline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/crypto", tags=["crypto"])

# Response models
class TickerResponse(BaseModel):
    symbol: str
    price: float
    price_change: float
    price_change_percent: float
    high_price: float
    low_price: float
    open_price: float
    volume: float
    quote_volume: float
    open_time: datetime
    close_time: datetime
    count: int

class KlineResponse(BaseModel):
    symbol: str
    open_time: datetime
    close_time: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    quote_asset_volume: float
    number_of_trades: int

class MarketOverviewResponse(BaseModel):
    total_pairs: int
    popular_pairs: List[str]
    top_gainers: List[str]
    top_losers: List[str]
    high_volume_pairs: List[str]
    supported_intervals: List[str]

class CryptoSymbolResponse(BaseModel):
    symbol: str
    base_asset: str
    quote_asset: str
    status: str

@router.get("/overview", response_model=MarketOverviewResponse)
async def get_market_overview(
    binance: BinanceService = Depends(get_binance_service)
):
    """Get cryptocurrency market overview"""
    try:
        async with binance:
            # Get popular pairs
            popular_pairs = await binance.get_popular_crypto_pairs()
            
            # Get price change leaders
            leaders = await binance.get_price_change_leaders(limit=10)
            
            # Get high volume pairs
            volume_pairs = await binance.get_top_volume_pairs(limit=10)
            
            # Get supported intervals
            intervals = binance.get_binance_intervals()
            
            return MarketOverviewResponse(
                total_pairs=len(popular_pairs),
                popular_pairs=popular_pairs[:20],
                top_gainers=leaders.get('gainers', []),
                top_losers=leaders.get('losers', []),
                high_volume_pairs=volume_pairs,
                supported_intervals=intervals
            )
    except Exception as e:
        logger.error(f"Error getting market overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/symbols", response_model=List[CryptoSymbolResponse])
async def get_crypto_symbols(
    quote_asset: Optional[str] = Query(None, description="Filter by quote asset (e.g., USDT, BTC)"),
    limit: Optional[int] = Query(50, description="Maximum number of symbols to return"),
    binance: BinanceService = Depends(get_binance_service)
):
    """Get active cryptocurrency trading pairs"""
    try:
        async with binance:
            symbols = await binance.get_active_crypto_symbols()
            
            # Filter by quote asset if provided
            if quote_asset:
                symbols = [s for s in symbols if s.quote_asset == quote_asset.upper()]
            
            # Convert to response model and limit results
            response_symbols = [
                CryptoSymbolResponse(
                    symbol=s.symbol,
                    base_asset=s.base_asset,
                    quote_asset=s.quote_asset,
                    status=s.status
                )
                for s in symbols[:limit]
            ]
            
            return response_symbols
    except Exception as e:
        logger.error(f"Error getting crypto symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ticker", response_model=List[TickerResponse])
async def get_ticker_data(
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols (e.g., BTCUSDT,ETHUSDT)"),
    limit: Optional[int] = Query(50, description="Maximum number of tickers to return"),
    binance: BinanceService = Depends(get_binance_service)
):
    """Get 24hr ticker price change statistics"""
    try:
        async with binance:
            if symbols:
                # Get specific symbols
                symbol_list = [s.strip().upper() for s in symbols.split(',')]
                all_tickers = []
                
                for symbol in symbol_list:
                    ticker_data = await binance.get_24hr_ticker(symbol)
                    all_tickers.extend(ticker_data)
            else:
                # Get popular pairs
                popular_pairs = await binance.get_popular_crypto_pairs()
                all_tickers = []
                
                # Get ticker data in batches to avoid overwhelming the API
                for i in range(0, min(len(popular_pairs), limit), 10):
                    batch = popular_pairs[i:i+10]
                    batch_tickers = await binance.get_24hr_ticker()
                    # Filter to only include symbols in our batch
                    filtered_tickers = [t for t in batch_tickers if t.symbol in batch]
                    all_tickers.extend(filtered_tickers)
                    
                    # Small delay between batches
                    await asyncio.sleep(0.1)
            
            # Convert to response model
            response_tickers = [
                TickerResponse(
                    symbol=t.symbol,
                    price=t.price,
                    price_change=t.price_change,
                    price_change_percent=t.price_change_percent,
                    high_price=t.high_price,
                    low_price=t.low_price,
                    open_price=t.open_price,
                    volume=t.volume,
                    quote_volume=t.quote_volume,
                    open_time=t.open_time,
                    close_time=t.close_time,
                    count=t.count
                )
                for t in all_tickers[:limit]
            ]
            
            return response_tickers
    except Exception as e:
        logger.error(f"Error getting ticker data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/klines/{symbol}", response_model=List[KlineResponse])
async def get_kline_data(
    symbol: str,
    interval: str = Query("1h", description="Kline interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)"),
    limit: Optional[int] = Query(500, description="Number of klines to return (max 1000)"),
    start_time: Optional[datetime] = Query(None, description="Start time for klines"),
    end_time: Optional[datetime] = Query(None, description="End time for klines"),
    binance: BinanceService = Depends(get_binance_service)
):
    """Get kline/candlestick data for a cryptocurrency pair"""
    try:
        async with binance:
            # Validate interval
            supported_intervals = binance.get_binance_intervals()
            if interval not in supported_intervals:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported interval. Use one of: {', '.join(supported_intervals)}"
                )
            
            # Limit to maximum
            limit = min(limit or 500, 1000)
            
            klines = await binance.get_klines(
                symbol=symbol.upper(),
                interval=interval,
                limit=limit,
                start_time=start_time,
                end_time=end_time
            )
            
            # Convert to response model
            response_klines = [
                KlineResponse(
                    symbol=k.symbol,
                    open_time=k.open_time,
                    close_time=k.close_time,
                    open_price=k.open_price,
                    high_price=k.high_price,
                    low_price=k.low_price,
                    close_price=k.close_price,
                    volume=k.volume,
                    quote_asset_volume=k.quote_asset_volume,
                    number_of_trades=k.number_of_trades
                )
                for k in klines
            ]
            
            return response_klines
    except Exception as e:
        logger.error(f"Error getting kline data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/popular", response_model=List[str])
async def get_popular_pairs(
    quote_asset: str = Query("USDT", description="Quote asset to filter by"),
    limit: int = Query(20, description="Number of pairs to return"),
    binance: BinanceService = Depends(get_binance_service)
):
    """Get popular cryptocurrency trading pairs"""
    try:
        async with binance:
            if quote_asset.upper() == "ALL":
                popular_pairs = await binance.get_popular_crypto_pairs()
            else:
                # Get all popular pairs and filter
                all_pairs = await binance.get_popular_crypto_pairs()
                popular_pairs = [p for p in all_pairs if p.endswith(quote_asset.upper())]
            
            return popular_pairs[:limit]
    except Exception as e:
        logger.error(f"Error getting popular pairs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/gainers-losers")
async def get_gainers_losers(
    limit: int = Query(10, description="Number of gainers/losers to return"),
    min_volume: float = Query(1000000, description="Minimum 24hr volume in quote asset"),
    binance: BinanceService = Depends(get_binance_service)
):
    """Get biggest price gainers and losers"""
    try:
        async with binance:
            leaders = await binance.get_price_change_leaders(
                min_volume=min_volume,
                limit=limit
            )
            
            return leaders
    except Exception as e:
        logger.error(f"Error getting gainers/losers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/volume-leaders", response_model=List[str])
async def get_volume_leaders(
    quote_asset: str = Query("USDT", description="Quote asset to filter by"),
    limit: int = Query(20, description="Number of pairs to return"),
    binance: BinanceService = Depends(get_binance_service)
):
    """Get cryptocurrency pairs with highest 24hr volume"""
    try:
        async with binance:
            volume_pairs = await binance.get_top_volume_pairs(
                quote_asset=quote_asset.upper(),
                limit=limit
            )
            
            return volume_pairs
    except Exception as e:
        logger.error(f"Error getting volume leaders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/intervals", response_model=List[str])
async def get_supported_intervals():
    """Get supported kline intervals"""
    return BinanceService().get_binance_intervals()

# Health check endpoint
@router.get("/health")
async def health_check(
    binance: BinanceService = Depends(get_binance_service)
):
    """Check if Binance API is accessible"""
    try:
        async with binance:
            # Try to get exchange info as a health check
            exchange_info = await binance.get_exchange_info()
            
            if exchange_info:
                return {
                    "status": "healthy",
                    "binance_api": "accessible",
                    "timestamp": datetime.now(),
                    "server_time": exchange_info.get("serverTime", 0)
                }
            else:
                return {
                    "status": "unhealthy", 
                    "binance_api": "inaccessible",
                    "timestamp": datetime.now()
                }
    except Exception as e:
        return {
            "status": "unhealthy",
            "binance_api": "error", 
            "error": str(e),
            "timestamp": datetime.now()
        }


@router.post("/patterns/detect/{symbol}")
async def detect_crypto_patterns(
    symbol: str,
    timeframe: str = Query("1h", description="Timeframe for pattern detection"),
    limit: int = Query(200, description="Number of klines to analyze"),
    binance: BinanceService = Depends(get_binance_service)
):
    """Detect cryptocurrency-specific patterns"""
    try:
        import pandas as pd
        from src.services.crypto_pattern_detector import create_crypto_pattern_service, CryptoPatternConfig
        from src.data.models.pattern_models import Timeframe
        from src.data.storage.pattern_repository import PatternRepository
        
        async with binance:
            # Get kline data for pattern detection
            klines = await binance.get_klines(
                symbol=symbol.upper(),
                interval=timeframe,
                limit=min(limit, 500)
            )
            
            if not klines:
                raise HTTPException(status_code=404, detail="No data found for symbol")
            
            # Convert to DataFrame for pattern detection
            df_data = {
                'timestamp': [pd.to_datetime(k.open_time) for k in klines],
                'open': [k.open_price for k in klines],
                'high': [k.high_price for k in klines],
                'low': [k.low_price for k in klines],
                'close': [k.close_price for k in klines],
                'volume': [k.volume for k in klines]
            }
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            
            # Create crypto-optimized pattern service
            config = CryptoPatternConfig(
                min_volume_threshold=100000,  # Lower threshold for smaller coins
                pump_detection_threshold=0.15,  # 15% pump threshold
                dump_detection_threshold=-0.12  # 12% dump threshold
            )
            
            # Initialize pattern service (simplified for API endpoint)
            from src.services.pattern_detection_service import PatternDetectionService
            service = PatternDetectionService(repository=None)  # No repository for API calls
            
            # Register crypto-specific detectors
            from src.services.crypto_pattern_detector import (
                CryptoVolumePatternDetector, 
                CryptoPumpDumpDetector,
                CryptoSupportResistanceDetector
            )
            
            service.register_detector(CryptoVolumePatternDetector(config))
            service.register_detector(CryptoPumpDumpDetector(config))
            service.register_detector(CryptoSupportResistanceDetector(config))
            
            # Detect patterns
            timeframe_enum = Timeframe(timeframe)
            patterns = await service.scan_for_patterns(df, symbol.upper(), timeframe_enum)
            
            # Convert to API response format
            pattern_responses = []
            for pattern in patterns:
                pattern_responses.append({
                    "pattern_type": pattern.pattern_type.value,
                    "pattern_name": pattern.pattern_name,
                    "direction": pattern.direction.value,
                    "confidence": round(pattern.confidence, 3),
                    "timeframe": pattern.timeframe.value,
                    "start_time": pattern.start_time.isoformat(),
                    "end_time": pattern.end_time.isoformat() if pattern.end_time else None,
                    "symbol": pattern.symbol,
                    "entry_price": pattern.entry_price,
                    "target_price": pattern.target_price,
                    "stop_loss": pattern.stop_loss,
                    "strength": pattern.strength,
                    "metadata": pattern.metadata
                })
            
            return {
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "total_patterns": len(pattern_responses),
                "patterns": pattern_responses,
                "analysis_time": datetime.now(),
                "data_points_analyzed": len(df)
            }
            
    except Exception as e:
        logger.error(f"Error detecting patterns for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))