"""
Core market data API endpoints with caching and pagination.
"""

from fastapi import APIRouter, HTTPException, status, Depends, Query, BackgroundTasks
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone, timedelta
import asyncio
import time
import hashlib
import json

from ..models.market_data import (
    SymbolRequest, MarketDataResponse, BatchMarketDataRequest, BatchMarketDataResponse,
    QuoteData, OHLCData, NewsData, MarketDataFilter, SymbolSearchRequest, SymbolInfo,
    MarketDataType, DataGranularity
)
from ..models.common import PaginatedResponse, PaginationParams
from ..auth.dependencies import get_optional_user, UserProfile

router = APIRouter(prefix="/api/v1/market-data", tags=["market-data"])

# Global cache and services
_cache_store: Dict[str, Dict[str, Any]] = {}
_ingestion_service = None
_storage_service = None


def get_services():
    """Get or initialize data services."""
    global _ingestion_service, _storage_service
    
    if _ingestion_service is None:
        try:
            from ...data.ingestion.client_factory import get_default_client
            from ...data.ingestion.transformers import MarketDataTransformer
            from ...data.ingestion.metrics import InMemoryMetricsCollector
            from ...data.ingestion.service import DataIngestionService
            from ...data.storage.sinks import InMemoryDataSink
            from ...data.storage.service import DataStorageService
            
            # Initialize ingestion service
            data_source = get_default_client()
            transformer = MarketDataTransformer()
            sink = InMemoryDataSink()
            metrics = InMemoryMetricsCollector()
            
            _ingestion_service = DataIngestionService(
                data_source=data_source,
                data_transformer=transformer,
                data_sink=sink,
                metrics_collector=metrics
            )
            
            # Initialize storage service
            _storage_service = DataStorageService(
                primary_sink=sink,
                batch_size=100,
                batch_timeout=10
            )
            
        except Exception as e:
            # Fallback to None if services can't be initialized
            _ingestion_service = None
            _storage_service = None
    
    return _ingestion_service, _storage_service


class CacheManager:
    """Simple in-memory cache manager."""
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes default
        self.default_ttl = default_ttl
    
    def _generate_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters."""
        # Sort parameters for consistent keys
        param_str = json.dumps(kwargs, sort_keys=True, default=str)
        key_hash = hashlib.md5(param_str.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache."""
        if key in _cache_store:
            entry = _cache_store[key]
            if time.time() < entry["expires_at"]:
                return entry["data"]
            else:
                # Expired, remove from cache
                del _cache_store[key]
        return None
    
    def set(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        ttl = ttl or self.default_ttl
        _cache_store[key] = {
            "data": data,
            "expires_at": time.time() + ttl,
            "created_at": time.time()
        }
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in _cache_store:
            del _cache_store[key]
            return True
        return False
    
    def clear(self) -> int:
        """Clear all cache entries."""
        count = len(_cache_store)
        _cache_store.clear()
        return count
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        active_entries = sum(1 for entry in _cache_store.values() if now < entry["expires_at"])
        expired_entries = len(_cache_store) - active_entries
        
        return {
            "total_entries": len(_cache_store),
            "active_entries": active_entries,
            "expired_entries": expired_entries,
            "hit_rate": "Not implemented",  # Would need hit/miss tracking
            "memory_usage": "Not implemented"
        }


# Initialize cache manager
cache = CacheManager(default_ttl=300)


@router.get("/{symbol}/quote", summary="Get Real-time Quote", response_model=MarketDataResponse)
async def get_symbol_quote(
    symbol: str,
    use_cache: bool = Query(default=True, description="Use cached data if available"),
    current_user: Optional[UserProfile] = Depends(get_optional_user)
) -> MarketDataResponse:
    """
    Get real-time quote data for a symbol.
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT)
        use_cache: Whether to use cached data if available
        current_user: Optional authenticated user
        
    Returns:
        MarketDataResponse with real-time quote data
        
    Raises:
        HTTPException: If symbol is invalid or data unavailable
    """
    symbol = symbol.upper().strip()
    
    # Validate symbol format
    if not symbol or len(symbol) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid symbol format"
        )
    
    # Check cache first
    cache_key = cache._generate_key("quote", symbol=symbol)
    
    if use_cache:
        cached_data = cache.get(cache_key)
        if cached_data:
            return MarketDataResponse(**cached_data)
    
    # Get data from ingestion service
    ingestion_service, _ = get_services()
    if not ingestion_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data ingestion service unavailable"
        )
    
    try:
        result = await ingestion_service.ingest_symbol_data(symbol, MarketDataType.QUOTE)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.get("error", "Symbol data not found")
            )
        
        # Create response
        response_data = {
            "symbol": symbol,
            "data_type": MarketDataType.QUOTE,
            "timestamp": datetime.now(timezone.utc),
            "data": {
                "quote": result.get("records", [{}])[0] if result.get("records") else {}
            },
            "success": True,
            "metadata": {
                "source": "real-time",
                "cache_used": False,
                "user_id": current_user.user_id if current_user else None
            }
        }
        
        # Cache the response
        cache.set(cache_key, response_data, ttl=60)  # Cache quotes for 1 minute
        
        return MarketDataResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch quote data: {str(e)}"
        )


@router.get("/{symbol}/daily", summary="Get Daily OHLC Data", response_model=MarketDataResponse)
async def get_symbol_daily(
    symbol: str,
    start_date: Optional[str] = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(default=None, description="End date (YYYY-MM-DD)"),
    outputsize: str = Query(default="compact", description="Data size (compact/full)"),
    use_cache: bool = Query(default=True, description="Use cached data if available"),
    current_user: Optional[UserProfile] = Depends(get_optional_user)
) -> MarketDataResponse:
    """
    Get daily OHLC data for a symbol.
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT)
        start_date: Start date for data range (YYYY-MM-DD)
        end_date: End date for data range (YYYY-MM-DD)
        outputsize: Size of data (compact/full)
        use_cache: Whether to use cached data if available
        current_user: Optional authenticated user
        
    Returns:
        MarketDataResponse with daily OHLC data
        
    Raises:
        HTTPException: If parameters are invalid or data unavailable
    """
    symbol = symbol.upper().strip()
    
    # Validate symbol
    if not symbol or len(symbol) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid symbol format"
        )
    
    # Validate date parameters
    if start_date and not _validate_date_format(start_date):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid start_date format. Use YYYY-MM-DD"
        )
    
    if end_date and not _validate_date_format(end_date):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid end_date format. Use YYYY-MM-DD"
        )
    
    # Check cache
    cache_key = cache._generate_key(
        "daily", 
        symbol=symbol, 
        start_date=start_date, 
        end_date=end_date, 
        outputsize=outputsize
    )
    
    if use_cache:
        cached_data = cache.get(cache_key)
        if cached_data:
            return MarketDataResponse(**cached_data)
    
    # Get data from ingestion service
    ingestion_service, _ = get_services()
    if not ingestion_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data ingestion service unavailable"
        )
    
    try:
        result = await ingestion_service.ingest_symbol_data(symbol, MarketDataType.DAILY)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.get("error", "Symbol data not found")
            )
        
        # Filter data by date range if specified
        records = result.get("records", [])
        if start_date or end_date:
            records = _filter_records_by_date(records, start_date, end_date)
        
        response_data = {
            "symbol": symbol,
            "data_type": MarketDataType.DAILY,
            "timestamp": datetime.now(timezone.utc),
            "data": {
                "ohlc_data": records
            },
            "success": True,
            "metadata": {
                "source": "daily",
                "cache_used": False,
                "record_count": len(records),
                "date_range": {
                    "start": start_date,
                    "end": end_date
                },
                "user_id": current_user.user_id if current_user else None
            }
        }
        
        # Cache daily data for longer (30 minutes)
        cache.set(cache_key, response_data, ttl=1800)
        
        return MarketDataResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch daily data: {str(e)}"
        )


@router.get("/{symbol}/intraday", summary="Get Intraday Data", response_model=MarketDataResponse)
async def get_symbol_intraday(
    symbol: str,
    interval: DataGranularity = Query(default=DataGranularity.MINUTE_5, description="Data granularity"),
    start_date: Optional[str] = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(default=None, description="End date (YYYY-MM-DD)"),
    use_cache: bool = Query(default=True, description="Use cached data if available"),
    current_user: Optional[UserProfile] = Depends(get_optional_user)
) -> MarketDataResponse:
    """
    Get intraday OHLC data for a symbol.
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT)
        interval: Data granularity (1min, 5min, 15min, 30min, 1hour)
        start_date: Start date for data range (YYYY-MM-DD)
        end_date: End date for data range (YYYY-MM-DD)
        use_cache: Whether to use cached data if available
        current_user: Optional authenticated user
        
    Returns:
        MarketDataResponse with intraday OHLC data
        
    Raises:
        HTTPException: If parameters are invalid or data unavailable
    """
    symbol = symbol.upper().strip()
    
    # Validate symbol
    if not symbol or len(symbol) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid symbol format"
        )
    
    # Check cache
    cache_key = cache._generate_key(
        "intraday",
        symbol=symbol,
        interval=interval.value,
        start_date=start_date,
        end_date=end_date
    )
    
    if use_cache:
        cached_data = cache.get(cache_key)
        if cached_data:
            return MarketDataResponse(**cached_data)
    
    # Get data from ingestion service
    ingestion_service, _ = get_services()
    if not ingestion_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data ingestion service unavailable"
        )
    
    try:
        result = await ingestion_service.ingest_symbol_data(
            symbol, 
            MarketDataType.INTRADAY, 
            granularity=interval
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.get("error", "Symbol data not found")
            )
        
        records = result.get("records", [])
        if start_date or end_date:
            records = _filter_records_by_date(records, start_date, end_date)
        
        response_data = {
            "symbol": symbol,
            "data_type": MarketDataType.INTRADAY,
            "timestamp": datetime.now(timezone.utc),
            "data": {
                "ohlc_data": records
            },
            "success": True,
            "metadata": {
                "source": "intraday",
                "cache_used": False,
                "record_count": len(records),
                "granularity": interval.value,
                "user_id": current_user.user_id if current_user else None
            }
        }
        
        # Cache intraday data for 5 minutes
        cache.set(cache_key, response_data, ttl=300)
        
        return MarketDataResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch intraday data: {str(e)}"
        )


@router.post("/batch", summary="Get Batch Market Data", response_model=BatchMarketDataResponse)
async def get_batch_market_data(
    request: BatchMarketDataRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[UserProfile] = Depends(get_optional_user)
) -> BatchMarketDataResponse:
    """
    Get market data for multiple symbols in batch.
    
    Args:
        request: Batch request with symbols and data type
        background_tasks: Background tasks for async processing
        current_user: Optional authenticated user
        
    Returns:
        BatchMarketDataResponse with results for all symbols
        
    Raises:
        HTTPException: If request parameters are invalid
    """
    if not request.symbols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one symbol is required"
        )
    
    if len(request.symbols) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 50 symbols per batch request"
        )
    
    ingestion_service, storage_service = get_services()
    if not ingestion_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data ingestion service unavailable"
        )
    
    try:
        # Process batch request
        result = await ingestion_service.ingest_batch_data(request.symbols, request.data_type)
        
        # Store results in background if storage service is available
        if storage_service and result.get("success", False):
            background_tasks.add_task(
                _store_batch_results,
                storage_service,
                result,
                current_user.user_id if current_user else None
            )
        
        # Format response
        batch_response = BatchMarketDataResponse(
            total_symbols=len(request.symbols),
            successful=result.get("successful", 0),
            failed=result.get("total_symbols", len(request.symbols)) - result.get("successful", 0),
            results=result.get("results", [])
        )
        
        return batch_response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch request failed: {str(e)}"
        )


@router.get("/search", summary="Search Symbols", response_model=List[SymbolInfo])
async def search_symbols(
    query: str = Query(description="Search query"),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum results"),
    include_inactive: bool = Query(default=False, description="Include inactive symbols"),
    current_user: Optional[UserProfile] = Depends(get_optional_user)
) -> List[SymbolInfo]:
    """
    Search for stock symbols matching query.
    
    Args:
        query: Search query (company name or symbol)
        limit: Maximum number of results
        include_inactive: Whether to include inactive symbols
        current_user: Optional authenticated user
        
    Returns:
        List of matching symbol information
        
    Raises:
        HTTPException: If search fails
    """
    if len(query.strip()) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must be at least 2 characters long"
        )
    
    # Check cache
    cache_key = cache._generate_key(
        "search",
        query=query.lower().strip(),
        limit=limit,
        include_inactive=include_inactive
    )
    
    cached_data = cache.get(cache_key)
    if cached_data:
        return [SymbolInfo(**item) for item in cached_data]
    
    try:
        # Mock symbol search results (in production, this would query a symbols database)
        mock_results = _generate_mock_search_results(query, limit, include_inactive)
        
        # Cache search results for 1 hour
        cache.set(cache_key, [result.model_dump() for result in mock_results], ttl=3600)
        
        return mock_results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Symbol search failed: {str(e)}"
        )


@router.get("/cache/stats", summary="Get Cache Statistics")
async def get_cache_stats(
    current_user: Optional[UserProfile] = Depends(get_optional_user)
) -> Dict[str, Any]:
    """
    Get cache statistics and performance metrics.
    
    Returns:
        Dict with cache statistics
    """
    stats = cache.stats()
    
    return {
        "cache_stats": stats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cache_enabled": True,
        "default_ttl": cache.default_ttl
    }


@router.delete("/cache/clear", summary="Clear Cache")
async def clear_cache(
    current_user: Optional[UserProfile] = Depends(get_optional_user)
) -> Dict[str, Any]:
    """
    Clear all cached data.
    
    Returns:
        Dict with clear operation result
    """
    cleared_count = cache.clear()
    
    return {
        "success": True,
        "cleared_entries": cleared_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": f"Cleared {cleared_count} cache entries"
    }


# Helper functions
def _validate_date_format(date_str: str) -> bool:
    """Validate date format (YYYY-MM-DD)."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _filter_records_by_date(records: List[Dict[str, Any]], start_date: Optional[str], end_date: Optional[str]) -> List[Dict[str, Any]]:
    """Filter records by date range."""
    if not records or (not start_date and not end_date):
        return records
    
    filtered_records = []
    
    for record in records:
        record_date_str = record.get("timestamp")
        if not record_date_str:
            continue
        
        try:
            # Parse record date
            if isinstance(record_date_str, str):
                if "T" in record_date_str:
                    record_date = datetime.fromisoformat(record_date_str.replace("Z", "+00:00"))
                else:
                    record_date = datetime.strptime(record_date_str, "%Y-%m-%d")
            else:
                record_date = record_date_str
            
            record_date = record_date.date()
            
            # Check date range
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
                if record_date < start_dt:
                    continue
            
            if end_date:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
                if record_date > end_dt:
                    continue
            
            filtered_records.append(record)
            
        except (ValueError, AttributeError):
            # Skip records with invalid dates
            continue
    
    return filtered_records


async def _store_batch_results(
    storage_service,
    result: Dict[str, Any],
    user_id: Optional[str]
) -> None:
    """Store batch results in background."""
    try:
        if result.get("success") and result.get("results"):
            for item in result["results"]:
                if item.get("success"):
                    await storage_service.store_data(
                        item,
                        metadata={"user_id": user_id, "source": "batch_api"}
                    )
    except Exception as e:
        # Log error but don't fail the API request
        import logging
        logging.getLogger(__name__).error(f"Failed to store batch results: {e}")


def _generate_mock_search_results(query: str, limit: int, include_inactive: bool) -> List[SymbolInfo]:
    """Generate mock search results."""
    # In production, this would query a real symbols database
    mock_symbols = [
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "type": "Common Stock",
            "region": "United States",
            "market_open": "09:30",
            "market_close": "16:00",
            "timezone": "US/Eastern",
            "currency": "USD"
        },
        {
            "symbol": "MSFT", 
            "name": "Microsoft Corporation",
            "type": "Common Stock",
            "region": "United States",
            "market_open": "09:30",
            "market_close": "16:00",
            "timezone": "US/Eastern",
            "currency": "USD"
        },
        {
            "symbol": "GOOGL",
            "name": "Alphabet Inc. Class A",
            "type": "Common Stock", 
            "region": "United States",
            "market_open": "09:30",
            "market_close": "16:00",
            "timezone": "US/Eastern",
            "currency": "USD"
        }
    ]
    
    # Filter by query
    query_lower = query.lower()
    matching_symbols = []
    
    for symbol_data in mock_symbols:
        if (query_lower in symbol_data["symbol"].lower() or 
            query_lower in symbol_data["name"].lower()):
            
            # Calculate match score
            symbol_match = 1.0 if query_lower == symbol_data["symbol"].lower() else 0.8
            name_match = 0.6 if query_lower in symbol_data["name"].lower() else 0.0
            match_score = max(symbol_match, name_match)
            
            symbol_info = SymbolInfo(**symbol_data, match_score=match_score)
            matching_symbols.append(symbol_info)
    
    # Sort by match score
    matching_symbols.sort(key=lambda x: x.match_score, reverse=True)
    
    return matching_symbols[:limit]