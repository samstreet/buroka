"""
Paginated data endpoints for large datasets.
"""

from fastapi import APIRouter, Query, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import math

from ..models.common import PaginatedResponse, PaginationParams
from ..models.market_data import MarketDataFilter, OHLCData
from ..auth.dependencies import get_optional_user, UserProfile

router = APIRouter(prefix="/api/v1/data", tags=["paginated-data"])


@router.get("/market-history", summary="Get Paginated Market History", response_model=PaginatedResponse)
async def get_paginated_market_history(
    symbol: str = Query(description="Stock symbol"),
    pagination: PaginationParams = Depends(),
    filters: MarketDataFilter = Depends(),
    current_user: Optional[UserProfile] = Depends(get_optional_user)
) -> PaginatedResponse:
    """
    Get paginated historical market data.
    
    Supports filtering, sorting, and pagination for large datasets.
    
    Args:
        symbol: Stock symbol to query
        pagination: Pagination parameters (page, page_size, sort_by, sort_order)
        filters: Filter parameters for data selection
        current_user: Optional authenticated user
        
    Returns:
        PaginatedResponse with market history data
    """
    # Mock data generation (in production, this would query actual database)
    total_records = 1000  # Mock total count
    
    # Calculate pagination
    total_pages = math.ceil(total_records / pagination.page_size)
    start_index = (pagination.page - 1) * pagination.page_size
    end_index = min(start_index + pagination.page_size, total_records)
    
    # Generate mock historical data
    mock_data = []
    for i in range(start_index, end_index):
        record = {
            "id": f"record_{i}",
            "symbol": symbol.upper(),
            "date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            "open": 100.0 + (i * 0.1),
            "high": 105.0 + (i * 0.1),
            "low": 95.0 + (i * 0.1),
            "close": 102.0 + (i * 0.1),
            "volume": 1000000 + (i * 1000),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Apply filters if specified
        if filters.min_price and record["close"] < filters.min_price:
            continue
        if filters.max_price and record["close"] > filters.max_price:
            continue
        if filters.min_volume and record["volume"] < filters.min_volume:
            continue
        if filters.max_volume and record["volume"] > filters.max_volume:
            continue
            
        mock_data.append(record)
    
    # Apply sorting
    if pagination.sort_by:
        reverse = pagination.sort_order == "desc"
        mock_data.sort(
            key=lambda x: x.get(pagination.sort_by, 0),
            reverse=reverse
        )
    
    # Create paginated response
    return PaginatedResponse(
        items=mock_data,
        total=total_records,
        page=pagination.page,
        page_size=pagination.page_size,
        total_pages=total_pages,
        has_next=pagination.page < total_pages,
        has_previous=pagination.page > 1
    )


@router.get("/symbols", summary="Get Paginated Symbol List", response_model=PaginatedResponse)
async def get_paginated_symbols(
    pagination: PaginationParams = Depends(),
    search: Optional[str] = Query(default=None, description="Search query"),
    exchange: Optional[str] = Query(default=None, description="Filter by exchange"),
    sector: Optional[str] = Query(default=None, description="Filter by sector"),
    current_user: Optional[UserProfile] = Depends(get_optional_user)
) -> PaginatedResponse:
    """
    Get paginated list of available symbols.
    
    Args:
        pagination: Pagination parameters
        search: Optional search query
        exchange: Filter by exchange (NYSE, NASDAQ, etc.)
        sector: Filter by sector
        current_user: Optional authenticated user
        
    Returns:
        PaginatedResponse with symbol information
    """
    # Mock symbol data
    all_symbols = [
        {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "sector": "Technology"},
        {"symbol": "MSFT", "name": "Microsoft Corp.", "exchange": "NASDAQ", "sector": "Technology"},
        {"symbol": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ", "sector": "Technology"},
        {"symbol": "AMZN", "name": "Amazon.com Inc.", "exchange": "NASDAQ", "sector": "Consumer Discretionary"},
        {"symbol": "TSLA", "name": "Tesla Inc.", "exchange": "NASDAQ", "sector": "Consumer Discretionary"},
        {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "exchange": "NYSE", "sector": "Financials"},
        {"symbol": "JNJ", "name": "Johnson & Johnson", "exchange": "NYSE", "sector": "Healthcare"},
        {"symbol": "V", "name": "Visa Inc.", "exchange": "NYSE", "sector": "Financials"},
        {"symbol": "PG", "name": "Procter & Gamble", "exchange": "NYSE", "sector": "Consumer Staples"},
        {"symbol": "UNH", "name": "UnitedHealth Group", "exchange": "NYSE", "sector": "Healthcare"},
    ]
    
    # Expand to simulate larger dataset
    expanded_symbols = []
    for i in range(100):  # Create 1000 mock symbols
        for base_symbol in all_symbols:
            expanded_symbols.append({
                **base_symbol,
                "symbol": f"{base_symbol['symbol']}{i}" if i > 0 else base_symbol['symbol'],
                "name": f"{base_symbol['name']} {i}" if i > 0 else base_symbol['name']
            })
    
    # Apply filters
    filtered_symbols = expanded_symbols
    
    if search:
        search_lower = search.lower()
        filtered_symbols = [
            sym for sym in filtered_symbols
            if search_lower in sym["symbol"].lower() or search_lower in sym["name"].lower()
        ]
    
    if exchange:
        filtered_symbols = [sym for sym in filtered_symbols if sym["exchange"] == exchange]
    
    if sector:
        filtered_symbols = [sym for sym in filtered_symbols if sym["sector"] == sector]
    
    # Apply sorting
    if pagination.sort_by:
        reverse = pagination.sort_order == "desc"
        filtered_symbols.sort(
            key=lambda x: x.get(pagination.sort_by, ""),
            reverse=reverse
        )
    
    # Apply pagination
    total_records = len(filtered_symbols)
    total_pages = math.ceil(total_records / pagination.page_size)
    start_index = (pagination.page - 1) * pagination.page_size
    end_index = start_index + pagination.page_size
    
    paginated_symbols = filtered_symbols[start_index:end_index]
    
    return PaginatedResponse(
        items=paginated_symbols,
        total=total_records,
        page=pagination.page,
        page_size=pagination.page_size,
        total_pages=total_pages,
        has_next=pagination.page < total_pages,
        has_previous=pagination.page > 1
    )


@router.get("/news", summary="Get Paginated News", response_model=PaginatedResponse)
async def get_paginated_news(
    pagination: PaginationParams = Depends(),
    symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
    start_date: Optional[str] = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(default=None, description="End date (YYYY-MM-DD)"),
    sentiment: Optional[str] = Query(default=None, description="Filter by sentiment (positive/negative/neutral)"),
    current_user: Optional[UserProfile] = Depends(get_optional_user)
) -> PaginatedResponse:
    """
    Get paginated news articles.
    
    Args:
        pagination: Pagination parameters
        symbol: Filter news by symbol
        start_date: Start date filter
        end_date: End date filter
        sentiment: Filter by sentiment
        current_user: Optional authenticated user
        
    Returns:
        PaginatedResponse with news articles
    """
    # Mock news data
    base_news = [
        {
            "id": "news_1",
            "title": "Company Reports Strong Q4 Earnings",
            "summary": "Revenue exceeded expectations by 15%",
            "source": "Reuters",
            "published_at": "2024-01-15T10:00:00Z",
            "symbols": ["AAPL", "MSFT"],
            "sentiment": "positive",
            "sentiment_score": 0.8
        },
        {
            "id": "news_2", 
            "title": "Market Volatility Continues",
            "summary": "Tech stocks show mixed performance",
            "source": "Bloomberg",
            "published_at": "2024-01-14T15:30:00Z",
            "symbols": ["GOOGL", "AMZN"],
            "sentiment": "neutral",
            "sentiment_score": 0.0
        },
        {
            "id": "news_3",
            "title": "Regulatory Concerns Impact Stock Price",
            "summary": "New regulations may affect profitability",
            "source": "WSJ",
            "published_at": "2024-01-13T09:15:00Z",
            "symbols": ["TSLA"],
            "sentiment": "negative",
            "sentiment_score": -0.6
        }
    ]
    
    # Expand news dataset
    expanded_news = []
    for i in range(200):  # Create larger dataset
        for base_article in base_news:
            expanded_news.append({
                **base_article,
                "id": f"{base_article['id']}_{i}",
                "title": f"{base_article['title']} (Update {i})" if i > 0 else base_article['title']
            })
    
    # Apply filters
    filtered_news = expanded_news
    
    if symbol:
        symbol_upper = symbol.upper()
        filtered_news = [
            article for article in filtered_news
            if symbol_upper in article.get("symbols", [])
        ]
    
    if sentiment:
        filtered_news = [
            article for article in filtered_news
            if article.get("sentiment") == sentiment
        ]
    
    # Apply sorting (default by published date, newest first)
    sort_key = pagination.sort_by or "published_at"
    reverse = pagination.sort_order == "desc" if pagination.sort_order else True
    
    filtered_news.sort(
        key=lambda x: x.get(sort_key, ""),
        reverse=reverse
    )
    
    # Apply pagination
    total_records = len(filtered_news)
    total_pages = math.ceil(total_records / pagination.page_size)
    start_index = (pagination.page - 1) * pagination.page_size
    end_index = start_index + pagination.page_size
    
    paginated_news = filtered_news[start_index:end_index]
    
    return PaginatedResponse(
        items=paginated_news,
        total=total_records,
        page=pagination.page,
        page_size=pagination.page_size,
        total_pages=total_pages,
        has_next=pagination.page < total_pages,
        has_previous=pagination.page > 1
    )


@router.get("/trades", summary="Get Paginated Trade History", response_model=PaginatedResponse)
async def get_paginated_trades(
    pagination: PaginationParams = Depends(),
    symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
    min_volume: Optional[int] = Query(default=None, description="Minimum trade volume"),
    trade_type: Optional[str] = Query(default=None, description="Trade type (buy/sell)"),
    current_user: Optional[UserProfile] = Depends(get_optional_user)
) -> PaginatedResponse:
    """
    Get paginated trade history.
    
    Args:
        pagination: Pagination parameters
        symbol: Filter by symbol
        min_volume: Minimum trade volume filter
        trade_type: Filter by trade type
        current_user: Optional authenticated user
        
    Returns:
        PaginatedResponse with trade history
    """
    # Mock trade data
    mock_trades = []
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    trade_types = ["buy", "sell"]
    
    # Generate mock trades
    for i in range(500):  # 500 mock trades
        trade = {
            "id": f"trade_{i}",
            "symbol": symbols[i % len(symbols)],
            "trade_type": trade_types[i % len(trade_types)],
            "price": 100.0 + (i * 0.5),
            "volume": 1000 + (i * 100),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trade_id": f"T{i:06d}"
        }
        mock_trades.append(trade)
    
    # Apply filters
    filtered_trades = mock_trades
    
    if symbol:
        symbol_upper = symbol.upper()
        filtered_trades = [t for t in filtered_trades if t["symbol"] == symbol_upper]
    
    if min_volume:
        filtered_trades = [t for t in filtered_trades if t["volume"] >= min_volume]
    
    if trade_type:
        filtered_trades = [t for t in filtered_trades if t["trade_type"] == trade_type]
    
    # Apply sorting (default by timestamp, newest first)
    sort_key = pagination.sort_by or "timestamp"
    reverse = pagination.sort_order == "desc" if pagination.sort_order else True
    
    filtered_trades.sort(
        key=lambda x: x.get(sort_key, ""),
        reverse=reverse
    )
    
    # Apply pagination
    total_records = len(filtered_trades)
    total_pages = math.ceil(total_records / pagination.page_size)
    start_index = (pagination.page - 1) * pagination.page_size
    end_index = start_index + pagination.page_size
    
    paginated_trades = filtered_trades[start_index:end_index]
    
    return PaginatedResponse(
        items=paginated_trades,
        total=total_records,
        page=pagination.page,
        page_size=pagination.page_size,
        total_pages=total_pages,
        has_next=pagination.page < total_pages,
        has_previous=pagination.page > 1
    )