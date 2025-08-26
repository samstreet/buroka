"""
Alpha Vantage API client implementation.
"""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from decimal import Decimal

from .base_client import BaseAPIClient, RateLimitBucket
from ..models.market_data import (
    MarketDataResponse, OHLCData, QuoteData, MarketDataType, 
    DataGranularity, APIError, DataValidationError
)


class AlphaVantageClient(BaseAPIClient):
    """Alpha Vantage API client with rate limiting and error handling."""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    # Alpha Vantage interval mapping
    INTERVAL_MAPPING = {
        DataGranularity.MINUTE_1: "1min",
        DataGranularity.MINUTE_5: "5min", 
        DataGranularity.MINUTE_15: "15min",
        DataGranularity.MINUTE_30: "30min",
        DataGranularity.HOUR_1: "60min",
        DataGranularity.DAILY: "daily",
        DataGranularity.WEEKLY: "weekly",
        DataGranularity.MONTHLY: "monthly"
    }
    
    def __init__(self, api_key: str, **kwargs):
        # Alpha Vantage rate limits: 5 API requests per minute, 500 per day for free tier
        rate_limits = {
            "default": RateLimitBucket(max_calls=5, time_window=60),  # 5 calls per minute
            "daily_limit": RateLimitBucket(max_calls=500, time_window=86400)  # 500 calls per day
        }
        
        super().__init__(
            api_key=api_key,
            base_url=self.BASE_URL,
            rate_limits=rate_limits,
            **kwargs
        )
    
    async def get_intraday_data(
        self, 
        symbol: str, 
        interval: str,
        adjusted: bool = True,
        outputsize: str = "compact",
        datatype: str = "json"
    ) -> MarketDataResponse:
        """Get intraday market data for a symbol."""
        if not self.validate_symbol(symbol):
            raise DataValidationError(f"Invalid symbol format: {symbol}")
        
        # Map interval to Alpha Vantage format
        if interval in self.INTERVAL_MAPPING.values():
            av_interval = interval
        else:
            # Try to find matching granularity
            granularity = None
            for gran, av_int in self.INTERVAL_MAPPING.items():
                if gran == interval or gran.value == interval:
                    av_interval = av_int
                    granularity = gran
                    break
            else:
                raise DataValidationError(f"Unsupported interval: {interval}")
        
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol.upper(),
            "interval": av_interval,
            "adjusted": "true" if adjusted else "false",
            "outputsize": outputsize,
            "datatype": datatype
        }
        
        try:
            raw_data = await self._make_request("", params)
            return self._parse_time_series_response(
                raw_data, symbol, MarketDataType.INTRADAY, 
                self._get_granularity_from_interval(av_interval)
            )
        except Exception as e:
            self.logger.error(f"Failed to get intraday data for {symbol}: {e}")
            return MarketDataResponse(
                symbol=symbol,
                data_type=MarketDataType.INTRADAY,
                timestamp=datetime.now(timezone.utc),
                data={},
                success=False,
                error_message=str(e)
            )
    
    async def get_daily_data(
        self, 
        symbol: str, 
        outputsize: str = "compact",
        datatype: str = "json"
    ) -> MarketDataResponse:
        """Get daily market data for a symbol."""
        if not self.validate_symbol(symbol):
            raise DataValidationError(f"Invalid symbol format: {symbol}")
        
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol.upper(),
            "outputsize": outputsize,
            "datatype": datatype
        }
        
        try:
            raw_data = await self._make_request("", params)
            return self._parse_time_series_response(
                raw_data, symbol, MarketDataType.DAILY, DataGranularity.DAILY
            )
        except Exception as e:
            self.logger.error(f"Failed to get daily data for {symbol}: {e}")
            return MarketDataResponse(
                symbol=symbol,
                data_type=MarketDataType.DAILY,
                timestamp=datetime.now(timezone.utc),
                data={},
                success=False,
                error_message=str(e)
            )
    
    async def get_weekly_data(
        self, 
        symbol: str,
        datatype: str = "json"
    ) -> MarketDataResponse:
        """Get weekly market data for a symbol."""
        if not self.validate_symbol(symbol):
            raise DataValidationError(f"Invalid symbol format: {symbol}")
        
        params = {
            "function": "TIME_SERIES_WEEKLY_ADJUSTED",
            "symbol": symbol.upper(),
            "datatype": datatype
        }
        
        try:
            raw_data = await self._make_request("", params)
            return self._parse_time_series_response(
                raw_data, symbol, MarketDataType.WEEKLY, DataGranularity.WEEKLY
            )
        except Exception as e:
            self.logger.error(f"Failed to get weekly data for {symbol}: {e}")
            return MarketDataResponse(
                symbol=symbol,
                data_type=MarketDataType.WEEKLY,
                timestamp=datetime.now(timezone.utc),
                data={},
                success=False,
                error_message=str(e)
            )
    
    async def get_monthly_data(
        self, 
        symbol: str,
        datatype: str = "json"
    ) -> MarketDataResponse:
        """Get monthly market data for a symbol."""
        if not self.validate_symbol(symbol):
            raise DataValidationError(f"Invalid symbol format: {symbol}")
        
        params = {
            "function": "TIME_SERIES_MONTHLY_ADJUSTED",
            "symbol": symbol.upper(),
            "datatype": datatype
        }
        
        try:
            raw_data = await self._make_request("", params)
            return self._parse_time_series_response(
                raw_data, symbol, MarketDataType.MONTHLY, DataGranularity.MONTHLY
            )
        except Exception as e:
            self.logger.error(f"Failed to get monthly data for {symbol}: {e}")
            return MarketDataResponse(
                symbol=symbol,
                data_type=MarketDataType.MONTHLY,
                timestamp=datetime.now(timezone.utc),
                data={},
                success=False,
                error_message=str(e)
            )
    
    async def get_quote(self, symbol: str) -> MarketDataResponse:
        """Get real-time quote for a symbol."""
        if not self.validate_symbol(symbol):
            raise DataValidationError(f"Invalid symbol format: {symbol}")
        
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol.upper()
        }
        
        try:
            raw_data = await self._make_request("", params)
            return self._parse_quote_response(raw_data, symbol)
        except Exception as e:
            self.logger.error(f"Failed to get quote for {symbol}: {e}")
            return MarketDataResponse(
                symbol=symbol,
                data_type=MarketDataType.QUOTE,
                timestamp=datetime.now(timezone.utc),
                data={},
                success=False,
                error_message=str(e)
            )
    
    async def search_symbols(self, keywords: str) -> MarketDataResponse:
        """Search for symbols matching keywords."""
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": keywords
        }
        
        try:
            raw_data = await self._make_request("", params)
            return MarketDataResponse(
                symbol=keywords,
                data_type=MarketDataType.QUOTE,  # Using QUOTE as generic type
                timestamp=datetime.now(timezone.utc),
                data=raw_data,
                success=True
            )
        except Exception as e:
            self.logger.error(f"Failed to search symbols for '{keywords}': {e}")
            return MarketDataResponse(
                symbol=keywords,
                data_type=MarketDataType.QUOTE,
                timestamp=datetime.now(timezone.utc),
                data={},
                success=False,
                error_message=str(e)
            )
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol format for Alpha Vantage."""
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Alpha Vantage supports various formats: AAPL, MSFT, TSM etc.
        # Must start with a letter, then can have letters/numbers/dots/hyphens
        pattern = r'^[A-Za-z][A-Za-z0-9.-]{0,9}$'
        return bool(re.match(pattern, symbol.strip()))
    
    def _parse_time_series_response(
        self, 
        raw_data: Dict[str, Any], 
        symbol: str,
        data_type: MarketDataType,
        granularity: DataGranularity
    ) -> MarketDataResponse:
        """Parse Alpha Vantage time series response."""
        if "Error Message" in raw_data:
            raise APIError(f"API Error: {raw_data['Error Message']}")
        
        if "Note" in raw_data:
            raise APIError(f"API Note (likely rate limit): {raw_data['Note']}")
        
        # Find the time series data key
        time_series_key = None
        for key in raw_data.keys():
            if "Time Series" in key:
                time_series_key = key
                break
        
        if not time_series_key:
            raise APIError("No time series data found in response")
        
        time_series_data = raw_data[time_series_key]
        metadata = raw_data.get("Meta Data", {})
        
        # Convert to standardized format
        ohlc_data = []
        for timestamp_str, data_point in time_series_data.items():
            try:
                # Parse timestamp
                if granularity in [DataGranularity.DAILY, DataGranularity.WEEKLY, DataGranularity.MONTHLY]:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                else:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                
                # Extract OHLCV data with proper key mapping
                open_key = next((k for k in data_point.keys() if "open" in k.lower()), None)
                high_key = next((k for k in data_point.keys() if "high" in k.lower()), None)
                low_key = next((k for k in data_point.keys() if "low" in k.lower()), None)
                close_key = next((k for k in data_point.keys() if "close" in k.lower()), None)
                volume_key = next((k for k in data_point.keys() if "volume" in k.lower()), None)
                adj_close_key = next((k for k in data_point.keys() if "adjusted" in k.lower()), None)
                
                if not all([open_key, high_key, low_key, close_key, volume_key]):
                    self.logger.warning(f"Missing OHLCV data for {timestamp_str}")
                    continue
                
                ohlc = OHLCData(
                    symbol=symbol,
                    timestamp=timestamp,
                    open_price=Decimal(data_point[open_key]),
                    high_price=Decimal(data_point[high_key]),
                    low_price=Decimal(data_point[low_key]),
                    close_price=Decimal(data_point[close_key]),
                    volume=int(data_point[volume_key]),
                    adjusted_close=Decimal(data_point[adj_close_key]) if adj_close_key else None,
                    granularity=granularity,
                    data_type=data_type
                )
                ohlc_data.append(ohlc.model_dump())
                
            except (ValueError, KeyError, TypeError) as e:
                self.logger.warning(f"Failed to parse data point for {timestamp_str}: {e}")
                continue
        
        return MarketDataResponse(
            symbol=symbol,
            data_type=data_type,
            timestamp=datetime.now(timezone.utc),
            data={"ohlc_data": ohlc_data, "raw_response": raw_data},
            metadata=metadata,
            success=True
        )
    
    def _parse_quote_response(self, raw_data: Dict[str, Any], symbol: str) -> MarketDataResponse:
        """Parse Alpha Vantage quote response."""
        if "Error Message" in raw_data:
            raise APIError(f"API Error: {raw_data['Error Message']}")
        
        if "Note" in raw_data:
            raise APIError(f"API Note (likely rate limit): {raw_data['Note']}")
        
        global_quote = raw_data.get("Global Quote", {})
        if not global_quote:
            raise APIError("No quote data found in response")
        
        # Parse quote data
        try:
            last_price = Decimal(global_quote.get("05. price", "0"))
            quote = QuoteData(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                bid_price=last_price,  # Use last price as approximation since AV doesn't provide bid/ask
                ask_price=last_price + Decimal("0.01"),  # Small spread approximation
                bid_size=0,
                ask_size=0,
                last_price=last_price,
                last_size=int(global_quote.get("06. volume", "0")),
                change=Decimal(global_quote.get("09. change", "0")),
                change_percent=Decimal(global_quote.get("10. change percent", "0%").rstrip('%'))
            )
            
            return MarketDataResponse(
                symbol=symbol,
                data_type=MarketDataType.QUOTE,
                timestamp=datetime.now(timezone.utc),
                data={"quote": quote.model_dump(), "raw_response": raw_data},
                success=True
            )
            
        except (ValueError, KeyError) as e:
            raise APIError(f"Failed to parse quote data: {e}")
    
    def _get_granularity_from_interval(self, interval: str) -> DataGranularity:
        """Get DataGranularity enum from interval string."""
        interval_map = {v: k for k, v in self.INTERVAL_MAPPING.items()}
        return interval_map.get(interval, DataGranularity.DAILY)
    
    def _prepare_params(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Prepare request parameters for Alpha Vantage."""
        prepared_params = {"apikey": self.api_key}
        prepared_params.update({k: str(v) for k, v in params.items() if v is not None})
        return prepared_params