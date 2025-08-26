"""
Tests for Alpha Vantage API client.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from decimal import Decimal

from src.data.ingestion.alpha_vantage_client import AlphaVantageClient
from src.data.models.market_data import MarketDataType, DataGranularity, APIError, DataValidationError


class TestAlphaVantageClient:
    """Test AlphaVantageClient class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.api_key = "test_api_key"
        self.client = AlphaVantageClient(api_key=self.api_key)
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, 'client'):
            asyncio.run(self.client.close())
    
    def test_client_initialization(self):
        """Test client initialization."""
        assert self.client.api_key == self.api_key
        assert self.client.base_url == AlphaVantageClient.BASE_URL
        assert "default" in self.client.rate_limits
        assert "daily_limit" in self.client.rate_limits
    
    def test_validate_symbol_valid(self):
        """Test valid symbol validation."""
        valid_symbols = ["AAPL", "MSFT", "BRK.A", "TSM", "VOO"]
        
        for symbol in valid_symbols:
            assert self.client.validate_symbol(symbol), f"Symbol {symbol} should be valid"
    
    def test_validate_symbol_invalid(self):
        """Test invalid symbol validation."""
        invalid_symbols = ["", None, "TOOLONGABC123", "AAPL$", "123", "AAPL@"]
        
        for symbol in invalid_symbols:
            assert not self.client.validate_symbol(symbol), f"Symbol {symbol} should be invalid"
    
    def test_interval_mapping(self):
        """Test interval mapping."""
        assert AlphaVantageClient.INTERVAL_MAPPING[DataGranularity.MINUTE_1] == "1min"
        assert AlphaVantageClient.INTERVAL_MAPPING[DataGranularity.MINUTE_5] == "5min"
        assert AlphaVantageClient.INTERVAL_MAPPING[DataGranularity.HOUR_1] == "60min"
        assert AlphaVantageClient.INTERVAL_MAPPING[DataGranularity.DAILY] == "daily"
    
    @pytest.mark.asyncio
    async def test_get_intraday_data_success(self):
        """Test successful intraday data retrieval."""
        mock_response = {
            "Meta Data": {
                "1. Information": "Intraday (5min) prices and volumes",
                "2. Symbol": "AAPL",
                "3. Last Refreshed": "2023-01-15 16:00:00",
                "4. Interval": "5min"
            },
            "Time Series (5min)": {
                "2023-01-15 16:00:00": {
                    "1. open": "150.00",
                    "2. high": "152.00",
                    "3. low": "149.00",
                    "4. close": "151.00",
                    "5. volume": "1000000"
                }
            }
        }
        
        with patch.object(self.client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            response = await self.client.get_intraday_data("AAPL", "5min")
            
            assert response.success
            assert response.symbol == "AAPL"
            assert response.data_type == MarketDataType.INTRADAY
            assert "ohlc_data" in response.data
            assert len(response.data["ohlc_data"]) == 1
            
            ohlc_item = response.data["ohlc_data"][0]
            assert ohlc_item["symbol"] == "AAPL"
            assert ohlc_item["open_price"] == "150.00"
            assert ohlc_item["volume"] == 1000000
    
    @pytest.mark.asyncio
    async def test_get_intraday_data_api_error(self):
        """Test intraday data retrieval with API error."""
        mock_response = {
            "Error Message": "Invalid API call"
        }
        
        with patch.object(self.client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            response = await self.client.get_intraday_data("INVALID", "5min")
            
            assert not response.success
            assert "API Error" in response.error_message
    
    @pytest.mark.asyncio
    async def test_get_intraday_data_invalid_symbol(self):
        """Test intraday data with invalid symbol."""
        with pytest.raises(DataValidationError):
            await self.client.get_intraday_data("", "5min")
    
    @pytest.mark.asyncio
    async def test_get_daily_data_success(self):
        """Test successful daily data retrieval."""
        mock_response = {
            "Meta Data": {
                "1. Information": "Daily Prices and Volumes",
                "2. Symbol": "AAPL"
            },
            "Time Series (Daily)": {
                "2023-01-15": {
                    "1. open": "150.00",
                    "2. high": "152.00",
                    "3. low": "149.00",
                    "4. close": "151.00",
                    "5. adjusted close": "151.00",
                    "6. volume": "1000000"
                }
            }
        }
        
        with patch.object(self.client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            response = await self.client.get_daily_data("AAPL")
            
            assert response.success
            assert response.symbol == "AAPL"
            assert response.data_type == MarketDataType.DAILY
            assert "ohlc_data" in response.data
    
    @pytest.mark.asyncio
    async def test_get_quote_success(self):
        """Test successful quote retrieval."""
        mock_response = {
            "Global Quote": {
                "01. symbol": "AAPL",
                "05. price": "150.00",
                "06. volume": "1000000",
                "09. change": "1.50",
                "10. change percent": "1.00%"
            }
        }
        
        with patch.object(self.client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            response = await self.client.get_quote("AAPL")
            
            assert response.success
            assert response.symbol == "AAPL"
            assert response.data_type == MarketDataType.QUOTE
            assert "quote" in response.data
            
            quote_data = response.data["quote"]
            assert quote_data["symbol"] == "AAPL"
            assert quote_data["last_price"] == Decimal("150.00")
    
    @pytest.mark.asyncio
    async def test_search_symbols_success(self):
        """Test successful symbol search."""
        mock_response = {
            "bestMatches": [
                {
                    "1. symbol": "AAPL",
                    "2. name": "Apple Inc.",
                    "3. type": "Equity",
                    "4. region": "United States",
                    "8. currency": "USD"
                }
            ]
        }
        
        with patch.object(self.client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            response = await self.client.search_symbols("Apple")
            
            assert response.success
            assert "bestMatches" in response.data
    
    def test_prepare_params(self):
        """Test parameter preparation."""
        params = {"function": "TIME_SERIES_DAILY", "symbol": "AAPL"}
        prepared = self.client._prepare_params(params)
        
        assert "apikey" in prepared
        assert prepared["apikey"] == self.api_key
        assert prepared["function"] == "TIME_SERIES_DAILY"
        assert prepared["symbol"] == "AAPL"
    
    def test_get_granularity_from_interval(self):
        """Test granularity extraction from interval."""
        assert self.client._get_granularity_from_interval("1min") == DataGranularity.MINUTE_1
        assert self.client._get_granularity_from_interval("5min") == DataGranularity.MINUTE_5
        assert self.client._get_granularity_from_interval("daily") == DataGranularity.DAILY
        assert self.client._get_granularity_from_interval("unknown") == DataGranularity.DAILY  # Default
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test client as async context manager."""
        async with AlphaVantageClient(api_key="test") as client:
            assert client.session is not None
            assert not client.session.closed
        
        # Session should be closed after context exit
        assert client.session.closed
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self):
        """Test rate limit handling."""
        # Test that rate limit bucket is checked before making requests
        rate_limit_bucket = self.client.rate_limits["default"]
        
        # Fill up the rate limit bucket
        for _ in range(rate_limit_bucket.max_calls):
            rate_limit_bucket.record_call()
        
        assert not rate_limit_bucket.can_make_call()
        
        rate_limit_info = self.client.get_rate_limit_info()
        assert rate_limit_info.calls_remaining == 0
    
    def test_parse_time_series_response_error_handling(self):
        """Test error handling in response parsing."""
        # Test with error message
        error_response = {"Error Message": "Invalid API call"}
        with pytest.raises(APIError):
            self.client._parse_time_series_response(
                error_response, "AAPL", MarketDataType.DAILY, DataGranularity.DAILY
            )
        
        # Test with rate limit note
        rate_limit_response = {"Note": "Thank you for using Alpha Vantage!"}
        with pytest.raises(APIError):
            self.client._parse_time_series_response(
                rate_limit_response, "AAPL", MarketDataType.DAILY, DataGranularity.DAILY
            )
        
        # Test with no time series data
        no_data_response = {"Meta Data": {}}
        with pytest.raises(APIError):
            self.client._parse_time_series_response(
                no_data_response, "AAPL", MarketDataType.DAILY, DataGranularity.DAILY
            )
    
    def test_parse_quote_response_error_handling(self):
        """Test error handling in quote response parsing."""
        # Test with error message
        error_response = {"Error Message": "Invalid symbol"}
        with pytest.raises(APIError):
            self.client._parse_quote_response(error_response, "INVALID")
        
        # Test with no quote data
        no_data_response = {}
        with pytest.raises(APIError):
            self.client._parse_quote_response(no_data_response, "AAPL")
    
    @pytest.mark.asyncio
    async def test_request_retry_mechanism(self):
        """Test request retry mechanism."""
        with patch.object(self.client, '_execute_request', new_callable=AsyncMock) as mock_execute:
            # First call fails, second succeeds
            mock_execute.side_effect = [
                Exception("Network error"),
                {"test": "data"}
            ]
            
            # Should succeed after retry
            result = await self.client._make_request("test", {})
            assert result == {"test": "data"}
            assert mock_execute.call_count == 2