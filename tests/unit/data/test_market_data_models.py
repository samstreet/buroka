"""
Tests for market data models.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from pydantic import ValidationError

from src.data.models.market_data import (
    OHLCData, QuoteData, NewsData, MarketDataResponse,
    APIRateLimitInfo, DataGranularity, MarketDataType,
    DataValidationError, APIError
)


class TestOHLCData:
    """Test OHLCData model."""
    
    def test_valid_ohlc_data(self):
        """Test creation of valid OHLC data."""
        ohlc = OHLCData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open_price=Decimal("150.00"),
            high_price=Decimal("152.00"),
            low_price=Decimal("149.00"),
            close_price=Decimal("151.00"),
            volume=1000000,
            granularity=DataGranularity.DAILY,
            data_type=MarketDataType.DAILY
        )
        
        assert ohlc.symbol == "AAPL"
        assert ohlc.open_price == Decimal("150.00")
        assert ohlc.high_price == Decimal("152.00")
        assert ohlc.low_price == Decimal("149.00")
        assert ohlc.close_price == Decimal("151.00")
        assert ohlc.volume == 1000000
    
    def test_symbol_validation(self):
        """Test symbol validation."""
        with pytest.raises(ValidationError):
            OHLCData(
                symbol="",  # Empty symbol
                timestamp=datetime.now(timezone.utc),
                open_price=Decimal("150.00"),
                high_price=Decimal("152.00"),
                low_price=Decimal("149.00"),
                close_price=Decimal("151.00"),
                volume=1000000,
                granularity=DataGranularity.DAILY,
                data_type=MarketDataType.DAILY
            )
    
    def test_symbol_normalization(self):
        """Test symbol is normalized to uppercase."""
        ohlc = OHLCData(
            symbol="aapl",  # lowercase
            timestamp=datetime.now(timezone.utc),
            open_price=Decimal("150.00"),
            high_price=Decimal("152.00"),
            low_price=Decimal("149.00"),
            close_price=Decimal("151.00"),
            volume=1000000,
            granularity=DataGranularity.DAILY,
            data_type=MarketDataType.DAILY
        )
        
        assert ohlc.symbol == "AAPL"
    
    def test_negative_price_validation(self):
        """Test negative price validation."""
        with pytest.raises(ValidationError):
            OHLCData(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                open_price=Decimal("-150.00"),  # Negative price
                high_price=Decimal("152.00"),
                low_price=Decimal("149.00"),
                close_price=Decimal("151.00"),
                volume=1000000,
                granularity=DataGranularity.DAILY,
                data_type=MarketDataType.DAILY
            )
    
    def test_high_price_validation(self):
        """Test high price should be highest."""
        with pytest.raises(ValidationError):
            OHLCData(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                open_price=Decimal("150.00"),
                high_price=Decimal("148.00"),  # High less than open
                low_price=Decimal("149.00"),
                close_price=Decimal("151.00"),
                volume=1000000,
                granularity=DataGranularity.DAILY,
                data_type=MarketDataType.DAILY
            )
    
    def test_low_price_validation(self):
        """Test low price should be lowest."""
        with pytest.raises(ValidationError):
            OHLCData(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                open_price=Decimal("150.00"),
                high_price=Decimal("152.00"),
                low_price=Decimal("153.00"),  # Low greater than high
                close_price=Decimal("151.00"),
                volume=1000000,
                granularity=DataGranularity.DAILY,
                data_type=MarketDataType.DAILY
            )
    
    def test_negative_volume_validation(self):
        """Test negative volume validation."""
        with pytest.raises(ValidationError):
            OHLCData(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                open_price=Decimal("150.00"),
                high_price=Decimal("152.00"),
                low_price=Decimal("149.00"),
                close_price=Decimal("151.00"),
                volume=-1000,  # Negative volume
                granularity=DataGranularity.DAILY,
                data_type=MarketDataType.DAILY
            )


class TestQuoteData:
    """Test QuoteData model."""
    
    def test_valid_quote_data(self):
        """Test creation of valid quote data."""
        quote = QuoteData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            bid_price=Decimal("149.50"),
            ask_price=Decimal("150.00"),
            bid_size=1000,
            ask_size=2000,
            last_price=Decimal("149.75"),
            last_size=500
        )
        
        assert quote.symbol == "AAPL"
        assert quote.bid_price == Decimal("149.50")
        assert quote.ask_price == Decimal("150.00")
        assert quote.last_price == Decimal("149.75")
    
    def test_ask_greater_than_bid_validation(self):
        """Test ask price should be greater than bid price."""
        with pytest.raises(ValidationError):
            QuoteData(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                bid_price=Decimal("150.00"),
                ask_price=Decimal("149.50"),  # Ask less than bid
                bid_size=1000,
                ask_size=2000,
                last_price=Decimal("149.75"),
                last_size=500
            )


class TestNewsData:
    """Test NewsData model."""
    
    def test_valid_news_data(self):
        """Test creation of valid news data."""
        news = NewsData(
            symbol="AAPL",
            headline="Apple Reports Strong Q4 Earnings",
            url="https://example.com/news/apple-earnings",
            source="Example News",
            published_at=datetime.now(timezone.utc),
            sentiment_score=0.8,
            relevance_score=0.9
        )
        
        assert news.symbol == "AAPL"
        assert news.headline == "Apple Reports Strong Q4 Earnings"
        assert news.sentiment_score == 0.8
        assert news.relevance_score == 0.9
    
    def test_sentiment_score_validation(self):
        """Test sentiment score validation."""
        with pytest.raises(ValidationError):
            NewsData(
                symbol="AAPL",
                headline="Test News",
                url="https://example.com/news",
                source="Example News",
                published_at=datetime.now(timezone.utc),
                sentiment_score=1.5  # Out of range
            )
    
    def test_relevance_score_validation(self):
        """Test relevance score validation."""
        with pytest.raises(ValidationError):
            NewsData(
                symbol="AAPL",
                headline="Test News",
                url="https://example.com/news",
                source="Example News",
                published_at=datetime.now(timezone.utc),
                relevance_score=-0.1  # Out of range
            )


class TestMarketDataResponse:
    """Test MarketDataResponse model."""
    
    def test_valid_response(self):
        """Test creation of valid market data response."""
        response = MarketDataResponse(
            symbol="AAPL",
            data_type=MarketDataType.DAILY,
            timestamp=datetime.now(timezone.utc),
            data={"test": "data"},
            metadata={"source": "test"},
            success=True
        )
        
        assert response.symbol == "AAPL"
        assert response.data_type == MarketDataType.DAILY
        assert response.success is True
        assert response.data == {"test": "data"}
    
    def test_error_response(self):
        """Test creation of error response."""
        response = MarketDataResponse(
            symbol="AAPL",
            data_type=MarketDataType.DAILY,
            timestamp=datetime.now(timezone.utc),
            data={},
            success=False,
            error_message="API Error"
        )
        
        assert response.success is False
        assert response.error_message == "API Error"


class TestAPIRateLimitInfo:
    """Test APIRateLimitInfo model."""
    
    def test_valid_rate_limit_info(self):
        """Test creation of valid rate limit info."""
        rate_limit = APIRateLimitInfo(
            calls_made=5,
            calls_remaining=10,
            reset_time=datetime.now(timezone.utc),
            limit_period=3600
        )
        
        assert rate_limit.calls_made == 5
        assert rate_limit.calls_remaining == 10
        assert rate_limit.limit_period == 3600
    
    def test_negative_calls_validation(self):
        """Test negative calls validation."""
        with pytest.raises(ValidationError):
            APIRateLimitInfo(
                calls_made=5,
                calls_remaining=-1,  # Negative calls remaining
                limit_period=3600
            )


class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_data_validation_error(self):
        """Test DataValidationError."""
        error = DataValidationError("Test validation error")
        assert str(error) == "Test validation error"
    
    def test_api_error(self):
        """Test APIError."""
        error = APIError("Test API error", status_code=400, response_data={"error": "bad request"})
        assert str(error) == "Test API error"
        assert error.status_code == 400
        assert error.response_data == {"error": "bad request"}
    
    def test_api_error_minimal(self):
        """Test APIError with minimal parameters."""
        error = APIError("Test error")
        assert str(error) == "Test error"
        assert error.status_code is None
        assert error.response_data is None