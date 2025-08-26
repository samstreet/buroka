"""
Tests for data validation functions.
"""

import pytest
from datetime import datetime, timezone, date
from decimal import Decimal

from src.data.validation import (
    DataValidator, ValidationResult, validate_symbol, 
    validate_ohlc_data, validate_quote_data, sanitize_symbol,
    is_valid_ohlc_data
)
from src.data.models.market_data import DataValidationError


class TestDataValidator:
    """Test DataValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
    
    def test_validate_symbol_valid_cases(self):
        """Test valid symbol validation."""
        test_cases = [
            ("AAPL", "us_stock"),
            ("MSFT", "us_stock"),
            ("BRK.A", "us_stock_extended"),
            ("BTCUSD", "crypto"),
            ("EURUSD", "forex"),
            ("^GSPC", "index"),
            ("TEST", "general")
        ]
        
        for symbol, symbol_type in test_cases:
            result = self.validator.validate_symbol(symbol, symbol_type)
            assert result.is_valid, f"Symbol {symbol} of type {symbol_type} should be valid"
            assert result.cleaned_data == symbol.upper()
    
    def test_validate_symbol_invalid_cases(self):
        """Test invalid symbol validation."""
        test_cases = [
            ("", "us_stock"),  # Empty
            ("TOOLONG123", "us_stock"),  # Too long for US stock
            ("123456ABC", "us_stock"),  # Invalid characters for US stock
            ("AAPL$", "general"),  # Invalid character
            (None, "general"),  # None value
            (123, "general")  # Wrong type
        ]
        
        for symbol, symbol_type in test_cases:
            result = self.validator.validate_symbol(symbol, symbol_type)
            assert not result.is_valid, f"Symbol {symbol} of type {symbol_type} should be invalid"
    
    def test_validate_symbol_normalization(self):
        """Test symbol normalization."""
        result = self.validator.validate_symbol("  aapl  ", "us_stock")
        assert result.is_valid
        assert result.cleaned_data == "AAPL"
    
    def test_validate_ohlc_data_valid(self):
        """Test valid OHLC data validation."""
        data = {
            "symbol": "AAPL",
            "timestamp": datetime.now(timezone.utc),
            "open_price": Decimal("150.00"),
            "high_price": Decimal("152.00"),
            "low_price": Decimal("149.00"),
            "close_price": Decimal("151.00"),
            "volume": 1000000
        }
        
        result = self.validator.validate_ohlc_data(data)
        assert result.is_valid
        assert result.cleaned_data["symbol"] == "AAPL"
    
    def test_validate_ohlc_data_missing_fields(self):
        """Test OHLC validation with missing fields."""
        data = {
            "symbol": "AAPL",
            "timestamp": datetime.now(timezone.utc),
            "open_price": Decimal("150.00")
            # Missing required fields
        }
        
        result = self.validator.validate_ohlc_data(data)
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_validate_ohlc_data_invalid_relationships(self):
        """Test OHLC validation with invalid price relationships."""
        data = {
            "symbol": "AAPL",
            "timestamp": datetime.now(timezone.utc),
            "open_price": Decimal("150.00"),
            "high_price": Decimal("148.00"),  # High less than open
            "low_price": Decimal("149.00"),
            "close_price": Decimal("151.00"),
            "volume": 1000000
        }
        
        result = self.validator.validate_ohlc_data(data)
        assert not result.is_valid
        assert any("high price" in error.lower() for error in result.errors)
    
    def test_validate_quote_data_valid(self):
        """Test valid quote data validation."""
        data = {
            "symbol": "AAPL",
            "timestamp": datetime.now(timezone.utc),
            "bid_price": Decimal("149.50"),
            "ask_price": Decimal("150.00"),
            "bid_size": 1000,
            "ask_size": 2000,
            "last_price": Decimal("149.75"),
            "last_size": 500
        }
        
        result = self.validator.validate_quote_data(data)
        assert result.is_valid
        assert result.cleaned_data["symbol"] == "AAPL"
    
    def test_validate_quote_data_invalid_spread(self):
        """Test quote validation with invalid bid/ask spread."""
        data = {
            "symbol": "AAPL",
            "timestamp": datetime.now(timezone.utc),
            "bid_price": Decimal("150.00"),
            "ask_price": Decimal("149.50"),  # Ask less than bid
            "last_price": Decimal("149.75")
        }
        
        result = self.validator.validate_quote_data(data)
        assert result.is_valid  # Should pass validation but have warnings
        assert len(result.warnings) > 0
    
    def test_validate_timestamp_string_formats(self):
        """Test timestamp validation with various string formats."""
        test_cases = [
            "2023-01-15 10:30:00",
            "2023-01-15",
            "2023-01-15T10:30:00",
            "2023-01-15T10:30:00Z",
            "2023-01-15T10:30:00.123456Z"
        ]
        
        for timestamp_str in test_cases:
            result = self.validator._validate_timestamp(timestamp_str)
            assert result.is_valid, f"Timestamp format {timestamp_str} should be valid"
            assert isinstance(result.cleaned_data, datetime)
    
    def test_validate_timestamp_datetime_object(self):
        """Test timestamp validation with datetime object."""
        timestamp = datetime.now()
        result = self.validator._validate_timestamp(timestamp)
        assert result.is_valid
        assert result.cleaned_data.tzinfo is not None  # Should have timezone
    
    def test_validate_timestamp_invalid(self):
        """Test invalid timestamp validation."""
        invalid_cases = [
            "invalid-date",
            "2023-13-45",  # Invalid date
            123456,  # Wrong type (should be handled but might fail)
            datetime(1800, 1, 1)  # Too far in past
        ]
        
        for timestamp in invalid_cases:
            result = self.validator._validate_timestamp(timestamp)
            assert not result.is_valid, f"Timestamp {timestamp} should be invalid"
    
    def test_validate_price_valid(self):
        """Test valid price validation."""
        test_cases = [
            (Decimal("100.00"), "price"),
            (100.50, "price"),
            ("99.99", "price"),
            (1, "price")
        ]
        
        for price, field_name in test_cases:
            result = self.validator._validate_price(price, field_name)
            assert result.is_valid, f"Price {price} should be valid"
            assert isinstance(result.cleaned_data, Decimal)
    
    def test_validate_price_invalid(self):
        """Test invalid price validation."""
        invalid_cases = [
            (-100.00, "negative price"),
            (0, "zero price"),
            ("invalid", "non-numeric string"),
            (None, "none value")
        ]
        
        for price, description in invalid_cases:
            result = self.validator._validate_price(price, "test_price")
            assert not result.is_valid, f"{description} should be invalid"
    
    def test_validate_volume_valid(self):
        """Test valid volume validation."""
        test_cases = [
            (1000000, "volume"),
            (0, "zero volume"),
            ("500000", "string volume"),
            (1.5e6, "scientific notation")
        ]
        
        for volume, description in test_cases:
            result = self.validator._validate_volume(volume, "volume")
            assert result.is_valid, f"{description} should be valid"
            assert isinstance(result.cleaned_data, int)
    
    def test_validate_volume_invalid(self):
        """Test invalid volume validation."""
        invalid_cases = [
            (-1000, "negative volume"),
            ("invalid", "non-numeric string"),
            (None, "none value")
        ]
        
        for volume, description in invalid_cases:
            result = self.validator._validate_volume(volume, "volume")
            assert not result.is_valid, f"{description} should be invalid"


class TestConvenienceFunctions:
    """Test convenience validation functions."""
    
    def test_validate_symbol_function(self):
        """Test validate_symbol convenience function."""
        result = validate_symbol("AAPL")
        assert result.is_valid
        assert result.cleaned_data == "AAPL"
    
    def test_validate_ohlc_data_function(self):
        """Test validate_ohlc_data convenience function."""
        data = {
            "symbol": "AAPL",
            "timestamp": datetime.now(timezone.utc),
            "open_price": Decimal("150.00"),
            "high_price": Decimal("152.00"),
            "low_price": Decimal("149.00"),
            "close_price": Decimal("151.00"),
            "volume": 1000000
        }
        
        result = validate_ohlc_data(data)
        assert result.is_valid
    
    def test_sanitize_symbol_valid(self):
        """Test sanitize_symbol with valid input."""
        cleaned = sanitize_symbol("  aapl  ")
        assert cleaned == "AAPL"
    
    def test_sanitize_symbol_invalid(self):
        """Test sanitize_symbol with invalid input."""
        with pytest.raises(DataValidationError):
            sanitize_symbol("")
    
    def test_is_valid_ohlc_data_strict(self):
        """Test is_valid_ohlc_data with strict validation."""
        valid_data = {
            "symbol": "AAPL",
            "timestamp": datetime.now(timezone.utc),
            "open_price": Decimal("150.00"),
            "high_price": Decimal("152.00"),
            "low_price": Decimal("149.00"),
            "close_price": Decimal("151.00"),
            "volume": 1000000
        }
        
        invalid_data = {
            "symbol": "AAPL",
            "timestamp": datetime.now(timezone.utc),
            "open_price": Decimal("150.00"),
            # Missing required fields
        }
        
        assert is_valid_ohlc_data(valid_data, strict=True)
        assert not is_valid_ohlc_data(invalid_data, strict=True)
    
    def test_is_valid_ohlc_data_non_strict(self):
        """Test is_valid_ohlc_data with non-strict validation."""
        data_with_warnings = {
            "symbol": "AAPL",
            "timestamp": datetime.now(timezone.utc),
            "open_price": Decimal("150.00"),
            "high_price": Decimal("152.00"),
            "low_price": Decimal("149.00"),
            "close_price": Decimal("151.00"),
            "volume": 1000000000000  # Very large volume (warning)
        }
        
        # Should be valid in non-strict mode despite warnings
        assert is_valid_ohlc_data(data_with_warnings, strict=False)