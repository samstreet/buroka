"""
Data validation and sanitization functions for market data.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timezone, date
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass

from .models.market_data import (
    OHLCData, QuoteData, NewsData, DataValidationError,
    DataGranularity, MarketDataType
)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    cleaned_data: Optional[Dict[str, Any]] = None


class DataValidator:
    """Comprehensive data validator for market data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Symbol validation patterns
        self.symbol_patterns = {
            'us_stock': r'^[A-Z]{1,5}$',  # US stocks: 1-5 uppercase letters
            'us_stock_extended': r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$',  # With class (e.g., BRK.A)
            'crypto': r'^[A-Z]{3,5}(USD|BTC|ETH|USDT)?$',  # Crypto pairs
            'forex': r'^[A-Z]{6}$',  # Forex pairs (e.g., EURUSD)
            'index': r'^\^[A-Z0-9]{1,10}$',  # Index (e.g., ^GSPC)
            'general': r'^[A-Z0-9.-]{1,10}$'  # General pattern
        }
        
        # Price validation limits
        self.price_limits = {
            'min_price': Decimal('0.0001'),
            'max_price': Decimal('1000000.00'),
            'max_volume': 10_000_000_000,
            'max_price_change_percent': Decimal('50.0')  # 50% daily change limit
        }
    
    def validate_symbol(self, symbol: str, symbol_type: str = 'general') -> ValidationResult:
        """Validate and sanitize a trading symbol."""
        errors = []
        warnings = []
        cleaned_symbol = None
        
        try:
            # Basic validation
            if not symbol or not isinstance(symbol, str):
                errors.append("Symbol must be a non-empty string")
                return ValidationResult(False, errors, warnings)
            
            # Clean and normalize
            cleaned_symbol = symbol.strip().upper()
            
            if len(cleaned_symbol) == 0:
                errors.append("Symbol cannot be empty after cleaning")
                return ValidationResult(False, errors, warnings, cleaned_symbol)
            
            if len(cleaned_symbol) > 10:
                errors.append(f"Symbol too long: {len(cleaned_symbol)} characters")
                return ValidationResult(False, errors, warnings, cleaned_symbol)
            
            # Pattern validation
            pattern = self.symbol_patterns.get(symbol_type, self.symbol_patterns['general'])
            if not re.match(pattern, cleaned_symbol):
                if symbol_type != 'general':
                    # Try general pattern as fallback
                    if re.match(self.symbol_patterns['general'], cleaned_symbol):
                        warnings.append(f"Symbol doesn't match {symbol_type} pattern, using general validation")
                    else:
                        errors.append(f"Symbol doesn't match expected pattern for {symbol_type}")
                        return ValidationResult(False, errors, warnings, cleaned_symbol)
                else:
                    errors.append("Symbol contains invalid characters")
                    return ValidationResult(False, errors, warnings, cleaned_symbol)
            
            return ValidationResult(True, errors, warnings, cleaned_symbol)
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return ValidationResult(False, errors, warnings, cleaned_symbol)
    
    def validate_ohlc_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate OHLC (Open, High, Low, Close) market data."""
        errors = []
        warnings = []
        cleaned_data = data.copy()
        
        try:
            # Required fields
            required_fields = ['symbol', 'timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
            for field in required_fields:
                if field not in data:
                    errors.append(f"Missing required field: {field}")
            
            if errors:
                return ValidationResult(False, errors, warnings, cleaned_data)
            
            # Validate symbol
            symbol_result = self.validate_symbol(data['symbol'])
            if not symbol_result.is_valid:
                errors.extend([f"Symbol: {err}" for err in symbol_result.errors])
            else:
                cleaned_data['symbol'] = symbol_result.cleaned_data
                warnings.extend([f"Symbol: {warn}" for warn in symbol_result.warnings])
            
            # Validate timestamp
            timestamp_result = self._validate_timestamp(data['timestamp'])
            if not timestamp_result.is_valid:
                errors.extend([f"Timestamp: {err}" for err in timestamp_result.errors])
            else:
                cleaned_data['timestamp'] = timestamp_result.cleaned_data
                warnings.extend([f"Timestamp: {warn}" for warn in timestamp_result.warnings])
            
            # Validate prices
            price_fields = ['open_price', 'high_price', 'low_price', 'close_price']
            validated_prices = {}
            
            for field in price_fields:
                price_result = self._validate_price(data[field], field)
                if not price_result.is_valid:
                    errors.extend([f"{field}: {err}" for err in price_result.errors])
                else:
                    validated_prices[field] = price_result.cleaned_data
                    cleaned_data[field] = price_result.cleaned_data
                    warnings.extend([f"{field}: {warn}" for warn in price_result.warnings])
            
            # Cross-validate prices (OHLC logic)
            if len(validated_prices) == 4:
                ohlc_result = self._validate_ohlc_relationships(validated_prices)
                if not ohlc_result.is_valid:
                    errors.extend(ohlc_result.errors)
                warnings.extend(ohlc_result.warnings)
            
            # Validate volume
            volume_result = self._validate_volume(data['volume'])
            if not volume_result.is_valid:
                errors.extend([f"Volume: {err}" for err in volume_result.errors])
            else:
                cleaned_data['volume'] = volume_result.cleaned_data
                warnings.extend([f"Volume: {warn}" for warn in volume_result.warnings])
            
            # Validate optional adjusted_close
            if 'adjusted_close' in data and data['adjusted_close'] is not None:
                adj_result = self._validate_price(data['adjusted_close'], 'adjusted_close')
                if not adj_result.is_valid:
                    errors.extend([f"Adjusted close: {err}" for err in adj_result.errors])
                else:
                    cleaned_data['adjusted_close'] = adj_result.cleaned_data
                    warnings.extend([f"Adjusted close: {warn}" for warn in adj_result.warnings])
            
            return ValidationResult(len(errors) == 0, errors, warnings, cleaned_data)
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return ValidationResult(False, errors, warnings, cleaned_data)
    
    def validate_quote_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate real-time quote data."""
        errors = []
        warnings = []
        cleaned_data = data.copy()
        
        try:
            # Required fields
            required_fields = ['symbol', 'timestamp', 'last_price']
            for field in required_fields:
                if field not in data:
                    errors.append(f"Missing required field: {field}")
            
            if errors:
                return ValidationResult(False, errors, warnings, cleaned_data)
            
            # Validate symbol
            symbol_result = self.validate_symbol(data['symbol'])
            if not symbol_result.is_valid:
                errors.extend([f"Symbol: {err}" for err in symbol_result.errors])
            else:
                cleaned_data['symbol'] = symbol_result.cleaned_data
            
            # Validate timestamp
            timestamp_result = self._validate_timestamp(data['timestamp'])
            if not timestamp_result.is_valid:
                errors.extend([f"Timestamp: {err}" for err in timestamp_result.errors])
            else:
                cleaned_data['timestamp'] = timestamp_result.cleaned_data
            
            # Validate prices
            price_fields = ['last_price', 'bid_price', 'ask_price']
            for field in price_fields:
                if field in data:
                    price_result = self._validate_price(data[field], field)
                    if not price_result.is_valid:
                        errors.extend([f"{field}: {err}" for err in price_result.errors])
                    else:
                        cleaned_data[field] = price_result.cleaned_data
                        warnings.extend([f"{field}: {warn}" for warn in price_result.warnings])
            
            # Validate bid/ask relationship
            if 'bid_price' in cleaned_data and 'ask_price' in cleaned_data:
                if cleaned_data['ask_price'] <= cleaned_data['bid_price']:
                    warnings.append("Ask price should be greater than bid price")
            
            # Validate sizes
            size_fields = ['last_size', 'bid_size', 'ask_size']
            for field in size_fields:
                if field in data:
                    size_result = self._validate_volume(data[field], field)
                    if not size_result.is_valid:
                        errors.extend([f"{field}: {err}" for err in size_result.errors])
                    else:
                        cleaned_data[field] = size_result.cleaned_data
            
            return ValidationResult(len(errors) == 0, errors, warnings, cleaned_data)
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return ValidationResult(False, errors, warnings, cleaned_data)
    
    def validate_news_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate news data."""
        errors = []
        warnings = []
        cleaned_data = data.copy()
        
        try:
            # Required fields
            required_fields = ['symbol', 'headline', 'url', 'source', 'published_at']
            for field in required_fields:
                if field not in data or not data[field]:
                    errors.append(f"Missing required field: {field}")
            
            if errors:
                return ValidationResult(False, errors, warnings, cleaned_data)
            
            # Validate symbol
            symbol_result = self.validate_symbol(data['symbol'])
            if not symbol_result.is_valid:
                errors.extend([f"Symbol: {err}" for err in symbol_result.errors])
            else:
                cleaned_data['symbol'] = symbol_result.cleaned_data
            
            # Validate URL
            url_result = self._validate_url(data['url'])
            if not url_result.is_valid:
                errors.extend([f"URL: {err}" for err in url_result.errors])
            else:
                cleaned_data['url'] = url_result.cleaned_data
            
            # Validate timestamp
            timestamp_result = self._validate_timestamp(data['published_at'])
            if not timestamp_result.is_valid:
                errors.extend([f"Published at: {err}" for err in timestamp_result.errors])
            else:
                cleaned_data['published_at'] = timestamp_result.cleaned_data
            
            # Validate sentiment score
            if 'sentiment_score' in data and data['sentiment_score'] is not None:
                sentiment_result = self._validate_sentiment_score(data['sentiment_score'])
                if not sentiment_result.is_valid:
                    errors.extend([f"Sentiment: {err}" for err in sentiment_result.errors])
                else:
                    cleaned_data['sentiment_score'] = sentiment_result.cleaned_data
            
            return ValidationResult(len(errors) == 0, errors, warnings, cleaned_data)
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return ValidationResult(False, errors, warnings, cleaned_data)
    
    def _validate_timestamp(self, timestamp: Any) -> ValidationResult:
        """Validate and normalize timestamp."""
        errors = []
        warnings = []
        cleaned_timestamp = None
        
        try:
            if isinstance(timestamp, str):
                # Try common timestamp formats
                formats = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d',
                    '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%dT%H:%M:%SZ',
                    '%Y-%m-%dT%H:%M:%S.%fZ'
                ]
                
                parsed_timestamp = None
                for fmt in formats:
                    try:
                        parsed_timestamp = datetime.strptime(timestamp, fmt)
                        break
                    except ValueError:
                        continue
                
                if parsed_timestamp is None:
                    errors.append(f"Unable to parse timestamp: {timestamp}")
                    return ValidationResult(False, errors, warnings)
                
                # Ensure timezone info
                if parsed_timestamp.tzinfo is None:
                    parsed_timestamp = parsed_timestamp.replace(tzinfo=timezone.utc)
                    warnings.append("Assumed UTC timezone for timestamp")
                
                cleaned_timestamp = parsed_timestamp
                
            elif isinstance(timestamp, datetime):
                cleaned_timestamp = timestamp
                if timestamp.tzinfo is None:
                    cleaned_timestamp = timestamp.replace(tzinfo=timezone.utc)
                    warnings.append("Assumed UTC timezone for timestamp")
                    
            elif isinstance(timestamp, date):
                cleaned_timestamp = datetime.combine(timestamp, datetime.min.time()).replace(tzinfo=timezone.utc)
                
            else:
                errors.append(f"Invalid timestamp type: {type(timestamp)}")
                return ValidationResult(False, errors, warnings)
            
            # Check if timestamp is reasonable
            now = datetime.now(timezone.utc)
            if cleaned_timestamp > now:
                warnings.append("Timestamp is in the future")
            
            if cleaned_timestamp < datetime(1900, 1, 1, tzinfo=timezone.utc):
                errors.append("Timestamp is too far in the past")
                return ValidationResult(False, errors, warnings)
            
            return ValidationResult(True, errors, warnings, cleaned_timestamp)
            
        except Exception as e:
            errors.append(f"Timestamp validation error: {str(e)}")
            return ValidationResult(False, errors, warnings)
    
    def _validate_price(self, price: Any, field_name: str) -> ValidationResult:
        """Validate and normalize price data."""
        errors = []
        warnings = []
        cleaned_price = None
        
        try:
            # Convert to Decimal
            if isinstance(price, (int, float)):
                cleaned_price = Decimal(str(price))
            elif isinstance(price, str):
                cleaned_price = Decimal(price)
            elif isinstance(price, Decimal):
                cleaned_price = price
            else:
                errors.append(f"Invalid price type for {field_name}: {type(price)}")
                return ValidationResult(False, errors, warnings)
            
            # Validate range
            if cleaned_price <= 0:
                errors.append(f"{field_name} must be positive")
                return ValidationResult(False, errors, warnings)
            
            if cleaned_price < self.price_limits['min_price']:
                warnings.append(f"{field_name} is very small: {cleaned_price}")
            
            if cleaned_price > self.price_limits['max_price']:
                errors.append(f"{field_name} is too large: {cleaned_price}")
                return ValidationResult(False, errors, warnings)
            
            # Round to 4 decimal places
            cleaned_price = cleaned_price.quantize(Decimal('0.0001'))
            
            return ValidationResult(True, errors, warnings, cleaned_price)
            
        except InvalidOperation:
            errors.append(f"Invalid decimal value for {field_name}: {price}")
            return ValidationResult(False, errors, warnings)
        except Exception as e:
            errors.append(f"Price validation error for {field_name}: {str(e)}")
            return ValidationResult(False, errors, warnings)
    
    def _validate_volume(self, volume: Any, field_name: str = "volume") -> ValidationResult:
        """Validate volume data."""
        errors = []
        warnings = []
        cleaned_volume = None
        
        try:
            # Convert to int
            if isinstance(volume, (int, float)):
                cleaned_volume = int(volume)
            elif isinstance(volume, str):
                cleaned_volume = int(float(volume))  # Handle string floats
            else:
                errors.append(f"Invalid volume type for {field_name}: {type(volume)}")
                return ValidationResult(False, errors, warnings)
            
            # Validate range
            if cleaned_volume < 0:
                errors.append(f"{field_name} cannot be negative")
                return ValidationResult(False, errors, warnings)
            
            if cleaned_volume > self.price_limits['max_volume']:
                warnings.append(f"{field_name} is very large: {cleaned_volume}")
            
            return ValidationResult(True, errors, warnings, cleaned_volume)
            
        except (ValueError, TypeError):
            errors.append(f"Invalid volume value for {field_name}: {volume}")
            return ValidationResult(False, errors, warnings)
        except Exception as e:
            errors.append(f"Volume validation error for {field_name}: {str(e)}")
            return ValidationResult(False, errors, warnings)
    
    def _validate_ohlc_relationships(self, prices: Dict[str, Decimal]) -> ValidationResult:
        """Validate OHLC price relationships."""
        errors = []
        warnings = []
        
        open_price = prices.get('open_price')
        high_price = prices.get('high_price')
        low_price = prices.get('low_price')
        close_price = prices.get('close_price')
        
        if not all([open_price, high_price, low_price, close_price]):
            errors.append("Missing price data for OHLC validation")
            return ValidationResult(False, errors, warnings)
        
        # High should be the highest
        if high_price < max(open_price, close_price):
            errors.append("High price should be >= max(open, close)")
        
        if high_price < low_price:
            errors.append("High price cannot be less than low price")
        
        # Low should be the lowest
        if low_price > min(open_price, close_price):
            errors.append("Low price should be <= min(open, close)")
        
        # Check for unusual price movements (more than 50% change)
        if open_price > 0:
            change_percent = abs((close_price - open_price) / open_price * 100)
            if change_percent > self.price_limits['max_price_change_percent']:
                warnings.append(f"Large price movement: {change_percent:.2f}%")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def _validate_url(self, url: str) -> ValidationResult:
        """Validate URL format."""
        errors = []
        warnings = []
        
        if not isinstance(url, str):
            errors.append("URL must be a string")
            return ValidationResult(False, errors, warnings, None)
        
        url = url.strip()
        
        # Basic URL validation
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if not re.match(url_pattern, url):
            errors.append("Invalid URL format")
            return ValidationResult(False, errors, warnings, None)
        
        if len(url) > 2000:
            warnings.append("URL is very long")
        
        return ValidationResult(True, errors, warnings, url)
    
    def _validate_sentiment_score(self, score: Any) -> ValidationResult:
        """Validate sentiment score (-1 to 1)."""
        errors = []
        warnings = []
        
        try:
            if isinstance(score, (int, float)):
                cleaned_score = float(score)
            elif isinstance(score, str):
                cleaned_score = float(score)
            else:
                errors.append(f"Invalid sentiment score type: {type(score)}")
                return ValidationResult(False, errors, warnings)
            
            if not -1 <= cleaned_score <= 1:
                errors.append(f"Sentiment score must be between -1 and 1: {cleaned_score}")
                return ValidationResult(False, errors, warnings)
            
            return ValidationResult(True, errors, warnings, cleaned_score)
            
        except (ValueError, TypeError):
            errors.append(f"Invalid sentiment score value: {score}")
            return ValidationResult(False, errors, warnings)


# Convenience functions
def validate_symbol(symbol: str, symbol_type: str = 'general') -> ValidationResult:
    """Validate a trading symbol."""
    validator = DataValidator()
    return validator.validate_symbol(symbol, symbol_type)


def validate_ohlc_data(data: Dict[str, Any]) -> ValidationResult:
    """Validate OHLC data."""
    validator = DataValidator()
    return validator.validate_ohlc_data(data)


def validate_quote_data(data: Dict[str, Any]) -> ValidationResult:
    """Validate quote data."""
    validator = DataValidator()
    return validator.validate_quote_data(data)


def sanitize_symbol(symbol: str) -> str:
    """Sanitize and normalize a symbol."""
    result = validate_symbol(symbol)
    if result.is_valid:
        return result.cleaned_data
    else:
        raise DataValidationError(f"Invalid symbol: {'; '.join(result.errors)}")


def is_valid_ohlc_data(data: Dict[str, Any], strict: bool = True) -> bool:
    """Check if OHLC data is valid."""
    result = validate_ohlc_data(data)
    return result.is_valid if strict else len(result.errors) == 0