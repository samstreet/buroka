"""
Data transformation implementations for market data ingestion.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from decimal import Decimal

from .interfaces import IDataTransformer
from ..models.market_data import MarketDataResponse, MarketDataType, DataGranularity, OHLCData, QuoteData
from ..validation import DataValidator, ValidationResult


class MarketDataTransformer(IDataTransformer):
    """Transforms raw market data into normalized format for storage."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validator = DataValidator()
    
    async def transform(self, raw_data: MarketDataResponse) -> Dict[str, Any]:
        """Transform raw market data into normalized format."""
        try:
            if not raw_data.success:
                return {
                    "success": False,
                    "error": f"Raw data contains errors: {raw_data.error_message}",
                    "data": None
                }
            
            transformed_data = {
                "symbol": raw_data.symbol,
                "data_type": raw_data.data_type.value,
                "timestamp": raw_data.timestamp.isoformat(),
                "metadata": raw_data.metadata or {},
                "success": True,
                "records": []
            }
            
            if raw_data.data_type == MarketDataType.INTRADAY or raw_data.data_type == MarketDataType.DAILY:
                transformed_data["records"] = await self._transform_ohlc_data(raw_data)
            elif raw_data.data_type == MarketDataType.QUOTE:
                transformed_data["records"] = await self._transform_quote_data(raw_data)
            else:
                # For other data types, pass through with basic transformation
                transformed_data["records"] = [{"raw_data": raw_data.data}]
            
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Error transforming data for {raw_data.symbol}: {e}")
            return {
                "success": False,
                "error": f"Transformation error: {str(e)}",
                "data": None,
                "symbol": raw_data.symbol,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _transform_ohlc_data(self, raw_data: MarketDataResponse) -> List[Dict[str, Any]]:
        """Transform OHLC data."""
        records = []
        
        ohlc_data = raw_data.data.get("ohlc_data", [])
        
        for ohlc_item in ohlc_data:
            try:
                # Validate the data first
                validation_result = self.validator.validate_ohlc_data(ohlc_item)
                
                if validation_result.is_valid:
                    record = {
                        "symbol": ohlc_item["symbol"],
                        "timestamp": ohlc_item["timestamp"],
                        "open": float(ohlc_item["open_price"]),
                        "high": float(ohlc_item["high_price"]),
                        "low": float(ohlc_item["low_price"]),
                        "close": float(ohlc_item["close_price"]),
                        "volume": ohlc_item["volume"],
                        "adjusted_close": float(ohlc_item["adjusted_close"]) if ohlc_item.get("adjusted_close") else None,
                        "granularity": ohlc_item["granularity"],
                        "data_type": ohlc_item["data_type"],
                        "validation_warnings": validation_result.warnings
                    }
                    records.append(record)
                else:
                    self.logger.warning(f"Invalid OHLC data for {ohlc_item.get('symbol', 'unknown')}: {validation_result.errors}")
                    
            except Exception as e:
                self.logger.error(f"Error processing OHLC item: {e}")
                continue
        
        return records
    
    async def _transform_quote_data(self, raw_data: MarketDataResponse) -> List[Dict[str, Any]]:
        """Transform quote data."""
        records = []
        
        quote_data = raw_data.data.get("quote", {})
        
        if quote_data:
            try:
                # Validate the data first
                validation_result = self.validator.validate_quote_data(quote_data)
                
                if validation_result.is_valid:
                    record = {
                        "symbol": quote_data["symbol"],
                        "timestamp": quote_data["timestamp"],
                        "bid_price": float(quote_data["bid_price"]),
                        "ask_price": float(quote_data["ask_price"]),
                        "bid_size": quote_data["bid_size"],
                        "ask_size": quote_data["ask_size"],
                        "last_price": float(quote_data["last_price"]),
                        "last_size": quote_data["last_size"],
                        "change": float(quote_data.get("change", 0)),
                        "change_percent": float(quote_data.get("change_percent", 0)),
                        "validation_warnings": validation_result.warnings
                    }
                    records.append(record)
                else:
                    self.logger.warning(f"Invalid quote data for {quote_data.get('symbol', 'unknown')}: {validation_result.errors}")
                    
            except Exception as e:
                self.logger.error(f"Error processing quote data: {e}")
        
        return records
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate transformed data."""
        try:
            # Check required fields
            required_fields = ["symbol", "data_type", "timestamp", "success", "records"]
            for field in required_fields:
                if field not in data:
                    self.logger.error(f"Missing required field: {field}")
                    return False
            
            # Check data types
            if not isinstance(data["symbol"], str):
                self.logger.error("Symbol must be a string")
                return False
            
            if not isinstance(data["records"], list):
                self.logger.error("Records must be a list")
                return False
            
            # Validate individual records
            for i, record in enumerate(data["records"]):
                if not isinstance(record, dict):
                    self.logger.error(f"Record {i} must be a dictionary")
                    return False
                
                if "symbol" not in record:
                    self.logger.error(f"Record {i} missing symbol")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating transformed data: {e}")
            return False


class BatchDataTransformer(IDataTransformer):
    """Transforms batch of market data for efficient processing."""
    
    def __init__(self, base_transformer: IDataTransformer):
        self.base_transformer = base_transformer
        self.logger = logging.getLogger(__name__)
    
    async def transform(self, raw_data: MarketDataResponse) -> Dict[str, Any]:
        """Transform batch data by delegating to base transformer."""
        return await self.base_transformer.transform(raw_data)
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate batch data."""
        return self.base_transformer.validate_data(data)
    
    async def transform_batch(self, raw_data_batch: List[MarketDataResponse]) -> List[Dict[str, Any]]:
        """Transform a batch of raw market data."""
        transformed_batch = []
        
        for raw_data in raw_data_batch:
            try:
                transformed = await self.transform(raw_data)
                transformed_batch.append(transformed)
            except Exception as e:
                self.logger.error(f"Error transforming batch item for {raw_data.symbol}: {e}")
                # Add error record to batch
                error_record = {
                    "symbol": raw_data.symbol,
                    "data_type": raw_data.data_type.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "success": False,
                    "error": str(e),
                    "records": []
                }
                transformed_batch.append(error_record)
        
        return transformed_batch


class AggregateDataTransformer(IDataTransformer):
    """Transforms data with aggregation capabilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validator = DataValidator()
    
    async def transform(self, raw_data: MarketDataResponse) -> Dict[str, Any]:
        """Transform with aggregation."""
        # Use base transformation first
        base_transformer = MarketDataTransformer()
        base_result = await base_transformer.transform(raw_data)
        
        if not base_result.get("success", False):
            return base_result
        
        # Add aggregation metrics
        records = base_result.get("records", [])
        if records and raw_data.data_type in [MarketDataType.INTRADAY, MarketDataType.DAILY]:
            aggregation = self._calculate_aggregates(records)
            base_result["aggregation"] = aggregation
        
        return base_result
    
    def _calculate_aggregates(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate statistics for OHLC data."""
        if not records:
            return {}
        
        prices = []
        volumes = []
        
        for record in records:
            if all(key in record for key in ["open", "high", "low", "close", "volume"]):
                prices.extend([record["open"], record["high"], record["low"], record["close"]])
                volumes.append(record["volume"])
        
        if not prices:
            return {}
        
        return {
            "price_stats": {
                "min": min(prices),
                "max": max(prices),
                "avg": sum(prices) / len(prices),
                "count": len(records)
            },
            "volume_stats": {
                "total": sum(volumes),
                "avg": sum(volumes) / len(volumes) if volumes else 0,
                "min": min(volumes) if volumes else 0,
                "max": max(volumes) if volumes else 0
            }
        }
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate aggregated data."""
        base_transformer = MarketDataTransformer()
        if not base_transformer.validate_data(data):
            return False
        
        # Additional validation for aggregated data
        if "aggregation" in data:
            aggregation = data["aggregation"]
            if not isinstance(aggregation, dict):
                self.logger.error("Aggregation must be a dictionary")
                return False
        
        return True