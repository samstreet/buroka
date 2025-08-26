"""
Interfaces for data ingestion services following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime

from ..models.market_data import MarketDataResponse, DataGranularity, MarketDataType


class IDataSource(ABC):
    """Interface for data sources (API clients)."""
    
    @abstractmethod
    async def get_intraday_data(self, symbol: str, interval: str) -> MarketDataResponse:
        """Get intraday market data for a symbol."""
        pass
    
    @abstractmethod
    async def get_daily_data(self, symbol: str, outputsize: str = "compact") -> MarketDataResponse:
        """Get daily market data for a symbol."""
        pass
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> MarketDataResponse:
        """Get real-time quote for a symbol."""
        pass
    
    @abstractmethod
    async def search_symbols(self, keywords: str) -> MarketDataResponse:
        """Search for symbols matching keywords."""
        pass
    
    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol format."""
        pass


class IDataTransformer(ABC):
    """Interface for data transformation."""
    
    @abstractmethod
    async def transform(self, raw_data: MarketDataResponse) -> Dict[str, Any]:
        """Transform raw market data into normalized format."""
        pass
    
    @abstractmethod
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate transformed data."""
        pass


class IDataSink(ABC):
    """Interface for data storage destinations."""
    
    @abstractmethod
    async def write_data(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Write data to storage."""
        pass
    
    @abstractmethod
    async def write_batch(self, data_batch: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Write batch of data to storage."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if storage is healthy."""
        pass


class IMetricsCollector(ABC):
    """Interface for metrics collection."""
    
    @abstractmethod
    def increment_counter(self, metric_name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        pass
    
    @abstractmethod
    def set_gauge(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        pass
    
    @abstractmethod
    def record_timing(self, metric_name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record timing metric."""
        pass
    
    @abstractmethod
    def record_histogram(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record histogram metric."""
        pass


class IIngestionScheduler(ABC):
    """Interface for scheduling data ingestion."""
    
    @abstractmethod
    async def schedule_ingestion(
        self, 
        symbol: str, 
        data_type: MarketDataType, 
        interval: int,
        granularity: Optional[DataGranularity] = None
    ) -> str:
        """Schedule data ingestion for a symbol."""
        pass
    
    @abstractmethod
    async def cancel_ingestion(self, job_id: str) -> bool:
        """Cancel a scheduled ingestion job."""
        pass
    
    @abstractmethod
    async def get_active_jobs(self) -> List[Dict[str, Any]]:
        """Get list of active ingestion jobs."""
        pass
    
    @abstractmethod
    async def pause_ingestion(self, job_id: str) -> bool:
        """Pause an ingestion job."""
        pass
    
    @abstractmethod
    async def resume_ingestion(self, job_id: str) -> bool:
        """Resume a paused ingestion job."""
        pass


class IDataIngestionService(ABC):
    """Main interface for data ingestion service."""
    
    @abstractmethod
    async def ingest_symbol_data(
        self, 
        symbol: str, 
        data_type: MarketDataType,
        granularity: Optional[DataGranularity] = None
    ) -> Dict[str, Any]:
        """Ingest data for a single symbol."""
        pass
    
    @abstractmethod
    async def ingest_batch_data(self, symbols: List[str], data_type: MarketDataType) -> Dict[str, Any]:
        """Ingest data for multiple symbols."""
        pass
    
    @abstractmethod
    async def start_continuous_ingestion(self, symbols: List[str], interval: int = 300) -> str:
        """Start continuous data ingestion for symbols."""
        pass
    
    @abstractmethod
    async def stop_continuous_ingestion(self, job_id: str) -> bool:
        """Stop continuous data ingestion."""
        pass
    
    @abstractmethod
    async def get_ingestion_status(self) -> Dict[str, Any]:
        """Get current ingestion status and statistics."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        pass


class ICircuitBreaker(ABC):
    """Interface for circuit breaker pattern."""
    
    @abstractmethod
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        pass
    
    @abstractmethod
    def get_state(self) -> str:
        """Get current circuit breaker state."""
        pass
    
    @abstractmethod
    def get_failure_count(self) -> int:
        """Get current failure count."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset circuit breaker to closed state."""
        pass


class IDataStorageService(ABC):
    """Interface for data storage service."""
    
    @abstractmethod
    async def store_data(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store single data record."""
        pass
    
    @abstractmethod
    async def store_batch(self, data_batch: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store batch of data records."""
        pass
    
    @abstractmethod
    async def create_retention_policy(self, policy_name: str, duration: str, replication: int = 1) -> bool:
        """Create data retention policy."""
        pass
    
    @abstractmethod
    async def cleanup_old_data(self, older_than: datetime) -> int:
        """Clean up data older than specified date."""
        pass
    
    @abstractmethod
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage performance statistics."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check health of storage service."""
        pass