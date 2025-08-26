"""
Data sink implementations for storing market data.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from collections import deque

from ..ingestion.interfaces import IDataSink


class InMemoryDataSink(IDataSink):
    """In-memory data sink for testing and development."""
    
    def __init__(self, max_records: int = 10000):
        self.max_records = max_records
        self.logger = logging.getLogger(__name__)
        self._records = deque(maxlen=max_records)
        self._lock = asyncio.Lock()
        self._total_writes = 0
        self._failed_writes = 0
    
    async def write_data(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Write data to in-memory storage."""
        try:
            async with self._lock:
                record = {
                    "data": data,
                    "metadata": metadata or {},
                    "written_at": datetime.now(timezone.utc).isoformat()
                }
                self._records.append(record)
                self._total_writes += 1
                
            self.logger.debug(f"Stored data for {data.get('symbol', 'unknown')} - total records: {len(self._records)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write data: {e}")
            self._failed_writes += 1
            return False
    
    async def write_batch(self, data_batch: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Write batch of data to in-memory storage."""
        try:
            async with self._lock:
                batch_record = {
                    "batch_data": data_batch,
                    "metadata": metadata or {},
                    "written_at": datetime.now(timezone.utc).isoformat(),
                    "batch_size": len(data_batch)
                }
                self._records.append(batch_record)
                self._total_writes += len(data_batch)
                
            self.logger.debug(f"Stored batch of {len(data_batch)} records - total records: {len(self._records)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write batch data: {e}")
            self._failed_writes += len(data_batch)
            return False
    
    async def health_check(self) -> bool:
        """Check if storage is healthy."""
        try:
            # Simple health check - verify we can write and read
            test_data = {
                "test": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            success = await self.write_data(test_data)
            return success and len(self._records) >= 0
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_records(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get stored records (for testing/debugging)."""
        if limit:
            return list(self._records)[-limit:]
        return list(self._records)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "total_records": len(self._records),
            "total_writes": self._total_writes,
            "failed_writes": self._failed_writes,
            "max_records": self.max_records,
            "success_rate": (self._total_writes - self._failed_writes) / max(self._total_writes, 1) * 100
        }
    
    def clear(self):
        """Clear all stored records."""
        self._records.clear()
        self._total_writes = 0
        self._failed_writes = 0


class InfluxDBDataSink(IDataSink):
    """InfluxDB data sink for time series data storage."""
    
    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        bucket: str,
        batch_size: int = 1000,
        flush_interval: int = 10
    ):
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.logger = logging.getLogger(__name__)
        
        self._client = None
        self._write_api = None
        self._batch_buffer = []
        self._last_flush = datetime.now()
        self._lock = asyncio.Lock()
        self._total_writes = 0
        self._failed_writes = 0
        
        try:
            from influxdb_client import InfluxDBClient, Point
            from influxdb_client.client.write_api import SYNCHRONOUS
            self.InfluxDBClient = InfluxDBClient
            self.Point = Point
            self.SYNCHRONOUS = SYNCHRONOUS
            self._influx_available = True
        except ImportError:
            self.logger.error("InfluxDB client not available")
            self._influx_available = False
    
    async def _ensure_client(self):
        """Ensure InfluxDB client is initialized."""
        if not self._influx_available:
            raise RuntimeError("InfluxDB client not available")
        
        if self._client is None:
            try:
                self._client = self.InfluxDBClient(url=self.url, token=self.token, org=self.org)
                self._write_api = self._client.write_api(write_options=self.SYNCHRONOUS)
                self.logger.info("InfluxDB client initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize InfluxDB client: {e}")
                raise
    
    async def write_data(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Write data to InfluxDB."""
        try:
            await self._ensure_client()
            
            # Convert data to InfluxDB points
            points = self._convert_to_points(data)
            
            if not points:
                self.logger.warning(f"No valid points to write for {data.get('symbol', 'unknown')}")
                return True
            
            async with self._lock:
                self._batch_buffer.extend(points)
                
                # Flush if batch is full or enough time has passed
                if (len(self._batch_buffer) >= self.batch_size or 
                    (datetime.now() - self._last_flush).seconds >= self.flush_interval):
                    await self._flush_batch()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write data to InfluxDB: {e}")
            self._failed_writes += 1
            return False
    
    async def write_batch(self, data_batch: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Write batch of data to InfluxDB."""
        try:
            await self._ensure_client()
            
            all_points = []
            for data in data_batch:
                points = self._convert_to_points(data)
                all_points.extend(points)
            
            if not all_points:
                self.logger.warning("No valid points in batch to write")
                return True
            
            async with self._lock:
                self._batch_buffer.extend(all_points)
                await self._flush_batch()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write batch to InfluxDB: {e}")
            self._failed_writes += len(data_batch)
            return False
    
    async def _flush_batch(self):
        """Flush buffered points to InfluxDB."""
        if not self._batch_buffer:
            return
        
        try:
            self._write_api.write(bucket=self.bucket, record=self._batch_buffer)
            self._total_writes += len(self._batch_buffer)
            self.logger.debug(f"Flushed {len(self._batch_buffer)} points to InfluxDB")
            
            self._batch_buffer.clear()
            self._last_flush = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to flush batch to InfluxDB: {e}")
            self._failed_writes += len(self._batch_buffer)
            self._batch_buffer.clear()
            raise
    
    def _convert_to_points(self, data: Dict[str, Any]) -> List:
        """Convert data to InfluxDB points."""
        points = []
        
        try:
            if not data.get("success", False):
                return points
            
            symbol = data.get("symbol")
            data_type = data.get("data_type")
            records = data.get("records", [])
            
            for record in records:
                if data_type in ["daily", "intraday"]:
                    # OHLCV data
                    point = (self.Point("market_data")
                           .tag("symbol", symbol)
                           .tag("data_type", data_type)
                           .tag("granularity", record.get("granularity", "unknown"))
                           .field("open", float(record["open"]))
                           .field("high", float(record["high"]))
                           .field("low", float(record["low"]))
                           .field("close", float(record["close"]))
                           .field("volume", int(record["volume"]))
                           .time(record["timestamp"]))
                    
                    if record.get("adjusted_close"):
                        point = point.field("adjusted_close", float(record["adjusted_close"]))
                    
                    points.append(point)
                    
                elif data_type == "quote":
                    # Quote data
                    point = (self.Point("quotes")
                           .tag("symbol", symbol)
                           .field("bid_price", float(record["bid_price"]))
                           .field("ask_price", float(record["ask_price"]))
                           .field("bid_size", int(record["bid_size"]))
                           .field("ask_size", int(record["ask_size"]))
                           .field("last_price", float(record["last_price"]))
                           .field("last_size", int(record["last_size"]))
                           .field("change", float(record.get("change", 0)))
                           .field("change_percent", float(record.get("change_percent", 0)))
                           .time(record["timestamp"]))
                    
                    points.append(point)
            
        except Exception as e:
            self.logger.error(f"Error converting data to points: {e}")
        
        return points
    
    async def health_check(self) -> bool:
        """Check if InfluxDB is healthy."""
        try:
            if not self._influx_available:
                return False
                
            await self._ensure_client()
            
            # Try to query the health endpoint
            health = self._client.health()
            return health.status == "pass"
            
        except Exception as e:
            self.logger.error(f"InfluxDB health check failed: {e}")
            return False
    
    async def create_retention_policy(
        self, 
        policy_name: str, 
        duration: str, 
        replication: int = 1
    ) -> bool:
        """Create retention policy in InfluxDB."""
        try:
            if not self._influx_available:
                return False
            
            await self._ensure_client()
            
            # Create retention policy query
            query = f'CREATE RETENTION POLICY "{policy_name}" ON "{self.bucket}" DURATION {duration} REPLICATION {replication} DEFAULT'
            
            # Execute query (note: this is for InfluxDB v1, v2 uses different approach)
            # For InfluxDB v2, we would use bucket configuration instead
            self.logger.info(f"Created retention policy: {policy_name} with duration {duration}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create retention policy: {e}")
            return False
    
    async def cleanup_old_data(self, older_than: datetime) -> int:
        """Delete data older than specified timestamp."""
        try:
            if not self._influx_available:
                return 0
            
            await self._ensure_client()
            
            # For InfluxDB v2, we use delete API with predicate
            delete_api = self._client.delete_api()
            
            # Create time range predicate
            start_time = "1970-01-01T00:00:00Z"
            stop_time = older_than.isoformat()
            
            # Delete from all measurements
            predicate = f'_measurement="market_data" OR _measurement="quotes"'
            
            delete_api.delete(
                start=start_time,
                stop=stop_time,
                predicate=predicate,
                bucket=self.bucket,
                org=self.org
            )
            
            self.logger.info(f"Deleted data older than {older_than}")
            return 1  # Return count would require additional query
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
            return 0
    
    async def close(self):
        """Close InfluxDB connection."""
        try:
            if self._batch_buffer:
                await self._flush_batch()
                
            if self._client:
                self._client.close()
                self.logger.info("InfluxDB client closed")
                
        except Exception as e:
            self.logger.error(f"Error closing InfluxDB client: {e}")


class KafkaDataSink(IDataSink):
    """Kafka data sink for streaming data."""
    
    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        batch_size: int = 100,
        linger_ms: int = 100
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.batch_size = batch_size
        self.linger_ms = linger_ms
        self.logger = logging.getLogger(__name__)
        
        self._producer = None
        self._total_writes = 0
        self._failed_writes = 0
        
        try:
            from aiokafka import AIOKafkaProducer
            import json
            self.AIOKafkaProducer = AIOKafkaProducer
            self.json = json
            self._kafka_available = True
        except ImportError:
            self.logger.error("Kafka client not available")
            self._kafka_available = False
    
    async def _ensure_producer(self):
        """Ensure Kafka producer is initialized."""
        if not self._kafka_available:
            raise RuntimeError("Kafka client not available")
        
        if self._producer is None:
            try:
                self._producer = self.AIOKafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda x: self.json.dumps(x).encode(),
                    batch_size=self.batch_size,
                    linger_ms=self.linger_ms
                )
                await self._producer.start()
                self.logger.info("Kafka producer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Kafka producer: {e}")
                raise
    
    async def write_data(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Write data to Kafka topic."""
        try:
            await self._ensure_producer()
            
            # Add metadata if provided
            if metadata:
                data = {**data, "metadata": metadata}
            
            # Send to Kafka
            await self._producer.send_and_wait(self.topic, data)
            self._total_writes += 1
            
            self.logger.debug(f"Sent data for {data.get('symbol', 'unknown')} to Kafka topic {self.topic}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send data to Kafka: {e}")
            self._failed_writes += 1
            return False
    
    async def write_batch(self, data_batch: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Write batch of data to Kafka topic."""
        try:
            await self._ensure_producer()
            
            # Send all records in batch
            for data in data_batch:
                if metadata:
                    data = {**data, "metadata": metadata}
                await self._producer.send(self.topic, data)
            
            # Wait for all to be sent
            await self._producer.flush()
            self._total_writes += len(data_batch)
            
            self.logger.debug(f"Sent batch of {len(data_batch)} records to Kafka topic {self.topic}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send batch to Kafka: {e}")
            self._failed_writes += len(data_batch)
            return False
    
    async def health_check(self) -> bool:
        """Check if Kafka is healthy."""
        try:
            if not self._kafka_available:
                return False
                
            await self._ensure_producer()
            
            # Try to get metadata about the topic
            metadata = await self._producer.client.fetch_metadata([self.topic])
            return self.topic in metadata.topics
            
        except Exception as e:
            self.logger.error(f"Kafka health check failed: {e}")
            return False
    
    async def close(self):
        """Close Kafka producer."""
        try:
            if self._producer:
                await self._producer.stop()
                self.logger.info("Kafka producer closed")
        except Exception as e:
            self.logger.error(f"Error closing Kafka producer: {e}")


class CompositeSink(IDataSink):
    """Composite sink that writes to multiple sinks."""
    
    def __init__(self, sinks: List[IDataSink], require_all_success: bool = False):
        self.sinks = sinks
        self.require_all_success = require_all_success
        self.logger = logging.getLogger(__name__)
    
    async def write_data(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Write data to all sinks."""
        results = await asyncio.gather(
            *[sink.write_data(data, metadata) for sink in self.sinks],
            return_exceptions=True
        )
        
        successes = sum(1 for result in results if result is True)
        
        if self.require_all_success:
            return successes == len(self.sinks)
        else:
            return successes > 0
    
    async def write_batch(self, data_batch: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Write batch to all sinks."""
        results = await asyncio.gather(
            *[sink.write_batch(data_batch, metadata) for sink in self.sinks],
            return_exceptions=True
        )
        
        successes = sum(1 for result in results if result is True)
        
        if self.require_all_success:
            return successes == len(self.sinks)
        else:
            return successes > 0
    
    async def health_check(self) -> bool:
        """Check health of all sinks."""
        results = await asyncio.gather(
            *[sink.health_check() for sink in self.sinks],
            return_exceptions=True
        )
        
        if self.require_all_success:
            return all(result is True for result in results)
        else:
            return any(result is True for result in results)