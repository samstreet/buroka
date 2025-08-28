"""
Optimized Kafka producer and consumer implementations with performance tuning.
Designed for high-throughput market data processing.
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from collections import defaultdict

try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    from aiokafka.helpers import create_ssl_context
    from kafka import KafkaAdminClient, ConfigResource, ConfigResourceType
    from kafka.admin import NewPartitions
    from kafka.errors import TopicAlreadyExistsError
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False

from src.utils.performance_profiler import get_performance_profiler

logger = logging.getLogger(__name__)


@dataclass
class KafkaOptimizationConfig:
    """Kafka optimization configuration settings."""
    
    # Producer optimizations
    batch_size: int = 65536  # 64KB
    linger_ms: int = 10  # Small linger for low latency
    compression_type: str = "lz4"  # Fast compression
    acks: Union[str, int] = 1  # Wait for leader acknowledgment
    retries: int = 3
    max_in_flight_requests_per_connection: int = 5
    buffer_memory: int = 33554432  # 32MB
    
    # Consumer optimizations
    fetch_min_bytes: int = 1024  # 1KB
    fetch_max_wait_ms: int = 500
    max_partition_fetch_bytes: int = 1048576  # 1MB
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 3000
    max_poll_records: int = 1000
    auto_offset_reset: str = "latest"
    
    # Connection settings
    bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_plain_username: Optional[str] = None
    sasl_plain_password: Optional[str] = None
    ssl_context: Optional[Any] = None
    
    # Performance tuning
    enable_metrics: bool = True
    metrics_sample_window_ms: int = 30000
    metrics_num_samples: int = 2


@dataclass
class ProducerMetrics:
    """Kafka producer performance metrics."""
    topic: str
    messages_sent: int = 0
    messages_failed: int = 0
    bytes_sent: int = 0
    total_send_time: float = 0.0
    batch_count: int = 0
    compression_ratio: float = 0.0
    error_rate: float = 0.0
    throughput_msg_per_sec: float = 0.0
    throughput_mb_per_sec: float = 0.0
    avg_latency_ms: float = 0.0


@dataclass  
class ConsumerMetrics:
    """Kafka consumer performance metrics."""
    topic: str
    messages_consumed: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    bytes_consumed: int = 0
    total_processing_time: float = 0.0
    lag_ms: float = 0.0
    throughput_msg_per_sec: float = 0.0
    throughput_mb_per_sec: float = 0.0
    avg_processing_time_ms: float = 0.0


class OptimizedKafkaProducer:
    """
    High-performance Kafka producer optimized for market data.
    """
    
    def __init__(
        self,
        topic: str,
        config: Optional[KafkaOptimizationConfig] = None,
        custom_partitioner: Optional[Callable] = None
    ):
        if not HAS_KAFKA:
            raise ImportError("aiokafka required for Kafka operations")
        
        self.topic = topic
        self.config = config or KafkaOptimizationConfig()
        self.custom_partitioner = custom_partitioner
        
        self._producer: Optional[AIOKafkaProducer] = None
        self._metrics = ProducerMetrics(topic=topic)
        self._last_metrics_update = time.time()
        self._lock = asyncio.Lock()
        
        self.profiler = get_performance_profiler()
    
    async def initialize(self):
        """Initialize the Kafka producer."""
        try:
            producer_config = {
                'bootstrap_servers': ','.join(self.config.bootstrap_servers),
                'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
                'key_serializer': lambda k: k.encode('utf-8') if k else None,
                'batch_size': self.config.batch_size,
                'linger_ms': self.config.linger_ms,
                'compression_type': self.config.compression_type,
                'acks': self.config.acks,
                'retries': self.config.retries,
                'max_in_flight_requests_per_connection': self.config.max_in_flight_requests_per_connection,
                'buffer_memory': self.config.buffer_memory,
                'security_protocol': self.config.security_protocol,
                'enable_idempotence': True,  # Prevent duplicate messages
            }
            
            # Add authentication if configured
            if self.config.sasl_mechanism:
                producer_config['sasl_mechanism'] = self.config.sasl_mechanism
                producer_config['sasl_plain_username'] = self.config.sasl_plain_username
                producer_config['sasl_plain_password'] = self.config.sasl_plain_password
            
            if self.config.ssl_context:
                producer_config['ssl_context'] = self.config.ssl_context
            
            # Add custom partitioner if provided
            if self.custom_partitioner:
                producer_config['partitioner'] = self.custom_partitioner
            
            self._producer = AIOKafkaProducer(**producer_config)
            await self._producer.start()
            
            logger.info(f"Kafka producer initialized for topic: {self.topic}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    async def send_message(
        self,
        message: Dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[Dict[str, bytes]] = None,
        partition: Optional[int] = None
    ) -> Dict[str, Any]:
        """Send a single message with performance tracking."""
        if not self._producer:
            raise RuntimeError("Producer not initialized")
        
        start_time = time.time()
        message_size = len(json.dumps(message).encode('utf-8'))
        
        # Add metadata to message
        enriched_message = {
            **message,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'producer_id': id(self),
            'message_id': str(uuid.uuid4())
        }
        
        try:
            # Profile message sending
            async with self.profiler.db_profiler.profile_query("kafka_send", f"topic_{self.topic}"):
                record_metadata = await self._producer.send(
                    self.topic,
                    value=enriched_message,
                    key=key,
                    headers=headers,
                    partition=partition
                )
            
            send_duration = time.time() - start_time
            
            # Update metrics
            async with self._lock:
                self._metrics.messages_sent += 1
                self._metrics.bytes_sent += message_size
                self._metrics.total_send_time += send_duration
                
                # Calculate throughput metrics
                if time.time() - self._last_metrics_update > 1.0:  # Update every second
                    self._update_throughput_metrics()
            
            return {
                'success': True,
                'topic': record_metadata.topic,
                'partition': record_metadata.partition,
                'offset': record_metadata.offset,
                'timestamp': record_metadata.timestamp,
                'send_duration_ms': send_duration * 1000,
                'message_size_bytes': message_size
            }
            
        except Exception as e:
            async with self._lock:
                self._metrics.messages_failed += 1
            
            logger.error(f"Failed to send message to {self.topic}: {e}")
            return {
                'success': False,
                'error': str(e),
                'send_duration_ms': (time.time() - start_time) * 1000
            }
    
    async def send_batch(
        self,
        messages: List[Dict[str, Any]],
        keys: Optional[List[str]] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Send multiple messages efficiently."""
        if not messages:
            return {'success': True, 'sent': 0, 'failed': 0}
        
        batch_size = batch_size or self.config.batch_size or len(messages)
        keys = keys or [None] * len(messages)
        
        results = {'sent': 0, 'failed': 0, 'results': []}
        
        # Process in batches
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i:i + batch_size]
            batch_keys = keys[i:i + batch_size] if keys else [None] * len(batch_messages)
            
            # Send batch concurrently
            tasks = [
                self.send_message(msg, key)
                for msg, key in zip(batch_messages, batch_keys)
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, dict) and result.get('success'):
                    results['sent'] += 1
                else:
                    results['failed'] += 1
                results['results'].append(result)
        
        async with self._lock:
            self._metrics.batch_count += 1
        
        return {
            'success': results['failed'] == 0,
            'total_messages': len(messages),
            **results
        }
    
    def _update_throughput_metrics(self):
        """Update throughput metrics."""
        now = time.time()
        time_window = now - self._last_metrics_update
        
        if time_window > 0:
            self._metrics.throughput_msg_per_sec = self._metrics.messages_sent / time_window
            self._metrics.throughput_mb_per_sec = (self._metrics.bytes_sent / time_window) / (1024 * 1024)
            self._metrics.avg_latency_ms = (self._metrics.total_send_time / max(1, self._metrics.messages_sent)) * 1000
            self._metrics.error_rate = (self._metrics.messages_failed / max(1, self._metrics.messages_sent + self._metrics.messages_failed)) * 100
        
        self._last_metrics_update = now
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get producer performance metrics."""
        async with self._lock:
            self._update_throughput_metrics()
            
            return {
                'topic': self._metrics.topic,
                'messages_sent': self._metrics.messages_sent,
                'messages_failed': self._metrics.messages_failed,
                'bytes_sent': self._metrics.bytes_sent,
                'batch_count': self._metrics.batch_count,
                'error_rate_percent': self._metrics.error_rate,
                'throughput_msg_per_sec': self._metrics.throughput_msg_per_sec,
                'throughput_mb_per_sec': self._metrics.throughput_mb_per_sec,
                'avg_latency_ms': self._metrics.avg_latency_ms,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
    
    async def close(self):
        """Close the producer."""
        if self._producer:
            await self._producer.stop()
            logger.info(f"Kafka producer closed for topic: {self.topic}")


class OptimizedKafkaConsumer:
    """
    High-performance Kafka consumer with optimizations for market data processing.
    """
    
    def __init__(
        self,
        topics: Union[str, List[str]],
        group_id: str,
        config: Optional[KafkaOptimizationConfig] = None,
        message_processor: Optional[Callable] = None
    ):
        if not HAS_KAFKA:
            raise ImportError("aiokafka required for Kafka operations")
        
        self.topics = [topics] if isinstance(topics, str) else topics
        self.group_id = group_id
        self.config = config or KafkaOptimizationConfig()
        self.message_processor = message_processor
        
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._metrics = {topic: ConsumerMetrics(topic=topic) for topic in self.topics}
        self._running = False
        self._consumer_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        self.profiler = get_performance_profiler()
    
    async def initialize(self):
        """Initialize the Kafka consumer."""
        try:
            consumer_config = {
                'bootstrap_servers': ','.join(self.config.bootstrap_servers),
                'group_id': self.group_id,
                'value_deserializer': lambda m: json.loads(m.decode('utf-8')) if m else None,
                'key_deserializer': lambda m: m.decode('utf-8') if m else None,
                'fetch_min_bytes': self.config.fetch_min_bytes,
                'fetch_max_wait_ms': self.config.fetch_max_wait_ms,
                'max_partition_fetch_bytes': self.config.max_partition_fetch_bytes,
                'session_timeout_ms': self.config.session_timeout_ms,
                'heartbeat_interval_ms': self.config.heartbeat_interval_ms,
                'max_poll_records': self.config.max_poll_records,
                'auto_offset_reset': self.config.auto_offset_reset,
                'enable_auto_commit': True,
                'auto_commit_interval_ms': 5000,
                'security_protocol': self.config.security_protocol,
            }
            
            # Add authentication if configured
            if self.config.sasl_mechanism:
                consumer_config['sasl_mechanism'] = self.config.sasl_mechanism
                consumer_config['sasl_plain_username'] = self.config.sasl_plain_username
                consumer_config['sasl_plain_password'] = self.config.sasl_plain_password
            
            if self.config.ssl_context:
                consumer_config['ssl_context'] = self.config.ssl_context
            
            self._consumer = AIOKafkaConsumer(*self.topics, **consumer_config)
            await self._consumer.start()
            
            logger.info(f"Kafka consumer initialized for topics: {self.topics} with group: {self.group_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise
    
    async def start_consuming(self, batch_processing: bool = True):
        """Start consuming messages."""
        if not self._consumer:
            raise RuntimeError("Consumer not initialized")
        
        self._running = True
        
        if batch_processing:
            self._consumer_task = asyncio.create_task(self._consume_batch())
        else:
            self._consumer_task = asyncio.create_task(self._consume_single())
        
        logger.info(f"Started consuming messages with batch_processing={batch_processing}")
    
    async def stop_consuming(self):
        """Stop consuming messages."""
        self._running = False
        
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped consuming messages")
    
    async def _consume_single(self):
        """Consume messages one at a time."""
        while self._running:
            try:
                async for message in self._consumer:
                    if not self._running:
                        break
                    
                    await self._process_message(message)
                    
            except Exception as e:
                logger.error(f"Error in single message consumption: {e}")
                await asyncio.sleep(1)  # Back off on error
    
    async def _consume_batch(self):
        """Consume messages in batches for better performance."""
        while self._running:
            try:
                # Get batch of messages
                message_batch = await self._get_message_batch()
                
                if message_batch:
                    # Process batch concurrently
                    tasks = [self._process_message(msg) for msg in message_batch]
                    await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    # No messages - small sleep to prevent busy waiting
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in batch consumption: {e}")
                await asyncio.sleep(1)  # Back off on error
    
    async def _get_message_batch(self, timeout_ms: int = 1000) -> List[Any]:
        """Get a batch of messages."""
        messages = []
        deadline = time.time() + (timeout_ms / 1000)
        
        try:
            data = await asyncio.wait_for(
                self._consumer.getmany(timeout_ms=timeout_ms, max_records=self.config.max_poll_records),
                timeout=timeout_ms / 1000
            )
            
            # Flatten messages from all partitions
            for partition_messages in data.values():
                messages.extend(partition_messages)
                
        except asyncio.TimeoutError:
            pass  # Normal timeout, return what we have
        except Exception as e:
            logger.warning(f"Error getting message batch: {e}")
        
        return messages
    
    async def _process_message(self, message):
        """Process a single message with performance tracking."""
        start_time = time.time()
        topic = message.topic
        message_size = len(message.value) if message.value else 0
        
        try:
            # Profile message processing
            async with self.profiler.db_profiler.profile_query("kafka_consume", f"topic_{topic}"):
                if self.message_processor:
                    await self.message_processor(message)
            
            processing_duration = time.time() - start_time
            
            # Update metrics
            async with self._lock:
                if topic in self._metrics:
                    metrics = self._metrics[topic]
                    metrics.messages_consumed += 1
                    metrics.messages_processed += 1
                    metrics.bytes_consumed += message_size
                    metrics.total_processing_time += processing_duration
                    
                    # Calculate lag
                    if hasattr(message, 'timestamp'):
                        message_timestamp = message.timestamp / 1000  # Convert to seconds
                        current_timestamp = time.time()
                        metrics.lag_ms = (current_timestamp - message_timestamp) * 1000
            
        except Exception as e:
            async with self._lock:
                if topic in self._metrics:
                    self._metrics[topic].messages_failed += 1
            
            logger.error(f"Failed to process message from {topic}: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get consumer performance metrics."""
        async with self._lock:
            metrics_summary = {}
            
            for topic, metrics in self._metrics.items():
                # Calculate throughput metrics
                total_time = metrics.total_processing_time
                if total_time > 0:
                    throughput_msg_per_sec = metrics.messages_processed / total_time
                    throughput_mb_per_sec = (metrics.bytes_consumed / total_time) / (1024 * 1024)
                    avg_processing_time = (total_time / max(1, metrics.messages_processed)) * 1000
                else:
                    throughput_msg_per_sec = 0
                    throughput_mb_per_sec = 0
                    avg_processing_time = 0
                
                metrics_summary[topic] = {
                    'messages_consumed': metrics.messages_consumed,
                    'messages_processed': metrics.messages_processed,
                    'messages_failed': metrics.messages_failed,
                    'bytes_consumed': metrics.bytes_consumed,
                    'lag_ms': metrics.lag_ms,
                    'throughput_msg_per_sec': throughput_msg_per_sec,
                    'throughput_mb_per_sec': throughput_mb_per_sec,
                    'avg_processing_time_ms': avg_processing_time,
                    'error_rate_percent': (metrics.messages_failed / max(1, metrics.messages_consumed)) * 100
                }
            
            return {
                'group_id': self.group_id,
                'topics': metrics_summary,
                'running': self._running,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
    
    async def get_consumer_lag(self) -> Dict[str, Any]:
        """Get detailed consumer lag information."""
        if not self._consumer:
            return {}
        
        try:
            # Get current position for all assigned partitions
            assigned_partitions = self._consumer.assignment()
            lag_info = {}
            
            for partition in assigned_partitions:
                try:
                    # Get current position and high water mark
                    position = await self._consumer.position(partition)
                    high_water_mark = await self._consumer.end_offsets([partition])
                    
                    lag = high_water_mark[partition] - position
                    
                    lag_info[f"{partition.topic}-{partition.partition}"] = {
                        'current_offset': position,
                        'high_water_mark': high_water_mark[partition],
                        'lag': lag
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to get lag for partition {partition}: {e}")
            
            return lag_info
            
        except Exception as e:
            logger.error(f"Failed to get consumer lag: {e}")
            return {}
    
    async def close(self):
        """Close the consumer."""
        await self.stop_consuming()
        
        if self._consumer:
            await self._consumer.stop()
            logger.info(f"Kafka consumer closed for topics: {self.topics}")


class KafkaTopicManager:
    """
    Utility class for managing Kafka topics and optimizations.
    """
    
    def __init__(self, bootstrap_servers: List[str]):
        self.bootstrap_servers = bootstrap_servers
    
    def create_optimized_topic(
        self,
        topic_name: str,
        num_partitions: int = 12,
        replication_factor: int = 1,
        cleanup_policy: str = "delete",
        retention_ms: int = 604800000,  # 7 days
        segment_ms: int = 86400000  # 1 day
    ) -> bool:
        """Create a topic with optimized settings for market data."""
        try:
            from kafka import KafkaAdminClient
            from kafka.admin import NewTopic
            
            admin_client = KafkaAdminClient(
                bootstrap_servers=self.bootstrap_servers,
                client_id='market_analysis_admin'
            )
            
            topic_config = {
                'cleanup.policy': cleanup_policy,
                'retention.ms': str(retention_ms),
                'segment.ms': str(segment_ms),
                'compression.type': 'lz4',
                'min.insync.replicas': '1',
                'unclean.leader.election.enable': 'false',
                'max.message.bytes': '1048588',  # 1MB + overhead
            }
            
            new_topic = NewTopic(
                name=topic_name,
                num_partitions=num_partitions,
                replication_factor=replication_factor,
                topic_configs=topic_config
            )
            
            admin_client.create_topics([new_topic], validate_only=False)
            logger.info(f"Created optimized topic: {topic_name}")
            
            admin_client.close()
            return True
            
        except TopicAlreadyExistsError:
            logger.info(f"Topic {topic_name} already exists")
            return True
        except Exception as e:
            logger.error(f"Failed to create topic {topic_name}: {e}")
            return False


# Factory for creating optimized Kafka clients
class KafkaClientFactory:
    """Factory for creating optimized Kafka clients."""
    
    @staticmethod
    def create_producer(
        topic: str,
        config: Optional[KafkaOptimizationConfig] = None
    ) -> OptimizedKafkaProducer:
        """Create optimized Kafka producer."""
        return OptimizedKafkaProducer(topic, config)
    
    @staticmethod
    def create_consumer(
        topics: Union[str, List[str]],
        group_id: str,
        config: Optional[KafkaOptimizationConfig] = None,
        message_processor: Optional[Callable] = None
    ) -> OptimizedKafkaConsumer:
        """Create optimized Kafka consumer."""
        return OptimizedKafkaConsumer(topics, group_id, config, message_processor)
    
    @staticmethod
    def create_topic_manager(bootstrap_servers: List[str]) -> KafkaTopicManager:
        """Create Kafka topic manager."""
        return KafkaTopicManager(bootstrap_servers)


# Convenient function for symbol-based partitioning
def symbol_partitioner(key_bytes: bytes, all_partitions: List[int], available_partitions: List[int]) -> int:
    """
    Custom partitioner that distributes messages by symbol for better parallelism.
    """
    if not key_bytes:
        # If no key, use round-robin
        import random
        return random.choice(available_partitions)
    
    # Use consistent hashing based on symbol
    symbol = key_bytes.decode('utf-8')
    partition_index = hash(symbol) % len(available_partitions)
    return available_partitions[partition_index]