"""
Optimized database clients with connection pooling and query optimization.
Implements SOLID principles for scalable database operations.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import json

try:
    import asyncpg
    import asyncpg.pool
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

try:
    import aioredis
    HAS_AIOREDIS = True
except ImportError:
    HAS_AIOREDIS = False

try:
    from influxdb_client import InfluxDBClient
    from influxdb_client.client.write_api import ASYNCHRONOUS
    HAS_INFLUXDB = True
except ImportError:
    HAS_INFLUXDB = False

from src.utils.performance_profiler import get_performance_profiler

logger = logging.getLogger(__name__)


@dataclass
class ConnectionPoolConfig:
    """Configuration for database connection pools."""
    min_connections: int = 5
    max_connections: int = 20
    max_queries: int = 50000
    max_inactive_connection_lifetime: float = 300.0  # 5 minutes
    command_timeout: float = 60.0
    connection_timeout: float = 10.0
    
    # Query optimization settings
    statement_cache_size: int = 100
    prepared_statement_cache_size: int = 100


@dataclass
class QueryMetrics:
    """Query performance metrics."""
    query_signature: str
    execution_count: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    last_executed: Optional[datetime] = None


class OptimizedPostgreSQLClient:
    """
    High-performance PostgreSQL client with connection pooling and query optimization.
    """
    
    def __init__(
        self,
        dsn: str,
        pool_config: Optional[ConnectionPoolConfig] = None,
        enable_query_caching: bool = True,
        cache_ttl: int = 300
    ):
        self.dsn = dsn
        self.pool_config = pool_config or ConnectionPoolConfig()
        self.enable_query_caching = enable_query_caching
        self.cache_ttl = cache_ttl
        
        self._pool: Optional[asyncpg.pool.Pool] = None
        self._query_cache: Dict[str, Any] = {}
        self._query_metrics: Dict[str, QueryMetrics] = {}
        self._prepared_statements: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        
        self.profiler = get_performance_profiler()
    
    async def initialize(self):
        """Initialize connection pool."""
        if not HAS_ASYNCPG:
            raise ImportError("asyncpg required for PostgreSQL client")
        
        try:
            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=self.pool_config.min_connections,
                max_size=self.pool_config.max_connections,
                max_queries=self.pool_config.max_queries,
                max_inactive_connection_lifetime=self.pool_config.max_inactive_connection_lifetime,
                command_timeout=self.pool_config.command_timeout,
                server_settings={
                    'application_name': 'market_analysis_system',
                    'jit': 'off',  # Disable JIT for consistent performance
                    'shared_preload_libraries': 'pg_stat_statements',
                }
            )
            
            logger.info(f"PostgreSQL connection pool initialized: {self.pool_config.min_connections}-{self.pool_config.max_connections} connections")
            
            # Initialize prepared statements for common queries
            await self._prepare_common_statements()
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise
    
    async def _prepare_common_statements(self):
        """Prepare commonly used SQL statements."""
        common_statements = {
            "select_market_data": """
                SELECT symbol, timestamp, open_price, high_price, low_price, close_price, volume
                FROM market_data 
                WHERE symbol = $1 AND timestamp >= $2 AND timestamp <= $3
                ORDER BY timestamp DESC
            """,
            "insert_market_data": """
                INSERT INTO market_data (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume = EXCLUDED.volume
            """,
            "select_technical_indicators": """
                SELECT symbol, indicator_type, value, timestamp
                FROM technical_indicators
                WHERE symbol = $1 AND indicator_type = $2 AND timestamp >= $3
                ORDER BY timestamp DESC
            """
        }
        
        if self._pool:
            async with self._pool.acquire() as conn:
                for name, statement in common_statements.items():
                    try:
                        await conn.prepare(statement)
                        self._prepared_statements[name] = statement
                        logger.debug(f"Prepared statement: {name}")
                    except Exception as e:
                        logger.warning(f"Failed to prepare statement {name}: {e}")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool."""
        if not self._pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self._pool.acquire() as connection:
            yield connection
    
    async def execute_query(
        self,
        query: str,
        *args,
        fetch_mode: str = "all",  # all, one, none
        use_cache: bool = True,
        query_signature: Optional[str] = None
    ) -> Any:
        """
        Execute optimized database query with caching and performance monitoring.
        """
        start_time = time.time()
        signature = query_signature or self._get_query_signature(query, args)
        
        # Check cache first
        if use_cache and self.enable_query_caching and fetch_mode != "none":
            cached_result = self._get_cached_result(signature)
            if cached_result is not None:
                await self._update_query_metrics(signature, 0, True, True)
                return cached_result
        
        try:
            # Profile database query
            async with self.profiler.db_profiler.profile_query("postgresql", signature):
                async with self.get_connection() as conn:
                    if fetch_mode == "all":
                        result = await conn.fetch(query, *args)
                    elif fetch_mode == "one":
                        result = await conn.fetchrow(query, *args)
                    elif fetch_mode == "none":
                        result = await conn.execute(query, *args)
                    else:
                        raise ValueError(f"Invalid fetch_mode: {fetch_mode}")
            
            duration = time.time() - start_time
            
            # Cache result if appropriate
            if (use_cache and self.enable_query_caching and 
                fetch_mode != "none" and duration < 1.0):  # Only cache fast queries
                self._cache_result(signature, result)
            
            await self._update_query_metrics(signature, duration, True, False)
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            await self._update_query_metrics(signature, duration, False, False)
            logger.error(f"Query execution failed: {e}")
            raise
    
    async def execute_batch(
        self,
        query: str,
        args_list: List[tuple],
        batch_size: int = 1000
    ) -> int:
        """Execute batch operations efficiently."""
        total_affected = 0
        
        for i in range(0, len(args_list), batch_size):
            batch = args_list[i:i + batch_size]
            
            try:
                async with self.get_connection() as conn:
                    async with conn.transaction():
                        for args in batch:
                            result = await conn.execute(query, *args)
                            # Extract affected rows count if available
                            if hasattr(result, 'split') and result.startswith(('INSERT', 'UPDATE', 'DELETE')):
                                total_affected += int(result.split()[-1])
                        
                logger.debug(f"Executed batch of {len(batch)} operations")
                
            except Exception as e:
                logger.error(f"Batch execution failed: {e}")
                raise
        
        return total_affected
    
    def _get_query_signature(self, query: str, args: tuple) -> str:
        """Generate query signature for caching and metrics."""
        # Normalize query by removing extra whitespace and converting to lowercase
        normalized = ' '.join(query.strip().lower().split())
        # Include argument types for signature (not values for security)
        arg_types = [type(arg).__name__ for arg in args]
        return f"{normalized}:{':'.join(arg_types)}"
    
    def _get_cached_result(self, signature: str) -> Optional[Any]:
        """Get cached query result if available and not expired."""
        if signature in self._query_cache:
            cached_data = self._query_cache[signature]
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                return cached_data['result']
            else:
                # Remove expired cache entry
                del self._query_cache[signature]
        return None
    
    def _cache_result(self, signature: str, result: Any):
        """Cache query result."""
        # Convert asyncpg records to serializable format
        if hasattr(result, '__iter__'):
            try:
                serializable_result = [dict(row) if hasattr(row, 'keys') else row for row in result]
            except:
                serializable_result = result
        else:
            serializable_result = dict(result) if hasattr(result, 'keys') else result
        
        self._query_cache[signature] = {
            'result': serializable_result,
            'timestamp': time.time()
        }
        
        # Limit cache size
        if len(self._query_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(self._query_cache.keys(), 
                               key=lambda k: self._query_cache[k]['timestamp'])[:100]
            for key in oldest_keys:
                del self._query_cache[key]
    
    async def _update_query_metrics(
        self,
        signature: str,
        duration: float,
        success: bool,
        cache_hit: bool
    ):
        """Update query performance metrics."""
        async with self._lock:
            if signature not in self._query_metrics:
                self._query_metrics[signature] = QueryMetrics(signature)
            
            metrics = self._query_metrics[signature]
            metrics.execution_count += 1
            metrics.last_executed = datetime.now(timezone.utc)
            
            if cache_hit:
                metrics.cache_hits += 1
            else:
                metrics.cache_misses += 1
                
                if success:
                    metrics.total_duration += duration
                    metrics.avg_duration = metrics.total_duration / (metrics.execution_count - metrics.cache_hits)
                    metrics.min_duration = min(metrics.min_duration, duration)
                    metrics.max_duration = max(metrics.max_duration, duration)
                else:
                    metrics.error_count += 1
    
    async def get_query_performance_stats(self) -> Dict[str, Any]:
        """Get query performance statistics."""
        async with self._lock:
            stats = {}
            for signature, metrics in self._query_metrics.items():
                stats[signature] = {
                    "execution_count": metrics.execution_count,
                    "avg_duration_ms": metrics.avg_duration * 1000 if metrics.avg_duration > 0 else 0,
                    "min_duration_ms": metrics.min_duration * 1000 if metrics.min_duration != float('inf') else 0,
                    "max_duration_ms": metrics.max_duration * 1000,
                    "error_count": metrics.error_count,
                    "error_rate": (metrics.error_count / max(1, metrics.execution_count)) * 100,
                    "cache_hit_rate": (metrics.cache_hits / max(1, metrics.execution_count)) * 100,
                    "last_executed": metrics.last_executed.isoformat() if metrics.last_executed else None
                }
            
            return {
                "query_stats": stats,
                "cache_size": len(self._query_cache),
                "pool_stats": await self._get_pool_stats() if self._pool else {}
            }
    
    async def _get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        if not self._pool:
            return {}
        
        return {
            "size": self._pool.get_size(),
            "min_size": self._pool.get_min_size(),
            "max_size": self._pool.get_max_size(),
            "idle_count": self._pool.get_idle_size(),
            "used_count": self._pool.get_size() - self._pool.get_idle_size()
        }
    
    async def optimize_tables(self, table_names: Optional[List[str]] = None):
        """Run table optimization operations."""
        tables_to_optimize = table_names or ['market_data', 'technical_indicators', 'patterns']
        
        async with self.get_connection() as conn:
            for table in tables_to_optimize:
                try:
                    # Update table statistics
                    await conn.execute(f"ANALYZE {table}")
                    
                    # Consider auto-vacuum settings
                    result = await conn.fetchrow(
                        "SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del "
                        "FROM pg_stat_user_tables WHERE tablename = $1", table
                    )
                    
                    if result:
                        logger.info(f"Table {table} stats: inserts={result['n_tup_ins']}, "
                                  f"updates={result['n_tup_upd']}, deletes={result['n_tup_del']}")
                
                except Exception as e:
                    logger.warning(f"Failed to optimize table {table}: {e}")
    
    async def close(self):
        """Close connection pool and cleanup resources."""
        if self._pool:
            await self._pool.close()
            logger.info("PostgreSQL connection pool closed")
        
        self._query_cache.clear()
        self._query_metrics.clear()


class OptimizedInfluxDBClient:
    """
    High-performance InfluxDB client for time-series data.
    """
    
    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        bucket: str,
        batch_size: int = 5000,
        flush_interval: int = 10000  # ms
    ):
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        self._client: Optional[InfluxDBClient] = None
        self._write_api = None
        self._query_api = None
        self._batch_points: List[str] = []
        self._last_flush = time.time()
        self._lock = asyncio.Lock()
        
        self.profiler = get_performance_profiler()
    
    async def initialize(self):
        """Initialize InfluxDB client."""
        if not HAS_INFLUXDB:
            raise ImportError("influxdb-client required for InfluxDB operations")
        
        try:
            self._client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org,
                timeout=30000,  # 30 seconds
                enable_gzip=True
            )
            
            self._write_api = self._client.write_api(
                write_options=ASYNCHRONOUS,
                batch_size=self.batch_size,
                flush_interval=self.flush_interval,
                jitter_interval=2000,
                retry_interval=5000,
                max_retries=3,
                max_retry_delay=30000,
                exponential_base=2
            )
            
            self._query_api = self._client.query_api()
            
            # Test connection
            health = self._client.health()
            if health.status == "pass":
                logger.info("InfluxDB client initialized successfully")
            else:
                raise Exception(f"InfluxDB health check failed: {health.message}")
        
        except Exception as e:
            logger.error(f"Failed to initialize InfluxDB client: {e}")
            raise
    
    async def write_market_data(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: int,
        additional_tags: Optional[Dict[str, str]] = None
    ):
        """Write market data point efficiently."""
        tags = {"symbol": symbol}
        if additional_tags:
            tags.update(additional_tags)
        
        fields = {
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume
        }
        
        point = f"market_data,{','.join([f'{k}={v}' for k, v in tags.items()])} " \
                f"{','.join([f'{k}={v}' for k, v in fields.items()])} " \
                f"{int(timestamp.timestamp() * 1000000000)}"
        
        async with self._lock:
            self._batch_points.append(point)
            
            # Flush if batch is full or time interval exceeded
            if (len(self._batch_points) >= self.batch_size or
                time.time() - self._last_flush > self.flush_interval / 1000):
                await self._flush_batch()
    
    async def _flush_batch(self):
        """Flush batched points to InfluxDB."""
        if not self._batch_points:
            return
        
        try:
            points_to_write = self._batch_points.copy()
            self._batch_points.clear()
            self._last_flush = time.time()
            
            # Use line protocol for maximum performance
            line_protocol = '\n'.join(points_to_write)
            
            async with self.profiler.db_profiler.profile_query("influxdb_write", f"batch_size_{len(points_to_write)}"):
                self._write_api.write(bucket=self.bucket, record=line_protocol)
            
            logger.debug(f"Flushed {len(points_to_write)} points to InfluxDB")
            
        except Exception as e:
            logger.error(f"Failed to flush batch to InfluxDB: {e}")
            # Re-add points for retry (with limit to prevent memory issues)
            if len(self._batch_points) < self.batch_size * 2:
                self._batch_points.extend(points_to_write[-self.batch_size:])
    
    async def query_market_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        aggregation_window: str = "1m"
    ) -> List[Dict[str, Any]]:
        """Query market data with optimization."""
        
        flux_query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r._measurement == "market_data")
                |> filter(fn: (r) => r.symbol == "{symbol}")
                |> aggregateWindow(every: {aggregation_window}, fn: last, createEmpty: false)
                |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        try:
            async with self.profiler.db_profiler.profile_query("influxdb_query", f"symbol_{symbol}"):
                tables = self._query_api.query(flux_query, org=self.org)
            
            results = []
            for table in tables:
                for record in table.records:
                    results.append({
                        "timestamp": record.get_time(),
                        "symbol": record.values.get("symbol"),
                        "open": record.values.get("open"),
                        "high": record.values.get("high"),
                        "low": record.values.get("low"),
                        "close": record.values.get("close"),
                        "volume": record.values.get("volume")
                    })
            
            return results
        
        except Exception as e:
            logger.error(f"InfluxDB query failed: {e}")
            raise
    
    async def close(self):
        """Close InfluxDB client and flush remaining data."""
        async with self._lock:
            if self._batch_points:
                await self._flush_batch()
        
        if self._client:
            self._client.close()
            logger.info("InfluxDB client closed")


class OptimizedRedisClient:
    """
    High-performance Redis client with connection pooling.
    """
    
    def __init__(
        self,
        redis_url: str,
        pool_size: int = 20,
        max_connections: int = 50,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0
    ):
        self.redis_url = redis_url
        self.pool_size = pool_size
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        
        self._pool: Optional[aioredis.ConnectionPool] = None
        self._redis: Optional[aioredis.Redis] = None
        
        self.profiler = get_performance_profiler()
    
    async def initialize(self):
        """Initialize Redis connection pool."""
        if not HAS_AIOREDIS:
            raise ImportError("aioredis required for Redis operations")
        
        try:
            self._pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                decode_responses=True
            )
            
            self._redis = aioredis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._redis.ping()
            logger.info("Redis connection pool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis pool: {e}")
            raise
    
    @asynccontextmanager
    async def pipeline(self):
        """Get Redis pipeline for batch operations."""
        pipe = self._redis.pipeline()
        try:
            yield pipe
        finally:
            await pipe.execute()
    
    async def cache_market_data(
        self,
        symbol: str,
        data: Dict[str, Any],
        ttl: int = 300
    ):
        """Cache market data with TTL."""
        key = f"market_data:{symbol}"
        
        async with self.profiler.db_profiler.profile_query("redis_write", f"cache_{symbol}"):
            await self._redis.setex(key, ttl, json.dumps(data, default=str))
    
    async def get_cached_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached market data."""
        key = f"market_data:{symbol}"
        
        try:
            async with self.profiler.db_profiler.profile_query("redis_read", f"cache_{symbol}"):
                cached_data = await self._redis.get(key)
            
            if cached_data:
                return json.loads(cached_data)
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get cached data for {symbol}: {e}")
            return None
    
    async def close(self):
        """Close Redis connection pool."""
        if self._redis:
            await self._redis.close()
        if self._pool:
            await self._pool.disconnect()
        logger.info("Redis connection pool closed")


# Factory for creating optimized database clients
class DatabaseClientFactory:
    """Factory for creating optimized database clients."""
    
    @staticmethod
    def create_postgresql_client(
        dsn: str,
        pool_config: Optional[ConnectionPoolConfig] = None,
        enable_query_caching: bool = True
    ) -> OptimizedPostgreSQLClient:
        """Create optimized PostgreSQL client."""
        return OptimizedPostgreSQLClient(dsn, pool_config, enable_query_caching)
    
    @staticmethod
    def create_influxdb_client(
        url: str,
        token: str,
        org: str,
        bucket: str,
        batch_size: int = 5000
    ) -> OptimizedInfluxDBClient:
        """Create optimized InfluxDB client."""
        return OptimizedInfluxDBClient(url, token, org, bucket, batch_size)
    
    @staticmethod
    def create_redis_client(
        redis_url: str,
        pool_size: int = 20
    ) -> OptimizedRedisClient:
        """Create optimized Redis client."""
        return OptimizedRedisClient(redis_url, pool_size)