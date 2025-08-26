"""
Redis client utilities for rate limiting and caching.
"""

import os
import asyncio
import logging
from typing import Optional, Union, Dict, Any
from contextlib import asynccontextmanager
import json
import time

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Async Redis client for rate limiting and caching operations.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 10,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        
        self._pool: Optional[Any] = None
        self._client: Optional[Any] = None
        self._connected = False
        
    async def connect(self) -> None:
        """Establish Redis connection."""
        if self._connected:
            return
            
        try:
            import redis.asyncio as redis
        except ImportError:
            logger.error("redis package not available. Install with: pip install redis")
            raise ImportError("redis package required for rate limiting")
        
        try:
            # Create connection pool
            self._pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                decode_responses=True
            )
            
            # Create Redis client
            self._client = redis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}/{self.db}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            raise
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
        self._connected = False
        logger.info("Disconnected from Redis")
    
    async def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if not self._connected or not self._client:
            return False
        
        try:
            await self._client.ping()
            return True
        except Exception:
            self._connected = False
            return False
    
    @asynccontextmanager
    async def ensure_connected(self):
        """Context manager to ensure Redis connection."""
        if not await self.is_connected():
            await self.connect()
        yield self._client
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        async with self.ensure_connected() as client:
            return await client.get(key)
    
    async def set(
        self, 
        key: str, 
        value: Union[str, int, float], 
        ex: Optional[int] = None,
        px: Optional[int] = None
    ) -> bool:
        """Set value in Redis with optional expiration."""
        async with self.ensure_connected() as client:
            return await client.set(key, value, ex=ex, px=px)
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment counter in Redis."""
        async with self.ensure_connected() as client:
            return await client.incr(key, amount)
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on key."""
        async with self.ensure_connected() as client:
            return await client.expire(key, seconds)
    
    async def delete(self, *keys: str) -> int:
        """Delete keys from Redis."""
        async with self.ensure_connected() as client:
            return await client.delete(*keys)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        async with self.ensure_connected() as client:
            return bool(await client.exists(key))
    
    async def ttl(self, key: str) -> int:
        """Get TTL of key."""
        async with self.ensure_connected() as client:
            return await client.ttl(key)
    
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field."""
        async with self.ensure_connected() as client:
            return await client.hget(name, key)
    
    async def hset(self, name: str, key: str, value: Union[str, int, float]) -> int:
        """Set hash field."""
        async with self.ensure_connected() as client:
            return await client.hset(name, key, value)
    
    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields."""
        async with self.ensure_connected() as client:
            return await client.hgetall(name)
    
    async def hincrby(self, name: str, key: str, amount: int = 1) -> int:
        """Increment hash field."""
        async with self.ensure_connected() as client:
            return await client.hincrby(name, key, amount)


class RateLimiter:
    """
    Token bucket rate limiter using Redis.
    """
    
    def __init__(
        self,
        redis_client: RedisClient,
        requests_per_hour: int = 1000,
        burst_size: Optional[int] = None
    ):
        self.redis = redis_client
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size or min(requests_per_hour // 10, 100)
        
        # Calculate tokens per second
        self.tokens_per_second = requests_per_hour / 3600.0
        self.bucket_capacity = self.burst_size
    
    def _get_bucket_key(self, identifier: str) -> str:
        """Get Redis key for rate limit bucket."""
        return f"ratelimit:bucket:{identifier}"
    
    def _get_stats_key(self, identifier: str) -> str:
        """Get Redis key for rate limit stats."""
        return f"ratelimit:stats:{identifier}"
    
    async def is_allowed(self, identifier: str) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed under rate limit.
        
        Args:
            identifier: Unique identifier (IP address, user ID, API key)
            
        Returns:
            Tuple of (allowed, metadata)
        """
        now = time.time()
        bucket_key = self._get_bucket_key(identifier)
        stats_key = self._get_stats_key(identifier)
        
        try:
            async with self.redis.ensure_connected() as client:
                # Get current bucket state
                bucket_data = await client.hgetall(bucket_key)
                
                if bucket_data:
                    tokens = float(bucket_data.get("tokens", 0))
                    last_refill = float(bucket_data.get("last_refill", now))
                else:
                    # Initialize bucket
                    tokens = self.bucket_capacity
                    last_refill = now
                
                # Calculate tokens to add
                time_passed = now - last_refill
                tokens_to_add = time_passed * self.tokens_per_second
                tokens = min(self.bucket_capacity, tokens + tokens_to_add)
                
                # Check if request is allowed
                if tokens >= 1.0:
                    # Allow request
                    tokens -= 1.0
                    allowed = True
                    
                    # Update bucket
                    await client.hset(bucket_key, "tokens", tokens)
                    await client.hset(bucket_key, "last_refill", now)
                    await client.expire(bucket_key, 7200)  # 2 hours TTL
                    
                    # Update stats
                    await client.hincrby(stats_key, "requests", 1)
                    await client.hincrby(stats_key, "allowed", 1)
                    await client.expire(stats_key, 7200)
                    
                else:
                    # Reject request
                    allowed = False
                    
                    # Update stats
                    await client.hincrby(stats_key, "requests", 1)
                    await client.hincrby(stats_key, "rejected", 1)
                    await client.expire(stats_key, 7200)
                
                # Calculate reset time
                if tokens < 1.0:
                    reset_time = int(now + ((1.0 - tokens) / self.tokens_per_second))
                else:
                    reset_time = int(now)
                
                metadata = {
                    "limit": self.requests_per_hour,
                    "remaining": int(tokens),
                    "reset": reset_time,
                    "reset_after": max(0, reset_time - int(now)),
                    "bucket_capacity": self.bucket_capacity
                }
                
                return allowed, metadata
                
        except Exception as e:
            logger.error(f"Rate limiter error for {identifier}: {e}")
            # Fail open - allow request if Redis is down
            return True, {
                "limit": self.requests_per_hour,
                "remaining": self.requests_per_hour,
                "reset": int(now + 3600),
                "reset_after": 3600,
                "error": "rate_limiter_unavailable"
            }
    
    async def get_stats(self, identifier: str) -> Dict[str, Any]:
        """Get rate limit statistics for identifier."""
        stats_key = self._get_stats_key(identifier)
        bucket_key = self._get_bucket_key(identifier)
        
        try:
            async with self.redis.ensure_connected() as client:
                stats = await client.hgetall(stats_key)
                bucket = await client.hgetall(bucket_key)
                
                return {
                    "identifier": identifier,
                    "requests": int(stats.get("requests", 0)),
                    "allowed": int(stats.get("allowed", 0)),
                    "rejected": int(stats.get("rejected", 0)),
                    "current_tokens": float(bucket.get("tokens", self.bucket_capacity)),
                    "bucket_capacity": self.bucket_capacity,
                    "requests_per_hour": self.requests_per_hour
                }
        except Exception as e:
            logger.error(f"Failed to get stats for {identifier}: {e}")
            return {"error": str(e)}
    
    async def reset_limit(self, identifier: str) -> bool:
        """Reset rate limit for identifier."""
        bucket_key = self._get_bucket_key(identifier)
        stats_key = self._get_stats_key(identifier)
        
        try:
            async with self.redis.ensure_connected() as client:
                await client.delete(bucket_key, stats_key)
                return True
        except Exception as e:
            logger.error(f"Failed to reset limit for {identifier}: {e}")
            return False


# Global Redis client instance
_redis_client: Optional[RedisClient] = None


def get_redis_client() -> RedisClient:
    """Get global Redis client instance."""
    global _redis_client
    
    if _redis_client is None:
        _redis_client = RedisClient(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            socket_timeout=float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0")),
            socket_connect_timeout=float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5.0"))
        )
    
    return _redis_client


async def init_redis():
    """Initialize Redis connection on startup."""
    redis_client = get_redis_client()
    await redis_client.connect()


async def close_redis():
    """Close Redis connection on shutdown."""
    global _redis_client
    if _redis_client:
        await _redis_client.disconnect()
        _redis_client = None