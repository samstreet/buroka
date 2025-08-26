"""
Comprehensive health check endpoints with database connectivity.
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import asyncio
import time
import os

from ..models.common import HealthCheckResponse

router = APIRouter(prefix="/api/v1", tags=["health"])

# Global health checkers
_health_checkers: Dict[str, Any] = {}


def get_health_checkers() -> Dict[str, Any]:
    """Get or initialize health checkers."""
    global _health_checkers
    
    if not _health_checkers:
        _health_checkers = {
            "postgres": PostgreSQLHealthChecker(),
            "influxdb": InfluxDBHealthChecker(),
            "redis": RedisHealthChecker(),
            "kafka": KafkaHealthChecker(),
            "storage": StorageHealthChecker(),
            "ingestion": IngestionHealthChecker()
        }
    
    return _health_checkers


class BaseHealthChecker:
    """Base health checker class."""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
    
    async def check_health(self) -> Dict[str, Any]:
        """Check health of the component."""
        start_time = time.time()
        
        try:
            if not self.enabled:
                return {
                    "status": "disabled",
                    "message": f"{self.name} health check is disabled",
                    "response_time": 0
                }
            
            result = await self._perform_check()
            response_time = time.time() - start_time
            
            return {
                "status": "healthy" if result.get("healthy", False) else "unhealthy",
                "message": result.get("message", "Health check completed"),
                "response_time": round(response_time, 4),
                "details": result.get("details", {})
            }
        
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "status": "unhealthy",
                "message": f"{self.name} health check failed: {str(e)}",
                "response_time": round(response_time, 4),
                "error": str(e)
            }
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Override this method in subclasses."""
        return {"healthy": True, "message": "Base health check passed"}


class PostgreSQLHealthChecker(BaseHealthChecker):
    """PostgreSQL database health checker."""
    
    def __init__(self):
        super().__init__("PostgreSQL")
        self.enabled = os.getenv("POSTGRES_HOST") is not None
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Check PostgreSQL connectivity."""
        try:
            # Try to import asyncpg
            import asyncpg
        except ImportError:
            return {
                "healthy": False,
                "message": "asyncpg not available",
                "details": {"import_error": "asyncpg module not found"}
            }
        
        try:
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = int(os.getenv("POSTGRES_PORT", "5432"))
            database = os.getenv("POSTGRES_DB", "market_analysis")
            user = os.getenv("POSTGRES_USER", "trader")
            password = os.getenv("POSTGRES_PASSWORD", "")
            
            # Test connection with timeout
            conn = await asyncio.wait_for(
                asyncpg.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password
                ),
                timeout=5.0
            )
            
            # Test simple query
            result = await conn.fetchval("SELECT 1")
            await conn.close()
            
            return {
                "healthy": result == 1,
                "message": "PostgreSQL connection successful",
                "details": {
                    "host": f"{host}:{port}",
                    "database": database,
                    "query_result": result
                }
            }
        
        except asyncio.TimeoutError:
            return {
                "healthy": False,
                "message": "PostgreSQL connection timeout",
                "details": {"timeout": "5 seconds"}
            }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"PostgreSQL connection failed: {str(e)}",
                "details": {"error": str(e)}
            }


class InfluxDBHealthChecker(BaseHealthChecker):
    """InfluxDB health checker."""
    
    def __init__(self):
        super().__init__("InfluxDB")
        self.enabled = os.getenv("INFLUXDB_HOST") is not None
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Check InfluxDB connectivity."""
        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.exceptions import InfluxDBError
        except ImportError:
            return {
                "healthy": False,
                "message": "influxdb-client not available",
                "details": {"import_error": "influxdb_client module not found"}
            }
        
        try:
            url = f"http://{os.getenv('INFLUXDB_HOST', 'localhost')}:{os.getenv('INFLUXDB_PORT', '8086')}"
            token = os.getenv("INFLUXDB_TOKEN", "dev-token")
            org = os.getenv("INFLUXDB_ORG", "market-analysis")
            
            client = InfluxDBClient(url=url, token=token, org=org)
            
            # Test health endpoint
            health = client.health()
            client.close()
            
            return {
                "healthy": health.status == "pass",
                "message": f"InfluxDB health status: {health.status}",
                "details": {
                    "url": url,
                    "org": org,
                    "status": health.status,
                    "message": getattr(health, 'message', 'No message')
                }
            }
        
        except Exception as e:
            return {
                "healthy": False,
                "message": f"InfluxDB connection failed: {str(e)}",
                "details": {"error": str(e)}
            }


class RedisHealthChecker(BaseHealthChecker):
    """Redis health checker."""
    
    def __init__(self):
        super().__init__("Redis")
        self.enabled = os.getenv("REDIS_HOST") is not None
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            import redis.asyncio as redis
        except ImportError:
            return {
                "healthy": False,
                "message": "redis not available",
                "details": {"import_error": "redis module not found"}
            }
        
        try:
            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", "6379"))
            db = int(os.getenv("REDIS_DB", "0"))
            password = os.getenv("REDIS_PASSWORD")
            
            # Create Redis client
            client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )
            
            # Test ping
            pong = await client.ping()
            
            # Test set/get
            test_key = "health_check_test"
            await client.set(test_key, "test_value", ex=10)  # Expire in 10 seconds
            test_value = await client.get(test_key)
            await client.delete(test_key)
            
            await client.close()
            
            return {
                "healthy": pong and test_value == b"test_value",
                "message": "Redis connection successful",
                "details": {
                    "host": f"{host}:{port}",
                    "db": db,
                    "ping": pong,
                    "test_operation": test_value == b"test_value"
                }
            }
        
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Redis connection failed: {str(e)}",
                "details": {"error": str(e)}
            }


class KafkaHealthChecker(BaseHealthChecker):
    """Kafka health checker."""
    
    def __init__(self):
        super().__init__("Kafka")
        self.enabled = os.getenv("KAFKA_BOOTSTRAP_SERVERS") is not None
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Check Kafka connectivity."""
        try:
            from aiokafka import AIOKafkaProducer
        except ImportError:
            return {
                "healthy": False,
                "message": "aiokafka not available",
                "details": {"import_error": "aiokafka module not found"}
            }
        
        try:
            bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
            
            # Create producer to test connectivity
            producer = AIOKafkaProducer(
                bootstrap_servers=bootstrap_servers,
                request_timeout_ms=5000
            )
            
            # Start and get metadata
            await producer.start()
            metadata = await producer.client.fetch_metadata()
            await producer.stop()
            
            brokers = len(metadata.brokers)
            topics = len(metadata.topics)
            
            return {
                "healthy": brokers > 0,
                "message": "Kafka connection successful",
                "details": {
                    "bootstrap_servers": bootstrap_servers,
                    "brokers": brokers,
                    "topics": topics
                }
            }
        
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Kafka connection failed: {str(e)}",
                "details": {"error": str(e)}
            }


class StorageHealthChecker(BaseHealthChecker):
    """Storage service health checker."""
    
    def __init__(self):
        super().__init__("Storage Service")
        self.enabled = True
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Check storage service health."""
        try:
            from ...data.storage.service import DataStorageService
            from ...data.storage.sinks import InMemoryDataSink
            
            # Create minimal storage service for health check
            sink = InMemoryDataSink()
            storage_service = DataStorageService(
                primary_sink=sink,
                batch_size=10,
                batch_timeout=1
            )
            
            # Perform health check
            health = await storage_service.health_check()
            await storage_service.shutdown()
            
            return {
                "healthy": health.get("overall") == "healthy",
                "message": f"Storage service status: {health.get('overall', 'unknown')}",
                "details": health
            }
        
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Storage service check failed: {str(e)}",
                "details": {"error": str(e)}
            }


class IngestionHealthChecker(BaseHealthChecker):
    """Data ingestion service health checker."""
    
    def __init__(self):
        super().__init__("Ingestion Service")
        self.enabled = True
    
    async def _perform_check(self) -> Dict[str, Any]:
        """Check ingestion service health."""
        try:
            from ...data.ingestion.client_factory import get_default_client
            from ...data.ingestion.transformers import MarketDataTransformer
            from ...data.ingestion.metrics import InMemoryMetricsCollector
            from ...data.ingestion.service import DataIngestionService
            from ...data.storage.sinks import InMemoryDataSink
            
            # Create minimal ingestion service for health check
            data_source = get_default_client()
            transformer = MarketDataTransformer()
            sink = InMemoryDataSink()
            metrics = InMemoryMetricsCollector()
            
            service = DataIngestionService(
                data_source=data_source,
                data_transformer=transformer,
                data_sink=sink,
                metrics_collector=metrics
            )
            
            # Perform health check
            health = await service.health_check()
            await service.shutdown()
            
            return {
                "healthy": health.get("overall") == "healthy",
                "message": f"Ingestion service status: {health.get('overall', 'unknown')}",
                "details": {
                    "components": len(health.get("components", {})),
                    "status": health.get("overall")
                }
            }
        
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Ingestion service check failed: {str(e)}",
                "details": {"error": str(e)}
            }


@router.get("/health", summary="Comprehensive Health Check", response_model=HealthCheckResponse)
async def comprehensive_health_check(
    checkers: Dict[str, Any] = Depends(get_health_checkers)
) -> HealthCheckResponse:
    """
    Perform comprehensive health check on all system components.
    
    Checks the health of:
    - PostgreSQL database
    - InfluxDB time series database  
    - Redis cache
    - Kafka message queue
    - Storage service
    - Data ingestion service
    
    Returns:
        HealthCheckResponse with overall status and component details
    """
    start_time = time.time()
    
    # Run all health checks in parallel
    health_tasks = {
        name: asyncio.create_task(checker.check_health()) 
        for name, checker in checkers.items()
    }
    
    # Wait for all checks to complete
    health_results = {}
    for name, task in health_tasks.items():
        try:
            health_results[name] = await task
        except Exception as e:
            health_results[name] = {
                "status": "error",
                "message": f"Health check task failed: {str(e)}",
                "response_time": 0,
                "error": str(e)
            }
    
    # Determine overall status
    healthy_count = sum(1 for result in health_results.values() if result["status"] == "healthy")
    disabled_count = sum(1 for result in health_results.values() if result["status"] == "disabled")
    total_enabled = len(health_results) - disabled_count
    
    if healthy_count == total_enabled:
        overall_status = "healthy"
    elif healthy_count > 0:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    # Calculate total response time
    total_response_time = time.time() - start_time
    
    # Add total response time to components instead of checks
    health_results["_system"] = {
        "status": overall_status,
        "response_time": round(total_response_time, 4),
        "healthy_components": healthy_count,
        "total_components": total_enabled
    }
    
    return HealthCheckResponse(
        status=overall_status,
        version="0.1.0",
        timestamp=datetime.now(timezone.utc),
        components=health_results,
        checks={
            "database_connections": any(
                result["status"] == "healthy" 
                for name, result in health_results.items() 
                if name in ["postgres", "influxdb", "redis"]
            ),
            "message_queue": health_results.get("kafka", {}).get("status") == "healthy",
            "core_services": all(
                result["status"] in ["healthy", "disabled"]
                for name, result in health_results.items()
                if name in ["storage", "ingestion"]
            )
        }
    )


@router.get("/health/simple", summary="Simple Health Check")
async def simple_health_check() -> Dict[str, Any]:
    """
    Simple health check endpoint for basic monitoring.
    
    Returns basic system status without database connectivity checks.
    Useful for load balancers and simple monitoring systems.
    
    Returns:
        Dict with basic health information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.0",
        "environment": os.getenv("DEBUG", "production"),
        "uptime": "Not implemented"  # Would track service start time
    }


@router.get("/health/ready", summary="Readiness Check")
async def readiness_check(
    checkers: Dict[str, Any] = Depends(get_health_checkers)
) -> Dict[str, Any]:
    """
    Kubernetes-style readiness probe.
    
    Checks if the service is ready to handle requests.
    Returns 200 if ready, 503 if not ready.
    
    Returns:
        Dict with readiness status
    """
    # Check critical components only
    critical_checks = ["storage", "ingestion"]
    
    ready = True
    check_results = {}
    
    for name in critical_checks:
        if name in checkers:
            try:
                result = await checkers[name].check_health()
                check_results[name] = result["status"]
                if result["status"] not in ["healthy", "disabled"]:
                    ready = False
            except Exception as e:
                check_results[name] = "error"
                ready = False
    
    if not ready:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "status": "ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": check_results
    }


@router.get("/health/live", summary="Liveness Check")
async def liveness_check() -> Dict[str, Any]:
    """
    Kubernetes-style liveness probe.
    
    Checks if the service is alive and should not be restarted.
    This is a minimal check that should always succeed unless
    the service is completely broken.
    
    Returns:
        Dict with liveness status
    """
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "process_id": os.getpid()
    }