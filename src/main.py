"""
Market Analysis System - Main FastAPI Application
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import os
from datetime import datetime
from typing import Dict, Any

# Import middleware
from src.api.middleware.security import (
    SecurityHeadersMiddleware, 
    RateLimitMiddleware, 
    RequestLoggingMiddleware,
    APIKeyMiddleware
)

# Import API routers
try:
    from src.api.indicators import router as indicators_router
    HAS_INDICATORS = True
except ImportError:
    HAS_INDICATORS = False
    print("⚠️  Indicators API not available")

try:
    from src.api.routers.monitoring import router as monitoring_router
    HAS_MONITORING = True
except ImportError:
    HAS_MONITORING = False
    print("⚠️  Monitoring API not available")

# Try to import configuration, fallback if not available
try:
    from config import get_settings, get_api_settings
    settings = get_settings()
    api_settings = get_api_settings()
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    print("⚠️  Configuration module not available, using defaults")

try:
    from src.api.routers.storage import router as storage_router
    HAS_STORAGE = True
except ImportError:
    HAS_STORAGE = False
    print("⚠️  Storage API not available")

try:
    from src.api.routers.auth import router as auth_router
    HAS_AUTH = True
except ImportError:
    HAS_AUTH = False
    print("⚠️  Authentication API not available")

try:
    from src.api.routers.health import router as health_router
    HAS_HEALTH = True
except ImportError:
    HAS_HEALTH = False
    print("⚠️  Health API not available")

try:
    from src.api.routers.market_data import router as market_data_router
    HAS_MARKET_DATA = True
except ImportError:
    HAS_MARKET_DATA = False
    print("⚠️  Market Data API not available")

try:
    from src.api.routers.paginated_data import router as paginated_data_router
    HAS_PAGINATED_DATA = True
except ImportError:
    HAS_PAGINATED_DATA = False
    print("⚠️  Paginated Data API not available")

# Create FastAPI application
app = FastAPI(
    title="Market Analysis System",
    description="Real-time market analysis platform for trading insights",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    debug=os.getenv("DEBUG", "false").lower() == "true"
)

# Add middleware (order matters - last added is executed first)
# Security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Rate limiting middleware
app.add_middleware(RateLimitMiddleware, requests_per_minute=1000)

# Request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# API key middleware (if keys are configured)
api_keys = {}  # In production, load from environment or database
if api_keys:
    app.add_middleware(APIKeyMiddleware, api_keys=api_keys)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
)

@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with basic system information."""
    return {
        "message": "Market Analysis System API",
        "version": "0.1.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": os.getenv("DEBUG", "false")
    }

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for monitoring."""
    
    # Basic health check
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "environment": {
            "debug": os.getenv("DEBUG", "false"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        },
        "services": {}
    }
    
    # Check database connections (basic checks for now)
    try:
        if HAS_CONFIG:
            db_settings = settings.database
            kafka_settings = settings.kafka
            
            # PostgreSQL check
            health_status["services"]["postgres"] = {
                "status": "configured",
                "host": f"{db_settings.postgres_host}:{db_settings.postgres_port}",
                "database": db_settings.postgres_db
            }
            
            # InfluxDB check
            health_status["services"]["influxdb"] = {
                "status": "configured",
                "host": f"{db_settings.influxdb_host}:{db_settings.influxdb_port}",
                "org": db_settings.influxdb_org,
                "bucket": db_settings.influxdb_bucket
            }
            
            # Redis check
            health_status["services"]["redis"] = {
                "status": "configured",
                "host": f"{db_settings.redis_host}:{db_settings.redis_port}",
                "database": db_settings.redis_db
            }
            
            # Kafka check
            health_status["services"]["kafka"] = {
                "status": "configured",
                "servers": kafka_settings.bootstrap_servers,
                "topic_prefix": kafka_settings.topic_prefix
            }
        else:
            # Use environment variables directly when config is not available
            health_status["services"]["postgres"] = {
                "status": "configured",
                "host": f"{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}",
                "database": os.getenv('POSTGRES_DB', 'market_analysis')
            }
            
            health_status["services"]["influxdb"] = {
                "status": "configured", 
                "host": f"{os.getenv('INFLUXDB_HOST', 'localhost')}:{os.getenv('INFLUXDB_PORT', '8086')}",
                "org": os.getenv('INFLUXDB_ORG', 'market-analysis'),
                "bucket": os.getenv('INFLUXDB_BUCKET', 'market-data')
            }
            
            health_status["services"]["redis"] = {
                "status": "configured",
                "host": f"{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}",
                "database": os.getenv('REDIS_DB', '0')
            }
            
            health_status["services"]["kafka"] = {
                "status": "configured",
                "servers": os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
                "topic_prefix": os.getenv('KAFKA_TOPIC_PREFIX', 'market_')
            }
        
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["error"] = str(e)
    
    return health_status

@app.get("/api/v1/info")
async def system_info() -> Dict[str, Any]:
    """Get detailed system information."""
    return {
        "system": {
            "name": "Market Analysis System",
            "version": "0.1.0",
            "environment": os.getenv("DEBUG", "false"),
            "uptime": "Just started",  # TODO: Calculate actual uptime
        },
        "features": {
            "data_ingestion": "planned",
            "pattern_detection": "planned",
            "real_time_analysis": "planned",
            "machine_learning": "planned",
            "api_endpoints": "basic"
        },
        "configuration": {
            "databases": {
                "postgres": f"{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}",
                "influxdb": f"{os.getenv('INFLUXDB_HOST', 'localhost')}:{os.getenv('INFLUXDB_PORT', '8086')}",
                "redis": f"{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}"
            },
            "message_queue": {
                "kafka": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
            }
        }
    }

@app.get("/api/v1/test")
async def test_endpoint() -> Dict[str, Any]:
    """Test endpoint for development purposes."""
    return {
        "message": "Test endpoint working",
        "timestamp": datetime.utcnow().isoformat(),
        "environment_variables": {
            "DEBUG": os.getenv("DEBUG"),
            "LOG_LEVEL": os.getenv("LOG_LEVEL"),
            "POSTGRES_HOST": os.getenv("POSTGRES_HOST"),
            "INFLUXDB_HOST": os.getenv("INFLUXDB_HOST"),
            "REDIS_HOST": os.getenv("REDIS_HOST"),
            "KAFKA_BOOTSTRAP_SERVERS": os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The requested resource was not found",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Include API routers
if HAS_INDICATORS:
    app.include_router(indicators_router)
    print("✅ Technical Indicators API loaded")

if HAS_MONITORING:
    app.include_router(monitoring_router)
    print("✅ Monitoring API loaded")

if HAS_STORAGE:
    app.include_router(storage_router)
    print("✅ Storage API loaded")

if HAS_AUTH:
    app.include_router(auth_router)
    print("✅ Authentication API loaded")

if HAS_HEALTH:
    app.include_router(health_router)
    print("✅ Health API loaded")

if HAS_MARKET_DATA:
    app.include_router(market_data_router)
    print("✅ Market Data API loaded")

if HAS_PAGINATED_DATA:
    app.include_router(paginated_data_router)
    print("✅ Paginated Data API loaded")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    print("🚀 Market Analysis System starting up...")
    print(f"📊 Environment: {os.getenv('DEBUG', 'production')}")
    print(f"🔧 Log Level: {os.getenv('LOG_LEVEL', 'INFO')}")
    if HAS_INDICATORS:
        print("📈 Technical Indicators: Available")
    else:
        print("📈 Technical Indicators: Not loaded")
    
    if HAS_MONITORING:
        print("📊 Monitoring API: Available")
    else:
        print("📊 Monitoring API: Not loaded")
    
    if HAS_STORAGE:
        print("💾 Storage API: Available")
    else:
        print("💾 Storage API: Not loaded")
    
    if HAS_AUTH:
        print("🔐 Authentication API: Available")
    else:
        print("🔐 Authentication API: Not loaded")
    
    if HAS_HEALTH:
        print("❤️ Health API: Available")
    else:
        print("❤️ Health API: Not loaded")
    
    if HAS_MARKET_DATA:
        print("📈 Market Data API: Available") 
    else:
        print("📈 Market Data API: Not loaded")
    
    if HAS_PAGINATED_DATA:
        print("📄 Paginated Data API: Available") 
    else:
        print("📄 Paginated Data API: Not loaded")
    
    print("✅ FastAPI application started successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    print("🛑 Market Analysis System shutting down...")
    print("✅ Shutdown completed successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )