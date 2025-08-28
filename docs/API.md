# Market Analysis System - API Documentation

## Overview

The Market Analysis System provides a comprehensive RESTful API for real-time market data access, technical analysis, user authentication, and system monitoring. The API is built with FastAPI and follows REST conventions with OpenAPI/Swagger documentation.

**Base URL:** `http://localhost:8000` (Development)  
**API Version:** v1  
**Documentation:** Available at `/docs` (Swagger UI) and `/redoc`

## Table of Contents

- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)
- [API Endpoints](#api-endpoints)
  - [System Endpoints](#system-endpoints)
  - [Authentication](#authentication-endpoints)
  - [Market Data](#market-data-endpoints)
  - [Data Management](#data-management-endpoints)
  - [Monitoring](#monitoring-endpoints)
  - [Security](#security-endpoints)
- [Request/Response Formats](#requestresponse-formats)
- [Usage Examples](#usage-examples)

## Authentication

The API uses JWT (JSON Web Token) based authentication for protected endpoints. Some endpoints are publicly accessible while others require authentication.

### JWT Token Structure

- **Access Token**: Short-lived (30 minutes) token for API access
- **Refresh Token**: Long-lived (7 days) token for obtaining new access tokens
- **Algorithm**: HS256
- **Token Location**: Authorization header with Bearer prefix

### Authentication Flow

1. **Register/Login**: Obtain access and refresh tokens
2. **API Requests**: Include access token in requests
3. **Token Refresh**: Use refresh token to get new access tokens
4. **Logout**: Invalidate tokens (optional)

### Protected Endpoints

Most endpoints accept an optional user context. Admin-only endpoints are clearly marked.

## Rate Limiting

The API implements rate limiting to ensure fair usage and system stability.

**Default Limits:**
- **Public Endpoints**: 1000 requests/hour per IP
- **Authenticated Users**: 5000 requests/hour per user
- **Burst Limit**: 100 requests/minute

**Headers:**
- `X-RateLimit-Limit`: Rate limit ceiling for the current request
- `X-RateLimit-Remaining`: Number of requests left for the current window
- `X-RateLimit-Reset`: UTC timestamp when the rate limit window resets

## Error Handling

The API uses standard HTTP status codes and returns consistent error responses.

### Error Response Format

```json
{
  "error": "Error Type",
  "message": "Detailed error message",
  "timestamp": "2024-01-15T10:00:00Z",
  "details": {
    "field": "validation_error"
  }
}
```

### Common Status Codes

- `200 OK`: Success
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## API Endpoints

### System Endpoints

#### GET /
Root endpoint with system information.

**Response:**
```json
{
  "message": "Market Analysis System API",
  "version": "0.1.0",
  "status": "running",
  "timestamp": "2024-01-15T10:00:00Z",
  "environment": "development"
}
```

#### GET /health
Basic health check for load balancers.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:00:00Z",
  "version": "0.1.0",
  "environment": {
    "debug": "true",
    "log_level": "INFO"
  },
  "services": {
    "postgres": {
      "status": "configured",
      "host": "postgres:5432",
      "database": "market_analysis"
    }
  }
}
```

#### GET /api/v1/info
Detailed system information.

**Response:**
```json
{
  "system": {
    "name": "Market Analysis System",
    "version": "0.1.0",
    "environment": "development"
  },
  "features": {
    "data_ingestion": "planned",
    "pattern_detection": "planned",
    "real_time_analysis": "planned"
  }
}
```

#### GET /api/v1/health
Comprehensive health check with database connectivity.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2024-01-15T10:00:00Z",
  "components": {
    "postgres": {
      "status": "healthy",
      "message": "PostgreSQL connection successful",
      "response_time": 0.0234
    },
    "influxdb": {
      "status": "healthy",
      "message": "InfluxDB health status: pass"
    }
  }
}
```

### Authentication Endpoints

All authentication endpoints are under `/api/v1/auth`.

#### POST /api/v1/auth/register
Register a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecurePassword123!",
  "username": "john_doe",
  "full_name": "John Doe"
}
```

**Response (201 Created):**
```json
{
  "user_id": "uuid-123",
  "username": "john_doe",
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "roles": ["user"],
  "created_at": "2024-01-15T10:00:00Z"
}
```

#### POST /api/v1/auth/login
Authenticate user and return tokens.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecurePassword123!"
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "expires_in": 1800,
  "user_info": {
    "user_id": "uuid-123",
    "username": "john_doe",
    "email": "user@example.com"
  }
}
```

#### POST /api/v1/auth/refresh
Refresh access token using refresh token.

**Request Body:**
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

**Response:** Same as login response with new tokens.

#### POST /api/v1/auth/logout
**Headers:** `Authorization: Bearer <access_token>`

Logout user and revoke tokens.

**Response (200 OK):**
```json
{
  "message": "Successfully logged out"
}
```

#### GET /api/v1/auth/me
**Headers:** `Authorization: Bearer <access_token>`

Get current user information.

**Response:**
```json
{
  "user_id": "uuid-123",
  "username": "john_doe",
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "roles": ["user"],
  "created_at": "2024-01-15T10:00:00Z"
}
```

### Market Data Endpoints

All market data endpoints are under `/api/v1/market-data`.

#### GET /api/v1/market-data/{symbol}/quote
Get real-time quote data for a symbol.

**Path Parameters:**
- `symbol`: Stock symbol (e.g., AAPL, MSFT)

**Query Parameters:**
- `use_cache`: Whether to use cached data (default: true)

**Response:**
```json
{
  "symbol": "AAPL",
  "data_type": "quote",
  "timestamp": "2024-01-15T10:00:00Z",
  "data": {
    "quote": {
      "bid_price": 149.50,
      "ask_price": 150.00,
      "last_price": 149.75,
      "volume": 1000000
    }
  },
  "success": true,
  "metadata": {
    "source": "real-time",
    "cache_used": false
  }
}
```

#### GET /api/v1/market-data/{symbol}/daily
Get daily OHLC data for a symbol.

**Path Parameters:**
- `symbol`: Stock symbol

**Query Parameters:**
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)
- `outputsize`: Data size (compact/full)
- `use_cache`: Use cached data (default: true)

**Response:**
```json
{
  "symbol": "AAPL",
  "data_type": "daily",
  "timestamp": "2024-01-15T10:00:00Z",
  "data": {
    "ohlc_data": [
      {
        "date": "2024-01-15",
        "open": 150.00,
        "high": 152.00,
        "low": 149.00,
        "close": 151.00,
        "volume": 1000000
      }
    ]
  },
  "success": true,
  "metadata": {
    "record_count": 1,
    "date_range": {
      "start": "2024-01-15",
      "end": "2024-01-15"
    }
  }
}
```

#### GET /api/v1/market-data/{symbol}/intraday
Get intraday OHLC data with granularity options.

**Query Parameters:**
- `interval`: Data granularity (1min, 5min, 15min, 30min, 1hour)
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)

**Response:** Similar to daily data with granularity metadata.

#### POST /api/v1/market-data/batch
Get market data for multiple symbols in batch.

**Request Body:**
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "data_type": "quote"
}
```

**Response:**
```json
{
  "total_symbols": 3,
  "successful": 3,
  "failed": 0,
  "results": [
    {
      "symbol": "AAPL",
      "success": true,
      "data": { /* quote data */ }
    }
  ]
}
```

#### GET /api/v1/market-data/search
Search for symbols matching query.

**Query Parameters:**
- `query`: Search query (minimum 2 characters)
- `limit`: Maximum results (1-100, default: 10)
- `include_inactive`: Include inactive symbols (default: false)

**Response:**
```json
[
  {
    "symbol": "AAPL",
    "name": "Apple Inc.",
    "type": "Common Stock",
    "region": "United States",
    "market_open": "09:30",
    "market_close": "16:00",
    "timezone": "US/Eastern",
    "currency": "USD",
    "match_score": 1.0
  }
]
```

### Data Management Endpoints

#### GET /api/v1/data/market-history
Get paginated historical market data with filtering and sorting.

**Query Parameters:**
- `symbol`: Stock symbol (required)
- `page`: Page number (default: 1)
- `page_size`: Items per page (1-100, default: 20)
- `sort_by`: Sort field (date, close, volume)
- `sort_order`: Sort direction (asc, desc)
- `min_price`: Minimum price filter
- `max_price`: Maximum price filter

**Response:**
```json
{
  "items": [
    {
      "id": "record_1",
      "symbol": "AAPL",
      "date": "2024-01-15",
      "open": 150.00,
      "high": 152.00,
      "low": 149.00,
      "close": 151.00,
      "volume": 1000000
    }
  ],
  "total": 1000,
  "page": 1,
  "page_size": 20,
  "total_pages": 50,
  "has_next": true,
  "has_previous": false
}
```

### Storage Management Endpoints

#### GET /api/v1/storage/health
Check storage pipeline health.

**Response:**
```json
{
  "overall": "healthy",
  "components": {
    "primary_sink": {
      "status": "healthy",
      "type": "InfluxDBDataSink"
    },
    "backup_sink": {
      "status": "healthy",
      "type": "InMemoryDataSink"
    }
  }
}
```

#### GET /api/v1/storage/stats
Get storage performance statistics.

**Response:**
```json
{
  "total_writes": 10000,
  "successful_writes": 9950,
  "failed_writes": 50,
  "success_rate": 99.5,
  "duplicates_filtered": 100,
  "batch_writes": 500,
  "batch_buffer_size": 150,
  "metadata": {
    "collection_timestamp": "2024-01-15T10:00:00Z"
  }
}
```

#### POST /api/v1/storage/store/single
Store a single data record.

**Request Body:**
```json
{
  "data": {
    "symbol": "AAPL",
    "price": 150.00,
    "volume": 1000
  },
  "metadata": {
    "source": "manual_entry"
  }
}
```

### Monitoring Endpoints

#### GET /api/v1/monitoring/health
System health check for ingestion pipeline.

**Response:**
```json
{
  "overall": "healthy",
  "components": {
    "data_source": {
      "status": "healthy",
      "type": "AlphaVantageClient"
    },
    "data_transformer": {
      "status": "healthy"
    }
  },
  "api": {
    "status": "healthy",
    "timestamp": "2024-01-15T10:00:00Z"
  }
}
```

#### GET /api/v1/monitoring/metrics
Get system metrics and performance statistics.

**Response:**
```json
{
  "counters": {
    "ingestion_requests_total": 1000,
    "ingestion_requests_success_total": 950
  },
  "timings": {
    "ingestion_request_duration_seconds": {
      "avg": 0.234,
      "min": 0.100,
      "max": 2.500
    }
  },
  "metadata": {
    "collection_timestamp": "2024-01-15T10:00:00Z"
  }
}
```

#### POST /api/v1/monitoring/jobs/start
Start continuous data ingestion.

**Request Body:**
```json
{
  "symbols": ["AAPL", "MSFT"],
  "interval": 300
}
```

**Response:**
```json
{
  "success": true,
  "batch_job_id": "job_123",
  "symbols": ["AAPL", "MSFT"],
  "interval": 300,
  "started_at": "2024-01-15T10:00:00Z"
}
```

### Security Endpoints

#### GET /api/v1/security/status
Get overall security status and configuration audit.

**Response:**
```json
{
  "timestamp": "2024-01-15T10:00:00Z",
  "overall_status": "secure",
  "security_score": 95,
  "security_grade": "A",
  "security_features": {
    "rate_limiting": true,
    "input_validation": true,
    "audit_logging": true,
    "https_enforcement": true
  },
  "recent_metrics": {
    "total_requests_24h": 10000,
    "error_rate_24h": 0.02
  }
}
```

#### GET /api/v1/security/metrics
Get security-related metrics and incident reports.

**Query Parameters:**
- `hours`: Time period for metrics (max 168 hours)

**Response:**
```json
{
  "security_score": 92,
  "traffic_metrics": {
    "total_requests": 10000,
    "error_rate": 0.02,
    "avg_response_time_ms": 250
  },
  "security_incidents": {
    "total_alerts": 5,
    "high_severity_alerts": 1,
    "rate_limit_violations": 10
  }
}
```

## Request/Response Formats

### Content Types
- **Request**: `application/json`
- **Response**: `application/json`
- **Character Encoding**: UTF-8

### Date/Time Format
- **Standard**: ISO 8601 format (`2024-01-15T10:00:00Z`)
- **Timezone**: UTC preferred
- **Date Only**: `YYYY-MM-DD` format

### Decimal Precision
- **Prices**: Up to 4 decimal places
- **Percentages**: Up to 2 decimal places
- **Volumes**: Integer values

### Pagination
Standard pagination parameters for list endpoints:

```json
{
  "page": 1,
  "page_size": 20,
  "sort_by": "field_name",
  "sort_order": "desc"
}
```

## Usage Examples

### Complete Authentication Flow

```bash
# 1. Register new user
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "trader@example.com",
    "password": "SecurePass123!",
    "username": "trader1",
    "full_name": "Market Trader"
  }'

# 2. Login to get tokens
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "trader@example.com",
    "password": "SecurePass123!"
  }'

# 3. Use access token for authenticated requests
curl -X GET "http://localhost:8000/api/v1/auth/me" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Market Data Retrieval

```bash
# Get real-time quote
curl -X GET "http://localhost:8000/api/v1/market-data/AAPL/quote"

# Get daily data with date range
curl -X GET "http://localhost:8000/api/v1/market-data/AAPL/daily?start_date=2024-01-01&end_date=2024-01-31"

# Search symbols
curl -X GET "http://localhost:8000/api/v1/market-data/search?query=apple&limit=5"

# Batch data request
curl -X POST "http://localhost:8000/api/v1/market-data/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "data_type": "quote"
  }'
```

### System Monitoring

```bash
# Check system health
curl -X GET "http://localhost:8000/api/v1/health"

# Get performance metrics
curl -X GET "http://localhost:8000/api/v1/monitoring/metrics"

# Start data ingestion job
curl -X POST "http://localhost:8000/api/v1/monitoring/jobs/start" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT"],
    "interval": 300
  }'
```

### Error Handling Example

```python
import requests
import json

def get_market_data(symbol):
    try:
        response = requests.get(f"http://localhost:8000/api/v1/market-data/{symbol}/quote")
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            print(f"Symbol {symbol} not found")
        elif response.status_code == 429:
            print("Rate limit exceeded, please wait")
        else:
            error_data = response.json()
            print(f"API Error: {error_data.get('message', 'Unknown error')}")
    
    except requests.exceptions.ConnectionError:
        print("Failed to connect to API")
    
    return None
```

## SDK and Client Libraries

While the API can be accessed directly via HTTP, consider using or developing client libraries for common programming languages:

- **Python**: Use `requests` or `httpx` libraries
- **JavaScript/Node.js**: Use `axios` or `fetch`
- **Java**: Use `OkHttp` or `RestTemplate`
- **Go**: Use `net/http` package

## Support and Further Documentation

- **Interactive API Documentation**: Available at `/docs` endpoint
- **Alternative Documentation**: Available at `/redoc` endpoint
- **OpenAPI Specification**: Available at `/openapi.json`
- **Development Guide**: See `DEVELOPMENT.md`
- **Deployment Guide**: See `DEPLOYMENT.md`

For technical support or questions about API usage, please refer to the project documentation or contact the development team.