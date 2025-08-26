"""
Security middleware for the API.
"""

import time
import logging
from typing import Callable, Optional, Dict, Any
from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
import hashlib
import hmac
import secrets


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Redis-based rate limiting middleware with token bucket algorithm."""
    
    def __init__(self, app, requests_per_hour: int = 1000, use_redis: bool = True):
        super().__init__(app)
        self.requests_per_hour = requests_per_hour
        self.use_redis = use_redis
        self.logger = logging.getLogger(__name__)
        
        # Fallback in-memory rate limiting
        self.request_counts: Dict[str, Dict[str, Any]] = {}
        
        # Redis rate limiter (initialized lazily)
        self._rate_limiter = None
    
    async def _get_rate_limiter(self):
        """Get or create Redis rate limiter."""
        if self._rate_limiter is None and self.use_redis:
            try:
                from ...utils.redis_client import get_redis_client, RateLimiter
                redis_client = get_redis_client()
                self._rate_limiter = RateLimiter(
                    redis_client=redis_client,
                    requests_per_hour=self.requests_per_hour
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize Redis rate limiter: {e}")
                self.use_redis = False
        
        return self._rate_limiter
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Get client identifier
        client_id = self._get_client_identifier(request)
        
        # Try Redis rate limiting first
        if self.use_redis:
            rate_limiter = await self._get_rate_limiter()
            if rate_limiter:
                allowed, metadata = await rate_limiter.is_allowed(client_id)
                
                if not allowed:
                    self.logger.warning(f"Redis rate limit exceeded for {client_id}")
                    return self._create_rate_limit_response(metadata)
                
                # Process request
                response = await call_next(request)
                
                # Add rate limit headers
                self._add_rate_limit_headers(response, metadata)
                return response
        
        # Fallback to in-memory rate limiting
        if await self._is_memory_rate_limited(client_id):
            self.logger.warning(f"Memory rate limit exceeded for {client_id}")
            return self._create_rate_limit_response({
                "limit": self.requests_per_hour,
                "remaining": 0,
                "reset_after": 60
            })
        
        # Record request in memory
        await self._record_memory_request(client_id)
        
        # Process request
        response = await call_next(request)
        
        # Add fallback rate limit headers
        remaining = await self._get_memory_remaining_requests(client_id)
        self._add_rate_limit_headers(response, {
            "limit": self.requests_per_hour,
            "remaining": remaining,
            "reset_after": 60
        })
        
        return response
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get unique client identifier for rate limiting."""
        # Try API key first (for authenticated requests)
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
        if api_key and api_key.startswith("Bearer "):
            return f"api_key:{api_key[7:20]}..."  # Use first 20 chars of token
        elif api_key:
            return f"api_key:{api_key[:20]}..."
        
        # Fall back to IP address
        return f"ip:{self._get_client_ip(request)}"
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client
        return request.client.host if request.client else "unknown"
    
    def _create_rate_limit_response(self, metadata: Dict[str, Any]) -> Response:
        """Create HTTP 429 rate limit response."""
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
                "limit": metadata.get("limit"),
                "remaining": metadata.get("remaining", 0),
                "reset_after": metadata.get("reset_after")
            },
            headers={
                "X-RateLimit-Limit": str(metadata.get("limit", self.requests_per_hour)),
                "X-RateLimit-Remaining": str(metadata.get("remaining", 0)),
                "X-RateLimit-Reset": str(metadata.get("reset", int(time.time()) + 3600)),
                "Retry-After": str(metadata.get("reset_after", 3600))
            }
        )
    
    def _add_rate_limit_headers(self, response: Response, metadata: Dict[str, Any]) -> None:
        """Add rate limit headers to response."""
        response.headers["X-RateLimit-Limit"] = str(metadata.get("limit", self.requests_per_hour))
        response.headers["X-RateLimit-Remaining"] = str(metadata.get("remaining", 0))
        response.headers["X-RateLimit-Reset"] = str(metadata.get("reset", int(time.time()) + 3600))
    
    # Memory-based fallback rate limiting methods
    async def _is_memory_rate_limited(self, client_id: str) -> bool:
        """Check if client is rate limited using in-memory storage."""
        current_time = time.time()
        hour_window = int(current_time // 3600)
        
        if client_id not in self.request_counts:
            return False
        
        client_data = self.request_counts[client_id]
        
        # Clean old entries
        self._cleanup_memory_entries(client_data, hour_window)
        
        # Check current hour requests
        current_requests = client_data.get(hour_window, 0)
        return current_requests >= self.requests_per_hour
    
    async def _record_memory_request(self, client_id: str) -> None:
        """Record a request for the client in memory."""
        current_time = time.time()
        hour_window = int(current_time // 3600)
        
        if client_id not in self.request_counts:
            self.request_counts[client_id] = {}
        
        client_data = self.request_counts[client_id]
        client_data[hour_window] = client_data.get(hour_window, 0) + 1
        
        # Cleanup old entries
        self._cleanup_memory_entries(client_data, hour_window)
    
    async def _get_memory_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for the client from memory."""
        current_time = time.time()
        hour_window = int(current_time // 3600)
        
        if client_id not in self.request_counts:
            return self.requests_per_hour
        
        client_data = self.request_counts[client_id]
        current_requests = client_data.get(hour_window, 0)
        
        return max(0, self.requests_per_hour - current_requests)
    
    def _cleanup_memory_entries(self, client_data: Dict[str, Any], current_window: int) -> None:
        """Clean up old rate limit entries from memory."""
        # Keep only current and previous hour
        keys_to_remove = [
            key for key in client_data.keys() 
            if isinstance(key, int) and key < current_window - 1
        ]
        for key in keys_to_remove:
            del client_data[key]


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced logging middleware with structured logging."""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger(__name__)
        
        # Try to import structured logger
        try:
            from ...utils.logging_config import get_request_logger
            self.structured_logger = get_request_logger()
            self.use_structured = True
        except ImportError:
            self.structured_logger = None
            self.use_structured = False
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate request ID
        request_id = f"req_{secrets.token_urlsafe(8)}"
        
        # Record start time
        start_time = time.time()
        
        # Get request details
        client_ip = self._get_client_ip(request)
        method = request.method
        url = str(request.url)
        user_agent = request.headers.get("User-Agent", "")
        
        # Get user info if available
        user_id = None
        if hasattr(request.state, 'user'):
            user_id = getattr(request.state.user, 'user_id', None)
        elif hasattr(request.state, 'api_key_info'):
            user_id = f"api_key:{request.state.api_key_info.get('id', 'unknown')}"
        
        # Get content length
        content_length = request.headers.get("content-length")
        body_size = int(content_length) if content_length else None
        
        # Log request
        if self.use_structured and self.structured_logger:
            self.structured_logger.log_request(
                method=method,
                url=url,
                ip_address=client_ip,
                user_agent=user_agent,
                user_id=user_id,
                request_id=request_id,
                body_size=body_size
            )
        else:
            # Fallback logging
            self.logger.info(f"Request: {method} {url} from {client_ip} [{request_id}]")
        
        try:
            # Add request ID to request state
            request.state.request_id = request_id
            
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            process_time = time.time() - start_time
            
            # Get response size
            response_size = None
            if hasattr(response, 'headers') and 'content-length' in response.headers:
                response_size = int(response.headers['content-length'])
            
            # Log response
            if self.use_structured and self.structured_logger:
                self.structured_logger.log_response(
                    method=method,
                    url=url,
                    status_code=response.status_code,
                    response_time=process_time,
                    response_size=response_size,
                    user_id=user_id,
                    request_id=request_id
                )
            else:
                # Fallback logging
                self.logger.info(
                    f"Response: {response.status_code} for {method} {url} "
                    f"- {process_time:.3f}s [{request_id}]"
                )
            
            # Add response headers
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate response time
            process_time = time.time() - start_time
            
            # Log error
            if self.use_structured and self.structured_logger:
                self.structured_logger.log_response(
                    method=method,
                    url=url,
                    status_code=500,
                    response_time=process_time,
                    user_id=user_id,
                    request_id=request_id,
                    error=str(e)
                )
            else:
                # Fallback logging
                self.logger.error(
                    f"Error: {str(e)} for {method} {url} "
                    f"- {process_time:.3f}s [{request_id}]"
                )
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


class APIKeyMiddleware(BaseHTTPMiddleware):
    """API key authentication middleware."""
    
    def __init__(self, app, api_keys: Optional[Dict[str, Dict[str, Any]]] = None):
        super().__init__(app)
        self.api_keys = api_keys or {}
        self.logger = logging.getLogger(__name__)
        self._get_api_keys = None  # Function to get current API keys
        
        # Paths that don't require authentication
        self.public_paths = {
            "/",
            "/health",
            "/docs", 
            "/redoc",
            "/openapi.json",
            "/api/v1/info",
            "/api/v1/test",
            "/api/v1/health",
            "/api/v1/health/simple",
            "/api/v1/health/ready", 
            "/api/v1/health/live",
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/status"
        }
    
    def set_api_keys_getter(self, getter_func):
        """Set function to dynamically get API keys."""
        self._get_api_keys = getter_func
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip authentication for public paths
        if self._is_public_path(request.url.path):
            return await call_next(request)
        
        # Get current API keys
        current_api_keys = self.api_keys
        if self._get_api_keys:
            try:
                current_api_keys = self._get_api_keys()
            except Exception as e:
                self.logger.error(f"Error getting API keys: {e}")
        
        # Skip authentication if no API keys configured
        if not current_api_keys:
            return await call_next(request)
        
        # Check for API key
        api_key = self._extract_api_key(request)
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required"
            )
        
        # Validate API key
        key_info = current_api_keys.get(api_key)
        if not key_info:
            self.logger.warning(f"Invalid API key used: {api_key[:8]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        # Check if key is active
        if not key_info.get("active", True):
            self.logger.warning(f"Inactive API key used: {api_key[:8]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is disabled"
            )
        
        # Add key info to request state
        request.state.api_key_info = key_info
        
        # Log API key usage
        self.logger.info(f"API key used: {key_info.get('name', 'unknown')} for {request.url.path}")
        
        return await call_next(request)
    
    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (doesn't require authentication)."""
        return path in self.public_paths or path.startswith("/api/v1/auth/")
    
    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request."""
        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        
        # Check X-API-Key header
        api_key_header = request.headers.get("X-API-Key")
        if api_key_header:
            return api_key_header
        
        # Check query parameter
        return request.query_params.get("api_key")


def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str, salt: str = "") -> str:
    """Hash an API key for secure storage."""
    if not salt:
        salt = secrets.token_hex(16)
    
    key_hash = hashlib.pbkdf2_hmac(
        'sha256',
        api_key.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    )
    
    return f"{salt}:{key_hash.hex()}"


def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """Verify an API key against its hash."""
    try:
        salt, key_hash = hashed_key.split(':', 1)
        return hashed_key == hash_api_key(api_key, salt)
    except ValueError:
        return False