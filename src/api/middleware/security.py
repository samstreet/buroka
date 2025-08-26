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
    """Simple in-memory rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 1000):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Check rate limit
        if await self._is_rate_limited(client_ip):
            self.logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Record request
        await self._record_request(client_ip)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = await self._get_remaining_requests(client_ip)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
        
        return response
    
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
    
    async def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate limited."""
        current_time = time.time()
        minute_window = int(current_time // 60)
        
        if client_ip not in self.request_counts:
            return False
        
        client_data = self.request_counts[client_ip]
        
        # Clean old entries
        self._cleanup_old_entries(client_data, minute_window)
        
        # Check current minute requests
        current_requests = client_data.get(minute_window, 0)
        return current_requests >= self.requests_per_minute
    
    async def _record_request(self, client_ip: str) -> None:
        """Record a request for the client."""
        current_time = time.time()
        minute_window = int(current_time // 60)
        
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = {}
        
        client_data = self.request_counts[client_ip]
        client_data[minute_window] = client_data.get(minute_window, 0) + 1
        
        # Cleanup old entries
        self._cleanup_old_entries(client_data, minute_window)
    
    async def _get_remaining_requests(self, client_ip: str) -> int:
        """Get remaining requests for the client."""
        current_time = time.time()
        minute_window = int(current_time // 60)
        
        if client_ip not in self.request_counts:
            return self.requests_per_minute
        
        client_data = self.request_counts[client_ip]
        current_requests = client_data.get(minute_window, 0)
        
        return max(0, self.requests_per_minute - current_requests)
    
    def _cleanup_old_entries(self, client_data: Dict[str, Any], current_window: int) -> None:
        """Clean up old rate limit entries."""
        # Keep only current and previous minute
        keys_to_remove = [
            key for key in client_data.keys() 
            if key < current_window - 1
        ]
        for key in keys_to_remove:
            del client_data[key]


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests and responses."""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger(__name__)
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Record start time
        start_time = time.time()
        
        # Get request details
        client_ip = self._get_client_ip(request)
        method = request.method
        url = str(request.url)
        user_agent = request.headers.get("User-Agent", "")
        
        # Log request
        self.logger.info(f"Request: {method} {url} from {client_ip} - {user_agent}")
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            process_time = time.time() - start_time
            
            # Log response
            self.logger.info(
                f"Response: {response.status_code} for {method} {url} "
                f"- {process_time:.3f}s"
            )
            
            # Add response time header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            self.logger.error(
                f"Error: {str(e)} for {method} {url} "
                f"- {process_time:.3f}s"
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
        
        # Paths that don't require authentication
        self.public_paths = {
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/info",
            "/api/v1/test"
        }
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip authentication for public paths
        if self._is_public_path(request.url.path):
            return await call_next(request)
        
        # Skip authentication if no API keys configured
        if not self.api_keys:
            return await call_next(request)
        
        # Check for API key
        api_key = self._extract_api_key(request)
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required"
            )
        
        # Validate API key
        key_info = self.api_keys.get(api_key)
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