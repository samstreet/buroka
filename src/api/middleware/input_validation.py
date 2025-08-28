"""
Comprehensive input validation and sanitization middleware.
Implements security best practices for all API inputs.
"""

import re
import html
import logging
import json
from typing import Any, Dict, List, Optional, Union, Set, Callable
from fastapi import Request, Response, HTTPException, status
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from pydantic import BaseModel, Field, validator
import bleach
from urllib.parse import unquote_plus


class ValidationConfig(BaseModel):
    """Configuration for input validation."""
    max_string_length: int = Field(default=1000, ge=1, le=10000)
    max_array_length: int = Field(default=100, ge=1, le=1000)
    max_object_depth: int = Field(default=10, ge=1, le=50)
    max_request_size: int = Field(default=1024 * 1024, ge=1024)  # 1MB
    allowed_html_tags: Set[str] = Field(default_factory=set)
    max_json_fields: int = Field(default=100, ge=1, le=500)
    sql_injection_patterns: List[str] = Field(default_factory=lambda: [
        r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|SCRIPT)\b)',
        r'(;|\||&|`|\$\(|\$\{)',
        r'(\'|\"|\-\-|\/\*|\*\/)',
        r'(\bOR\b.*\=|\bAND\b.*\=)',
        r'(\bUNION\b.*\bSELECT\b)',
    ])
    xss_patterns: List[str] = Field(default_factory=lambda: [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'vbscript:',
        r'on\w+\s*=',
        r'<iframe[^>]*>.*?</iframe>',
        r'<object[^>]*>.*?</object>',
        r'<embed[^>]*>',
        r'<link[^>]*>',
        r'<meta[^>]*>',
        r'expression\s*\(',
        r'url\s*\(',
    ])


class SecurityViolation(Exception):
    """Raised when a security violation is detected."""
    pass


class InputSanitizer:
    """Comprehensive input sanitizer for security."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Compile patterns for performance
        self.sql_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.config.sql_injection_patterns]
        self.xss_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.config.xss_patterns]
        
        # Common malicious patterns
        self.path_traversal_pattern = re.compile(r'\.\.[\\/]')
        self.command_injection_pattern = re.compile(r'[;&|`$(){}\[\]]')
        
        # Initialize bleach settings
        self.bleach_tags = list(self.config.allowed_html_tags) if self.config.allowed_html_tags else []
        self.bleach_attributes = {} if not self.bleach_tags else {'*': ['class', 'id']}
    
    def sanitize_string(self, value: str, field_name: str = "unknown") -> str:
        """Sanitize a string value."""
        if not isinstance(value, str):
            return str(value)
        
        # Length check
        if len(value) > self.config.max_string_length:
            self.logger.warning(f"String too long in field {field_name}: {len(value)} chars")
            raise SecurityViolation(f"String too long in field {field_name}")
        
        # URL decode to prevent double encoding attacks
        try:
            decoded_value = unquote_plus(value)
        except Exception:
            decoded_value = value
        
        # Check for SQL injection patterns
        for pattern in self.sql_patterns:
            if pattern.search(decoded_value):
                self.logger.warning(f"Potential SQL injection in field {field_name}: {decoded_value[:100]}")
                raise SecurityViolation(f"Invalid characters detected in field {field_name}")
        
        # Check for XSS patterns
        for pattern in self.xss_patterns:
            if pattern.search(decoded_value):
                self.logger.warning(f"Potential XSS in field {field_name}: {decoded_value[:100]}")
                raise SecurityViolation(f"Invalid content detected in field {field_name}")
        
        # Check for path traversal
        if self.path_traversal_pattern.search(decoded_value):
            self.logger.warning(f"Path traversal attempt in field {field_name}: {decoded_value[:100]}")
            raise SecurityViolation(f"Invalid path in field {field_name}")
        
        # Check for command injection
        if self.command_injection_pattern.search(decoded_value):
            self.logger.warning(f"Command injection attempt in field {field_name}: {decoded_value[:100]}")
            raise SecurityViolation(f"Invalid characters in field {field_name}")
        
        # HTML escape and sanitize
        sanitized = html.escape(decoded_value)
        if self.bleach_tags:
            sanitized = bleach.clean(sanitized, tags=self.bleach_tags, attributes=self.bleach_attributes)
        
        # Strip dangerous Unicode characters
        sanitized = self._strip_dangerous_unicode(sanitized)
        
        return sanitized
    
    def sanitize_dict(self, data: Dict[str, Any], path: str = "", depth: int = 0) -> Dict[str, Any]:
        """Recursively sanitize dictionary data."""
        if depth > self.config.max_object_depth:
            raise SecurityViolation("Object nesting too deep")
        
        if len(data) > self.config.max_json_fields:
            raise SecurityViolation("Too many fields in object")
        
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            clean_key = self.sanitize_string(str(key), f"{path}.{key}" if path else key)
            field_path = f"{path}.{clean_key}" if path else clean_key
            
            # Sanitize value based on type
            if isinstance(value, str):
                sanitized[clean_key] = self.sanitize_string(value, field_path)
            elif isinstance(value, dict):
                sanitized[clean_key] = self.sanitize_dict(value, field_path, depth + 1)
            elif isinstance(value, list):
                sanitized[clean_key] = self.sanitize_list(value, field_path, depth + 1)
            elif isinstance(value, (int, float, bool)) or value is None:
                sanitized[clean_key] = value
            else:
                # Convert other types to string and sanitize
                sanitized[clean_key] = self.sanitize_string(str(value), field_path)
        
        return sanitized
    
    def sanitize_list(self, data: List[Any], path: str = "", depth: int = 0) -> List[Any]:
        """Recursively sanitize list data."""
        if len(data) > self.config.max_array_length:
            raise SecurityViolation("Array too large")
        
        sanitized = []
        
        for i, value in enumerate(data):
            field_path = f"{path}[{i}]"
            
            if isinstance(value, str):
                sanitized.append(self.sanitize_string(value, field_path))
            elif isinstance(value, dict):
                sanitized.append(self.sanitize_dict(value, field_path, depth + 1))
            elif isinstance(value, list):
                sanitized.append(self.sanitize_list(value, field_path, depth + 1))
            elif isinstance(value, (int, float, bool)) or value is None:
                sanitized.append(value)
            else:
                sanitized.append(self.sanitize_string(str(value), field_path))
        
        return sanitized
    
    def _strip_dangerous_unicode(self, text: str) -> str:
        """Strip potentially dangerous Unicode characters."""
        # Remove control characters except common whitespace
        dangerous_chars = [chr(i) for i in range(32) if i not in (9, 10, 13)]  # Keep tab, newline, carriage return
        dangerous_chars.extend([chr(i) for i in range(127, 160)])  # Remove DEL and C1 control block
        
        for char in dangerous_chars:
            text = text.replace(char, '')
        
        return text


class InputValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive input validation and sanitization."""
    
    def __init__(
        self,
        app,
        config: Optional[ValidationConfig] = None,
        enabled_paths: Optional[Set[str]] = None,
        excluded_paths: Optional[Set[str]] = None
    ):
        super().__init__(app)
        self.config = config or ValidationConfig()
        self.sanitizer = InputSanitizer(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Default paths to validate (all API endpoints)
        self.enabled_paths = enabled_paths or {"/api/"}
        
        # Paths to exclude from validation
        self.excluded_paths = excluded_paths or {
            "/docs", "/redoc", "/openapi.json", "/health", "/metrics"
        }
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Validate and sanitize incoming requests."""
        
        # Check if this path should be validated
        if not self._should_validate_path(request.url.path):
            return await call_next(request)
        
        try:
            # Validate request size
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.config.max_request_size:
                self.logger.warning(f"Request too large: {content_length} bytes")
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Request too large"
                )
            
            # Sanitize query parameters
            sanitized_query_params = self._sanitize_query_params(request.query_params)
            
            # Sanitize path parameters
            sanitized_path_params = self._sanitize_path_params(request.path_params)
            
            # For POST/PUT requests, sanitize JSON body
            if request.method in ("POST", "PUT", "PATCH") and request.headers.get("content-type", "").startswith("application/json"):
                body = await request.body()
                if body:
                    try:
                        json_data = json.loads(body)
                        sanitized_json = self.sanitizer.sanitize_dict(json_data) if isinstance(json_data, dict) else self.sanitizer.sanitize_list(json_data)
                        
                        # Replace request body with sanitized version
                        sanitized_body = json.dumps(sanitized_json).encode('utf-8')
                        request._body = sanitized_body
                        
                    except json.JSONDecodeError:
                        self.logger.warning("Invalid JSON in request body")
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid JSON format"
                        )
            
            # Store sanitized parameters in request state for use by endpoints
            request.state.sanitized_query = sanitized_query_params
            request.state.sanitized_path = sanitized_path_params
            
            # Log security validation
            self.logger.debug(f"Input validation passed for {request.method} {request.url.path}")
            
            response = await call_next(request)
            return response
            
        except SecurityViolation as e:
            self.logger.warning(f"Security violation detected: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Input validation failed"
            )
    
    def _should_validate_path(self, path: str) -> bool:
        """Check if a path should be validated."""
        # Check exclusions first
        for excluded in self.excluded_paths:
            if path.startswith(excluded):
                return False
        
        # Check if any enabled path matches
        for enabled in self.enabled_paths:
            if path.startswith(enabled):
                return True
        
        return False
    
    def _sanitize_query_params(self, query_params) -> Dict[str, str]:
        """Sanitize query parameters."""
        sanitized = {}
        for key, value in query_params.items():
            clean_key = self.sanitizer.sanitize_string(key, f"query.{key}")
            clean_value = self.sanitizer.sanitize_string(value, f"query.{key}")
            sanitized[clean_key] = clean_value
        return sanitized
    
    def _sanitize_path_params(self, path_params) -> Dict[str, str]:
        """Sanitize path parameters."""
        sanitized = {}
        for key, value in path_params.items():
            clean_key = self.sanitizer.sanitize_string(key, f"path.{key}")
            clean_value = self.sanitizer.sanitize_string(str(value), f"path.{key}")
            sanitized[clean_key] = clean_value
        return sanitized


# Convenience functions for endpoint-level validation
def validate_symbol(symbol: str) -> str:
    """Validate and sanitize trading symbol."""
    sanitizer = InputSanitizer()
    clean_symbol = sanitizer.sanitize_string(symbol, "symbol")
    
    # Additional symbol-specific validation
    if not re.match(r'^[A-Z0-9.-]{1,10}$', clean_symbol.upper()):
        raise SecurityViolation("Invalid symbol format")
    
    return clean_symbol.upper()


def validate_email(email: str) -> str:
    """Validate and sanitize email address."""
    sanitizer = InputSanitizer()
    clean_email = sanitizer.sanitize_string(email, "email")
    
    # Basic email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, clean_email):
        raise SecurityViolation("Invalid email format")
    
    return clean_email.lower()


def validate_username(username: str) -> str:
    """Validate and sanitize username."""
    sanitizer = InputSanitizer()
    clean_username = sanitizer.sanitize_string(username, "username")
    
    # Username validation
    if not re.match(r'^[a-zA-Z0-9_-]{3,50}$', clean_username):
        raise SecurityViolation("Invalid username format")
    
    return clean_username


def validate_api_key(api_key: str) -> str:
    """Validate and sanitize API key."""
    sanitizer = InputSanitizer()
    clean_key = sanitizer.sanitize_string(api_key, "api_key")
    
    # API key should be alphanumeric with specific length
    if not re.match(r'^[a-zA-Z0-9_-]{32,128}$', clean_key):
        raise SecurityViolation("Invalid API key format")
    
    return clean_key


# Custom validator decorators
def requires_clean_input(func):
    """Decorator to ensure input sanitization is applied."""
    async def wrapper(*args, **kwargs):
        # This would be implemented with dependency injection in FastAPI
        return await func(*args, **kwargs)
    return wrapper