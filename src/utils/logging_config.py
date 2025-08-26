"""
Advanced logging configuration for the Market Analysis System.
"""

import logging
import sys
import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler
import traceback


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter for logs.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        # Create structured log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        if hasattr(record, 'ip_address'):
            log_entry["ip_address"] = record.ip_address
        if hasattr(record, 'user_agent'):
            log_entry["user_agent"] = record.user_agent
        if hasattr(record, 'method'):
            log_entry["method"] = record.method
        if hasattr(record, 'url'):
            log_entry["url"] = record.url
        if hasattr(record, 'status_code'):
            log_entry["status_code"] = record.status_code
        if hasattr(record, 'response_time'):
            log_entry["response_time"] = record.response_time
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry, ensure_ascii=False)


class APIRequestLogger:
    """
    Specialized logger for API requests with structured logging.
    """
    
    def __init__(self, logger_name: str = "api_requests"):
        self.logger = logging.getLogger(logger_name)
        
    def log_request(
        self,
        method: str,
        url: str,
        ip_address: str,
        user_agent: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        body_size: Optional[int] = None
    ):
        """Log API request."""
        extra = {
            'method': method,
            'url': url,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'event_type': 'request',
        }
        
        if user_id:
            extra['user_id'] = user_id
        if request_id:
            extra['request_id'] = request_id
        if body_size is not None:
            extra['body_size'] = body_size
        if headers:
            extra['headers'] = headers
            
        self.logger.info(f"API Request: {method} {url}", extra=extra)
    
    def log_response(
        self,
        method: str,
        url: str,
        status_code: int,
        response_time: float,
        response_size: Optional[int] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Log API response."""
        extra = {
            'method': method,
            'url': url,
            'status_code': status_code,
            'response_time': response_time,
            'event_type': 'response',
        }
        
        if user_id:
            extra['user_id'] = user_id
        if request_id:
            extra['request_id'] = request_id
        if response_size is not None:
            extra['response_size'] = response_size
        if error:
            extra['error'] = error
            
        level = logging.ERROR if status_code >= 500 else logging.WARNING if status_code >= 400 else logging.INFO
        self.logger.log(level, f"API Response: {status_code} {method} {url} ({response_time:.3f}s)", extra=extra)


class SecurityLogger:
    """
    Specialized logger for security events.
    """
    
    def __init__(self, logger_name: str = "security"):
        self.logger = logging.getLogger(logger_name)
    
    def log_rate_limit_exceeded(
        self,
        ip_address: str,
        user_agent: str,
        url: str,
        limit: int,
        user_id: Optional[str] = None
    ):
        """Log rate limit exceeded event."""
        extra = {
            'ip_address': ip_address,
            'user_agent': user_agent,
            'url': url,
            'rate_limit': limit,
            'event_type': 'rate_limit_exceeded',
            'severity': 'medium'
        }
        
        if user_id:
            extra['user_id'] = user_id
            
        self.logger.warning(f"Rate limit exceeded: {ip_address} -> {url}", extra=extra)
    
    def log_auth_failure(
        self,
        ip_address: str,
        user_agent: str,
        url: str,
        reason: str,
        username: Optional[str] = None
    ):
        """Log authentication failure."""
        extra = {
            'ip_address': ip_address,
            'user_agent': user_agent,
            'url': url,
            'reason': reason,
            'event_type': 'auth_failure',
            'severity': 'high'
        }
        
        if username:
            extra['username'] = username
            
        self.logger.warning(f"Authentication failed: {reason} from {ip_address}", extra=extra)
    
    def log_api_key_usage(
        self,
        api_key_id: str,
        ip_address: str,
        url: str,
        method: str,
        user_agent: str
    ):
        """Log API key usage."""
        extra = {
            'api_key_id': api_key_id,
            'ip_address': ip_address,
            'url': url,
            'method': method,
            'user_agent': user_agent,
            'event_type': 'api_key_usage',
            'severity': 'info'
        }
        
        self.logger.info(f"API key used: {api_key_id} from {ip_address}", extra=extra)


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "structured",
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """
    Set up comprehensive logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ('structured' for JSON, 'standard' for plain text)
        log_file: Optional log file path
        max_file_size: Maximum size of each log file
        backup_count: Number of backup files to keep
    """
    # Convert string level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters
    if log_format == "structured":
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("api_requests").setLevel(level)
    logging.getLogger("security").setLevel(level)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)  # Reduce uvicorn noise
    logging.getLogger("fastapi").setLevel(logging.WARNING)  # Reduce FastAPI noise
    
    # Log the setup completion
    root_logger.info(f"Logging configured - Level: {log_level}, Format: {log_format}")


# Global logger instances
api_request_logger = APIRequestLogger()
security_logger = SecurityLogger()


def get_request_logger() -> APIRequestLogger:
    """Get the API request logger instance."""
    return api_request_logger


def get_security_logger() -> SecurityLogger:
    """Get the security logger instance."""
    return security_logger