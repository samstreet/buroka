"""
Security configuration management and HTTPS setup.
Implements security best practices and configuration.
"""

import os
import secrets
import logging
from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import json
from pathlib import Path


class SecuritySettings(BaseModel):
    """Comprehensive security settings."""
    
    # JWT Configuration
    jwt_secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(default=30, ge=5, le=1440)
    jwt_refresh_token_expire_days: int = Field(default=7, ge=1, le=30)
    
    # Rate Limiting
    rate_limit_requests_per_hour: int = Field(default=1000, ge=100, le=10000)
    rate_limit_burst_size: int = Field(default=100, ge=10, le=1000)
    rate_limit_enabled: bool = Field(default=True)
    
    # API Key Settings
    api_key_length: int = Field(default=32, ge=16, le=128)
    api_key_expiry_days: int = Field(default=365, ge=1, le=3650)
    require_api_key: bool = Field(default=False)
    
    # Password Policy
    min_password_length: int = Field(default=8, ge=6, le=128)
    require_password_uppercase: bool = Field(default=True)
    require_password_lowercase: bool = Field(default=True)
    require_password_digit: bool = Field(default=True)
    require_password_special: bool = Field(default=True)
    password_history_count: int = Field(default=5, ge=0, le=24)
    
    # HTTPS Configuration
    https_only: bool = Field(default=True)
    https_port: int = Field(default=443, ge=1, le=65535)
    ssl_cert_file: Optional[str] = Field(default=None)
    ssl_key_file: Optional[str] = Field(default=None)
    ssl_ca_file: Optional[str] = Field(default=None)
    
    # Security Headers
    enable_hsts: bool = Field(default=True)
    hsts_max_age: int = Field(default=31536000)  # 1 year
    hsts_include_subdomains: bool = Field(default=True)
    enable_csp: bool = Field(default=True)
    csp_policy: str = Field(default="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'")
    
    # Input Validation
    max_request_size: int = Field(default=1024 * 1024, ge=1024)  # 1MB
    max_string_length: int = Field(default=1000, ge=10, le=10000)
    max_array_length: int = Field(default=100, ge=1, le=1000)
    enable_input_sanitization: bool = Field(default=True)
    
    # Session Security
    session_timeout_minutes: int = Field(default=30, ge=5, le=1440)
    max_concurrent_sessions: int = Field(default=5, ge=1, le=100)
    
    # Audit Logging
    enable_audit_logging: bool = Field(default=True)
    audit_log_file: str = Field(default="logs/audit.log")
    audit_log_retention_days: int = Field(default=90, ge=1, le=3650)
    log_sensitive_data: bool = Field(default=False)
    
    # IP Restrictions
    allowed_ip_ranges: List[str] = Field(default_factory=list)
    blocked_ip_ranges: List[str] = Field(default_factory=list)
    enable_geo_blocking: bool = Field(default=False)
    allowed_countries: List[str] = Field(default_factory=list)
    
    # Security Monitoring
    enable_intrusion_detection: bool = Field(default=True)
    max_failed_login_attempts: int = Field(default=5, ge=1, le=20)
    lockout_duration_minutes: int = Field(default=15, ge=1, le=1440)
    suspicious_activity_threshold: int = Field(default=10, ge=1, le=100)
    
    # CORS Configuration
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    cors_methods: List[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    cors_headers: List[str] = Field(default_factory=lambda: ["*"])
    cors_expose_headers: List[str] = Field(default_factory=lambda: ["X-Process-Time", "X-RateLimit-*"])
    
    @validator('jwt_secret_key')
    def jwt_secret_must_be_strong(cls, v):
        if len(v) < 32:
            raise ValueError('JWT secret key must be at least 32 characters long')
        return v
    
    @validator('csp_policy')
    def csp_policy_must_be_valid(cls, v):
        # Basic CSP validation
        if 'default-src' not in v:
            raise ValueError('CSP policy must include default-src directive')
        return v


class SecurityConfigManager:
    """Manages security configuration and validation."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file or "config/security.json"
        self.settings = self._load_settings()
        self._validate_configuration()
    
    def _load_settings(self) -> SecuritySettings:
        """Load security settings from file or environment."""
        settings_dict = {}
        
        # Load from file if exists
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    settings_dict.update(json.load(f))
                self.logger.info(f"Loaded security config from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Override with environment variables
        env_mappings = {
            'JWT_SECRET_KEY': 'jwt_secret_key',
            'JWT_ALGORITHM': 'jwt_algorithm',
            'RATE_LIMIT_PER_HOUR': 'rate_limit_requests_per_hour',
            'HTTPS_ONLY': 'https_only',
            'SSL_CERT_FILE': 'ssl_cert_file',
            'SSL_KEY_FILE': 'ssl_key_file',
            'MAX_REQUEST_SIZE': 'max_request_size',
            'ENABLE_AUDIT_LOGGING': 'enable_audit_logging',
            'CORS_ORIGINS': 'cors_origins',
        }
        
        for env_var, setting_name in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Handle different data types
                if setting_name in ['https_only', 'enable_audit_logging']:
                    settings_dict[setting_name] = value.lower() in ('true', '1', 'yes', 'on')
                elif setting_name in ['rate_limit_requests_per_hour', 'max_request_size']:
                    settings_dict[setting_name] = int(value)
                elif setting_name == 'cors_origins':
                    settings_dict[setting_name] = value.split(',')
                else:
                    settings_dict[setting_name] = value
        
        return SecuritySettings(**settings_dict)
    
    def _validate_configuration(self) -> None:
        """Validate security configuration."""
        issues = []
        
        # Check for weak JWT secret
        if len(self.settings.jwt_secret_key) < 32:
            issues.append("JWT secret key is too short")
        
        # Check SSL configuration
        if self.settings.https_only:
            if not self.settings.ssl_cert_file or not self.settings.ssl_key_file:
                issues.append("HTTPS enabled but SSL cert/key files not configured")
            else:
                if not Path(self.settings.ssl_cert_file).exists():
                    issues.append(f"SSL cert file not found: {self.settings.ssl_cert_file}")
                if not Path(self.settings.ssl_key_file).exists():
                    issues.append(f"SSL key file not found: {self.settings.ssl_key_file}")
        
        # Check audit log directory
        if self.settings.enable_audit_logging:
            audit_dir = Path(self.settings.audit_log_file).parent
            if not audit_dir.exists():
                try:
                    audit_dir.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Created audit log directory: {audit_dir}")
                except Exception as e:
                    issues.append(f"Cannot create audit log directory: {e}")
        
        # Check rate limiting configuration
        if self.settings.rate_limit_requests_per_hour < 100:
            issues.append("Rate limit is too restrictive (< 100 requests/hour)")
        
        if issues:
            self.logger.warning(f"Security configuration issues: {'; '.join(issues)}")
        else:
            self.logger.info("Security configuration validation passed")
    
    def save_settings(self) -> None:
        """Save current settings to file."""
        try:
            config_path = Path(self.config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(self.settings.dict(), f, indent=2, default=str)
            
            self.logger.info(f"Security settings saved to {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save security settings: {e}")
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration for FastAPI."""
        return {
            "allow_origins": self.settings.cors_origins,
            "allow_credentials": True,
            "allow_methods": self.settings.cors_methods,
            "allow_headers": self.settings.cors_headers,
            "expose_headers": self.settings.cors_expose_headers,
        }
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers configuration."""
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }
        
        if self.settings.enable_hsts:
            hsts_value = f"max-age={self.settings.hsts_max_age}"
            if self.settings.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            headers["Strict-Transport-Security"] = hsts_value
        
        if self.settings.enable_csp:
            headers["Content-Security-Policy"] = self.settings.csp_policy
        
        return headers
    
    def is_ip_allowed(self, ip_address: str) -> bool:
        """Check if IP address is allowed."""
        if not self.settings.allowed_ip_ranges and not self.settings.blocked_ip_ranges:
            return True
        
        # TODO: Implement IP range checking
        # This would require a library like netaddr for proper CIDR matching
        return True
    
    def generate_api_key(self) -> str:
        """Generate a new API key."""
        return secrets.token_urlsafe(self.settings.api_key_length)
    
    def validate_password(self, password: str) -> List[str]:
        """Validate password against security policy."""
        errors = []
        
        if len(password) < self.settings.min_password_length:
            errors.append(f"Password must be at least {self.settings.min_password_length} characters")
        
        if self.settings.require_password_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.settings.require_password_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.settings.require_password_digit and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if self.settings.require_password_special:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                errors.append("Password must contain at least one special character")
        
        return errors
    
    def get_jwt_settings(self) -> Dict[str, Any]:
        """Get JWT configuration."""
        return {
            "secret_key": self.settings.jwt_secret_key,
            "algorithm": self.settings.jwt_algorithm,
            "access_token_expire_minutes": self.settings.jwt_access_token_expire_minutes,
            "refresh_token_expire_days": self.settings.jwt_refresh_token_expire_days,
        }
    
    def get_rate_limit_settings(self) -> Dict[str, Any]:
        """Get rate limiting configuration."""
        return {
            "requests_per_hour": self.settings.rate_limit_requests_per_hour,
            "burst_size": self.settings.rate_limit_burst_size,
            "enabled": self.settings.rate_limit_enabled,
        }


class SecurityAuditor:
    """Security auditing and compliance checking."""
    
    def __init__(self, config_manager: SecurityConfigManager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
    
    def audit_configuration(self) -> Dict[str, Any]:
        """Perform security configuration audit."""
        issues = []
        recommendations = []
        score = 100
        
        settings = self.config.settings
        
        # Check JWT configuration
        if len(settings.jwt_secret_key) < 32:
            issues.append("JWT secret key is too short")
            score -= 10
        
        if settings.jwt_access_token_expire_minutes > 60:
            recommendations.append("Consider shorter JWT token expiration")
            score -= 2
        
        # Check HTTPS configuration
        if not settings.https_only:
            issues.append("HTTPS not enforced")
            score -= 15
        
        # Check password policy
        if settings.min_password_length < 8:
            issues.append("Password minimum length is too short")
            score -= 10
        
        if not all([
            settings.require_password_uppercase,
            settings.require_password_lowercase,
            settings.require_password_digit,
            settings.require_password_special
        ]):
            recommendations.append("Enable all password complexity requirements")
            score -= 5
        
        # Check rate limiting
        if not settings.rate_limit_enabled:
            issues.append("Rate limiting is disabled")
            score -= 15
        elif settings.rate_limit_requests_per_hour > 5000:
            recommendations.append("Consider more restrictive rate limiting")
            score -= 2
        
        # Check audit logging
        if not settings.enable_audit_logging:
            issues.append("Audit logging is disabled")
            score -= 10
        
        # Check security headers
        if not settings.enable_hsts:
            issues.append("HSTS not enabled")
            score -= 5
        
        if not settings.enable_csp:
            recommendations.append("Consider enabling Content Security Policy")
            score -= 2
        
        return {
            "score": max(0, score),
            "issues": issues,
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat(),
            "audit_passed": score >= 80
        }


# Global security configuration instance
_security_config = None


def get_security_config() -> SecurityConfigManager:
    """Get the global security configuration manager."""
    global _security_config
    if _security_config is None:
        _security_config = SecurityConfigManager()
    return _security_config


def init_security_config(config_file: Optional[str] = None) -> SecurityConfigManager:
    """Initialize security configuration."""
    global _security_config
    _security_config = SecurityConfigManager(config_file)
    return _security_config