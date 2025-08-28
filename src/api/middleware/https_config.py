"""
HTTPS configuration and SSL/TLS security setup.
Implements security best practices for production deployment.
"""

import os
import ssl
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, validator
import uvicorn


class SSLConfig(BaseModel):
    """SSL/TLS configuration settings."""
    
    # Certificate files
    ssl_cert_file: Optional[str] = Field(default=None, description="Path to SSL certificate file")
    ssl_key_file: Optional[str] = Field(default=None, description="Path to SSL private key file")
    ssl_ca_file: Optional[str] = Field(default=None, description="Path to CA certificate file")
    ssl_cert_chain_file: Optional[str] = Field(default=None, description="Path to certificate chain file")
    
    # SSL settings
    ssl_version: str = Field(default="TLSv1.2", description="Minimum TLS version")
    ssl_ciphers: str = Field(
        default="ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS",
        description="SSL cipher suite"
    )
    ssl_protocols: List[str] = Field(
        default_factory=lambda: ["TLSv1.2", "TLSv1.3"],
        description="Allowed SSL/TLS protocols"
    )
    
    # Security options
    require_client_cert: bool = Field(default=False, description="Require client certificate")
    verify_client_cert: bool = Field(default=False, description="Verify client certificate")
    ssl_check_hostname: bool = Field(default=True, description="Check SSL hostname")
    
    # HTTPS enforcement
    force_https: bool = Field(default=True, description="Force HTTPS redirects")
    https_port: int = Field(default=443, ge=1, le=65535, description="HTTPS port")
    http_port: int = Field(default=80, ge=1, le=65535, description="HTTP port (for redirects)")
    
    # HSTS settings
    hsts_max_age: int = Field(default=31536000, ge=0, description="HSTS max age in seconds")
    hsts_include_subdomains: bool = Field(default=True, description="Include subdomains in HSTS")
    hsts_preload: bool = Field(default=False, description="Enable HSTS preload")
    
    @validator('ssl_cert_file')
    def cert_file_exists(cls, v):
        if v and not Path(v).exists():
            raise ValueError(f'SSL certificate file not found: {v}')
        return v
    
    @validator('ssl_key_file')
    def key_file_exists(cls, v):
        if v and not Path(v).exists():
            raise ValueError(f'SSL key file not found: {v}')
        return v
    
    @validator('ssl_version')
    def valid_ssl_version(cls, v):
        valid_versions = ['TLSv1.2', 'TLSv1.3', 'SSLv23']
        if v not in valid_versions:
            raise ValueError(f'Invalid SSL version: {v}. Must be one of {valid_versions}')
        return v


class HTTPSRedirectMiddleware:
    """Middleware to enforce HTTPS redirects."""
    
    def __init__(self, app, ssl_config: SSLConfig):
        self.app = app
        self.ssl_config = ssl_config
        self.logger = logging.getLogger(__name__)
    
    async def __call__(self, scope, receive, send):
        if (
            scope["type"] == "http" and
            self.ssl_config.force_https and
            scope.get("scheme") == "http"
        ):
            # Redirect HTTP to HTTPS
            url = scope.get("path", "/")
            query_string = scope.get("query_string", b"").decode()
            if query_string:
                url += f"?{query_string}"
            
            # Determine HTTPS URL
            host = None
            for name, value in scope.get("headers", []):
                if name == b"host":
                    host = value.decode()
                    break
            
            if host:
                # Remove port if it's the default HTTP port
                if f":{self.ssl_config.http_port}" in host and self.ssl_config.http_port == 80:
                    host = host.replace(f":{self.ssl_config.http_port}", "")
                
                # Add HTTPS port if not default
                if self.ssl_config.https_port != 443:
                    if ":" not in host:
                        host = f"{host}:{self.ssl_config.https_port}"
                
                https_url = f"https://{host}{url}"
                
                self.logger.info(f"Redirecting HTTP to HTTPS: {url} -> {https_url}")
                
                # Send redirect response
                response = {
                    "type": "http.response.start",
                    "status": 301,
                    "headers": [
                        (b"location", https_url.encode()),
                        (b"content-length", b"0"),
                    ],
                }
                await send(response)
                
                await send({"type": "http.response.body"})
                return
        
        await self.app(scope, receive, send)


class SecurityHeadersEnhanced:
    """Enhanced security headers for HTTPS environments."""
    
    def __init__(self, ssl_config: SSLConfig):
        self.ssl_config = ssl_config
    
    def get_headers(self) -> Dict[str, str]:
        """Get enhanced security headers."""
        headers = {
            # Basic security headers
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "connect-src 'self' wss: https:; "
                "font-src 'self' data:; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            ),
            
            # Permissions Policy (formerly Feature Policy)
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "accelerometer=(), "
                "gyroscope=()"
            ),
            
            # Additional security headers
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Resource-Policy": "same-origin",
        }
        
        # HSTS header for HTTPS
        if self.ssl_config.force_https:
            hsts_value = f"max-age={self.ssl_config.hsts_max_age}"
            if self.ssl_config.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            if self.ssl_config.hsts_preload:
                hsts_value += "; preload"
            
            headers["Strict-Transport-Security"] = hsts_value
        
        return headers


class SSLContextManager:
    """Manages SSL context creation and configuration."""
    
    def __init__(self, ssl_config: SSLConfig):
        self.ssl_config = ssl_config
        self.logger = logging.getLogger(__name__)
    
    def create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create and configure SSL context."""
        if not self.ssl_config.ssl_cert_file or not self.ssl_config.ssl_key_file:
            self.logger.warning("SSL certificate or key file not provided")
            return None
        
        try:
            # Create SSL context
            if self.ssl_config.ssl_version == "TLSv1.3":
                context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                context.minimum_version = ssl.TLSVersion.TLSv1_3
            elif self.ssl_config.ssl_version == "TLSv1.2":
                context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                context.minimum_version = ssl.TLSVersion.TLSv1_2
            else:
                context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            
            # Load certificate and key
            context.load_cert_chain(
                certfile=self.ssl_config.ssl_cert_file,
                keyfile=self.ssl_config.ssl_key_file
            )
            
            # Load CA certificates if provided
            if self.ssl_config.ssl_ca_file:
                context.load_verify_locations(cafile=self.ssl_config.ssl_ca_file)
            
            # Configure ciphers
            if self.ssl_config.ssl_ciphers:
                context.set_ciphers(self.ssl_config.ssl_ciphers)
            
            # Client certificate options
            if self.ssl_config.require_client_cert:
                context.verify_mode = ssl.CERT_REQUIRED
            elif self.ssl_config.verify_client_cert:
                context.verify_mode = ssl.CERT_OPTIONAL
            else:
                context.verify_mode = ssl.CERT_NONE
            
            # Hostname checking
            context.check_hostname = self.ssl_config.ssl_check_hostname
            
            # Security options
            context.options |= ssl.OP_NO_SSLv2
            context.options |= ssl.OP_NO_SSLv3
            context.options |= ssl.OP_NO_TLSv1
            context.options |= ssl.OP_NO_TLSv1_1
            context.options |= ssl.OP_SINGLE_DH_USE
            context.options |= ssl.OP_SINGLE_ECDH_USE
            
            # Disable compression to prevent CRIME attack
            context.options |= ssl.OP_NO_COMPRESSION
            
            self.logger.info("SSL context created successfully")
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to create SSL context: {e}")
            raise
    
    def validate_certificates(self) -> bool:
        """Validate SSL certificates."""
        try:
            if not self.ssl_config.ssl_cert_file:
                return False
            
            # Load and verify certificate
            with open(self.ssl_config.ssl_cert_file, 'r') as f:
                cert_content = f.read()
                
            # Basic validation - check if it looks like a certificate
            if '-----BEGIN CERTIFICATE-----' not in cert_content:
                self.logger.error("Certificate file does not contain valid certificate")
                return False
            
            # Check if key file exists and is readable
            if self.ssl_config.ssl_key_file:
                with open(self.ssl_config.ssl_key_file, 'r') as f:
                    key_content = f.read()
                    
                if '-----BEGIN PRIVATE KEY-----' not in key_content and '-----BEGIN RSA PRIVATE KEY-----' not in key_content:
                    self.logger.error("Key file does not contain valid private key")
                    return False
            
            self.logger.info("Certificate validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Certificate validation failed: {e}")
            return False


def create_self_signed_cert(
    domain: str = "localhost",
    cert_file: str = "cert.pem",
    key_file: str = "key.pem",
    days_valid: int = 365
) -> bool:
    """Create self-signed certificate for development."""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from datetime import datetime, timedelta
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Development"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Local"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Market Analysis System"),
            x509.NameAttribute(NameOID.COMMON_NAME, domain),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=days_valid)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(domain),
                x509.DNSName(f"*.{domain}"),
                x509.DNSName("localhost"),
                x509.IPAddress("127.0.0.1"),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Write certificate
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        # Write private key
        with open(key_file, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        logging.info(f"Self-signed certificate created: {cert_file}, {key_file}")
        return True
        
    except ImportError:
        logging.error("cryptography package required for certificate generation")
        return False
    except Exception as e:
        logging.error(f"Failed to create self-signed certificate: {e}")
        return False


def get_ssl_config_from_env() -> SSLConfig:
    """Create SSL configuration from environment variables."""
    return SSLConfig(
        ssl_cert_file=os.getenv('SSL_CERT_FILE'),
        ssl_key_file=os.getenv('SSL_KEY_FILE'),
        ssl_ca_file=os.getenv('SSL_CA_FILE'),
        ssl_version=os.getenv('SSL_VERSION', 'TLSv1.2'),
        ssl_ciphers=os.getenv('SSL_CIPHERS'),
        force_https=os.getenv('FORCE_HTTPS', 'true').lower() == 'true',
        https_port=int(os.getenv('HTTPS_PORT', 443)),
        http_port=int(os.getenv('HTTP_PORT', 80)),
        hsts_max_age=int(os.getenv('HSTS_MAX_AGE', 31536000)),
        hsts_include_subdomains=os.getenv('HSTS_INCLUDE_SUBDOMAINS', 'true').lower() == 'true',
        hsts_preload=os.getenv('HSTS_PRELOAD', 'false').lower() == 'true',
        require_client_cert=os.getenv('REQUIRE_CLIENT_CERT', 'false').lower() == 'true',
        verify_client_cert=os.getenv('VERIFY_CLIENT_CERT', 'false').lower() == 'true',
    )


def configure_uvicorn_ssl(ssl_config: SSLConfig) -> Dict[str, Any]:
    """Configure uvicorn SSL settings."""
    ssl_context_manager = SSLContextManager(ssl_config)
    
    config = {
        "host": "0.0.0.0",
        "port": ssl_config.https_port,
        "ssl_keyfile": ssl_config.ssl_key_file,
        "ssl_certfile": ssl_config.ssl_cert_file,
        "ssl_version": getattr(ssl, f"PROTOCOL_{ssl_config.ssl_version.replace('.', '_')}", ssl.PROTOCOL_TLS_SERVER),
        "ssl_cert_reqs": ssl.CERT_NONE,
        "ssl_ciphers": ssl_config.ssl_ciphers,
    }
    
    # Client certificate settings
    if ssl_config.require_client_cert:
        config["ssl_cert_reqs"] = ssl.CERT_REQUIRED
    elif ssl_config.verify_client_cert:
        config["ssl_cert_reqs"] = ssl.CERT_OPTIONAL
    
    # CA file
    if ssl_config.ssl_ca_file:
        config["ssl_ca_certs"] = ssl_config.ssl_ca_file
    
    return config


class SecurityAuditor:
    """Audits SSL/TLS configuration for security issues."""
    
    def __init__(self, ssl_config: SSLConfig):
        self.ssl_config = ssl_config
        self.logger = logging.getLogger(__name__)
    
    def audit_ssl_config(self) -> Dict[str, Any]:
        """Audit SSL configuration for security issues."""
        issues = []
        recommendations = []
        score = 100
        
        # Check certificate files
        if not self.ssl_config.ssl_cert_file:
            issues.append("No SSL certificate configured")
            score -= 50
        
        if not self.ssl_config.ssl_key_file:
            issues.append("No SSL private key configured")
            score -= 50
        
        # Check TLS version
        if self.ssl_config.ssl_version not in ['TLSv1.2', 'TLSv1.3']:
            issues.append(f"Insecure SSL version: {self.ssl_config.ssl_version}")
            score -= 30
        elif self.ssl_config.ssl_version == 'TLSv1.2':
            recommendations.append("Consider upgrading to TLSv1.3")
            score -= 5
        
        # Check HTTPS enforcement
        if not self.ssl_config.force_https:
            issues.append("HTTPS not enforced")
            score -= 20
        
        # Check HSTS
        if self.ssl_config.hsts_max_age < 31536000:  # 1 year
            recommendations.append("HSTS max-age should be at least 1 year")
            score -= 5
        
        if not self.ssl_config.hsts_include_subdomains:
            recommendations.append("Enable HSTS includeSubDomains")
            score -= 3
        
        # Check cipher suite
        weak_ciphers = ['RC4', 'DES', '3DES', 'MD5', 'SHA1']
        if any(weak in self.ssl_config.ssl_ciphers.upper() for weak in weak_ciphers):
            issues.append("Weak ciphers detected in cipher suite")
            score -= 15
        
        # Check ports
        if self.ssl_config.https_port == self.ssl_config.http_port:
            issues.append("HTTPS and HTTP ports are the same")
            score -= 10
        
        return {
            "score": max(0, score),
            "grade": self._get_security_grade(score),
            "issues": issues,
            "recommendations": recommendations,
            "audit_passed": score >= 70
        }
    
    def _get_security_grade(self, score: int) -> str:
        """Get security grade based on score."""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"