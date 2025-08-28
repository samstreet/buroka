"""
Comprehensive tests for security middleware functionality.
"""

import pytest
import time
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI, Request, HTTPException
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from src.api.middleware.security import (
    SecurityHeadersMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    APIKeyMiddleware,
    generate_api_key,
    hash_api_key,
    verify_api_key
)


class TestSecurityHeadersMiddleware:
    """Test security headers middleware."""
    
    def setup_method(self):
        """Set up test app with security headers."""
        self.app = FastAPI()
        
        @self.app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        self.app.add_middleware(SecurityHeadersMiddleware)
        self.client = TestClient(self.app)
    
    def test_security_headers_present(self):
        """Test that all security headers are added."""
        response = self.client.get("/test")
        headers = response.headers
        
        # Check all required security headers
        assert "x-content-type-options" in headers
        assert headers["x-content-type-options"] == "nosniff"
        
        assert "x-frame-options" in headers
        assert headers["x-frame-options"] == "DENY"
        
        assert "x-xss-protection" in headers
        assert headers["x-xss-protection"] == "1; mode=block"
        
        assert "strict-transport-security" in headers
        assert "max-age=31536000" in headers["strict-transport-security"]
        
        assert "referrer-policy" in headers
        assert headers["referrer-policy"] == "strict-origin-when-cross-origin"
        
        assert "content-security-policy" in headers
        assert "default-src 'self'" in headers["content-security-policy"]
    
    def test_headers_on_error_responses(self):
        """Test that security headers are added even on error responses."""
        response = self.client.get("/nonexistent")
        headers = response.headers
        
        # Even 404 responses should have security headers
        assert "x-content-type-options" in headers
        assert "x-frame-options" in headers


class TestRateLimitMiddleware:
    """Test rate limiting middleware."""
    
    def setup_method(self):
        """Set up test app with rate limiting."""
        self.app = FastAPI()
        
        @self.app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        # Use low limits for testing
        self.app.add_middleware(RateLimitMiddleware, requests_per_hour=5, use_redis=False)
        self.client = TestClient(self.app)
    
    def test_rate_limit_allows_normal_usage(self):
        """Test that normal usage is allowed."""
        response = self.client.get("/test")
        assert response.status_code == 200
        
        # Check rate limit headers
        headers = response.headers
        assert "x-ratelimit-limit" in headers
        assert "x-ratelimit-remaining" in headers
        assert "x-ratelimit-reset" in headers
    
    def test_rate_limit_blocks_excessive_requests(self):
        """Test that excessive requests are blocked."""
        # Make requests up to the limit
        for _ in range(5):
            response = self.client.get("/test")
            assert response.status_code == 200
        
        # The 6th request should be rate limited
        response = self.client.get("/test")
        assert response.status_code == 429
        
        # Check rate limit response
        data = response.json()
        assert "Rate limit exceeded" in data["error"]
        assert "remaining" in data
        assert data["remaining"] == 0
    
    def test_rate_limit_headers(self):
        """Test rate limit headers are correct."""
        response = self.client.get("/test")
        headers = response.headers
        
        assert int(headers["x-ratelimit-limit"]) == 5
        assert int(headers["x-ratelimit-remaining"]) >= 0
        assert "x-ratelimit-reset" in headers
    
    def test_different_clients_separate_limits(self):
        """Test that different clients have separate rate limits."""
        # This is hard to test with TestClient as it doesn't simulate different IPs
        # But we can test the client identification logic
        middleware = RateLimitMiddleware(None, requests_per_hour=5)
        
        # Mock requests from different IPs
        request1 = Mock()
        request1.client.host = "192.168.1.1"
        request1.headers = {}
        
        request2 = Mock()
        request2.client.host = "192.168.1.2"
        request2.headers = {}
        
        client_id1 = middleware._get_client_identifier(request1)
        client_id2 = middleware._get_client_identifier(request2)
        
        assert client_id1 != client_id2
        assert "192.168.1.1" in client_id1
        assert "192.168.1.2" in client_id2
    
    def test_api_key_based_identification(self):
        """Test client identification with API keys."""
        middleware = RateLimitMiddleware(None, requests_per_hour=5)
        
        request = Mock()
        request.client.host = "192.168.1.1"
        request.headers = {"Authorization": "Bearer abc123xyz"}
        
        client_id = middleware._get_client_identifier(request)
        assert "api_key:" in client_id
        assert "abc123" in client_id  # Should include part of the token


class TestRequestLoggingMiddleware:
    """Test request logging middleware."""
    
    def setup_method(self):
        """Set up test app with request logging."""
        self.app = FastAPI()
        
        @self.app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        @self.app.post("/test")
        async def test_post():
            return {"message": "posted"}
        
        @self.app.get("/error")
        async def error_endpoint():
            raise HTTPException(500, "Test error")
        
        self.app.add_middleware(RequestLoggingMiddleware)
        self.client = TestClient(self.app)
    
    @patch('src.api.middleware.security.logging.getLogger')
    def test_request_logging(self, mock_get_logger):
        """Test that requests are logged."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        response = self.client.get("/test")
        assert response.status_code == 200
        
        # Should have logged at least once
        assert mock_logger.info.called
    
    def test_request_id_header(self):
        """Test that request ID is added to response."""
        response = self.client.get("/test")
        headers = response.headers
        
        assert "x-request-id" in headers
        request_id = headers["x-request-id"]
        assert request_id.startswith("req_")
        assert len(request_id) > 10  # Should be reasonably long
    
    def test_process_time_header(self):
        """Test that process time is added to response."""
        response = self.client.get("/test")
        headers = response.headers
        
        assert "x-process-time" in headers
        process_time = float(headers["x-process-time"])
        assert process_time >= 0
        assert process_time < 1.0  # Should be very fast for test endpoint
    
    @patch('src.api.middleware.security.logging.getLogger')
    def test_error_logging(self, mock_get_logger):
        """Test that errors are logged."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        response = self.client.get("/error")
        assert response.status_code == 500
        
        # Error should still be logged
        assert mock_logger.info.called or mock_logger.error.called


class TestAPIKeyMiddleware:
    """Test API key authentication middleware."""
    
    def setup_method(self):
        """Set up test app with API key middleware."""
        self.app = FastAPI()
        
        @self.app.get("/test")
        async def test_endpoint():
            return {"message": "authenticated"}
        
        @self.app.get("/public")
        async def public_endpoint():
            return {"message": "public"}
        
        # Configure with test API keys
        test_api_keys = {
            "test_key_123": {
                "name": "Test Key",
                "active": True,
                "permissions": ["read"]
            },
            "inactive_key": {
                "name": "Inactive Key",
                "active": False,
                "permissions": ["read"]
            }
        }
        
        self.app.add_middleware(APIKeyMiddleware, api_keys=test_api_keys)
        self.client = TestClient(self.app)
    
    def test_public_endpoint_no_auth_required(self):
        """Test that public endpoints don't require authentication."""
        # Health endpoint should be public
        response = self.client.get("/health")
        # Might be 404 (not found) but shouldn't be 401 (unauthorized)
        assert response.status_code != 401
    
    def test_protected_endpoint_requires_auth(self):
        """Test that protected endpoints require authentication."""
        response = self.client.get("/test")
        assert response.status_code == 401
        
        data = response.json()
        assert "API key required" in data["detail"]
    
    def test_valid_api_key_allows_access(self):
        """Test that valid API key allows access."""
        headers = {"Authorization": "Bearer test_key_123"}
        response = self.client.get("/test", headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "authenticated"
    
    def test_invalid_api_key_blocks_access(self):
        """Test that invalid API key blocks access."""
        headers = {"Authorization": "Bearer invalid_key"}
        response = self.client.get("/test", headers=headers)
        assert response.status_code == 401
        
        data = response.json()
        assert "Invalid API key" in data["detail"]
    
    def test_inactive_api_key_blocks_access(self):
        """Test that inactive API key blocks access."""
        headers = {"Authorization": "Bearer inactive_key"}
        response = self.client.get("/test", headers=headers)
        assert response.status_code == 401
        
        data = response.json()
        assert "API key is disabled" in data["detail"]
    
    def test_api_key_in_header(self):
        """Test API key in X-API-Key header."""
        headers = {"X-API-Key": "test_key_123"}
        response = self.client.get("/test", headers=headers)
        assert response.status_code == 200
    
    def test_api_key_in_query_param(self):
        """Test API key in query parameter."""
        response = self.client.get("/test?api_key=test_key_123")
        assert response.status_code == 200
    
    def test_no_api_keys_configured(self):
        """Test behavior when no API keys are configured."""
        # Create new app without API keys
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "no auth required"}
        
        app.add_middleware(APIKeyMiddleware)  # No api_keys parameter
        client = TestClient(app)
        
        # Should allow access when no keys are configured
        response = client.get("/test")
        assert response.status_code == 200


class TestAPIKeyUtilities:
    """Test API key utility functions."""
    
    def test_generate_api_key(self):
        """Test API key generation."""
        key1 = generate_api_key()
        key2 = generate_api_key()
        
        # Keys should be different
        assert key1 != key2
        
        # Keys should be reasonable length
        assert len(key1) >= 32
        assert len(key2) >= 32
        
        # Keys should be URL-safe
        assert all(c.isalnum() or c in '-_' for c in key1)
        assert all(c.isalnum() or c in '-_' for c in key2)
    
    def test_hash_api_key(self):
        """Test API key hashing."""
        key = "test_api_key_123"
        
        # Hash the key
        hashed1 = hash_api_key(key)
        hashed2 = hash_api_key(key)
        
        # Hashes should be different due to salt
        assert hashed1 != hashed2
        
        # Hashes should contain salt
        assert ':' in hashed1
        assert ':' in hashed2
        
        # Should be able to verify both
        assert verify_api_key(key, hashed1)
        assert verify_api_key(key, hashed2)
    
    def test_verify_api_key(self):
        """Test API key verification."""
        key = "test_api_key_123"
        wrong_key = "wrong_key_456"
        
        hashed = hash_api_key(key)
        
        # Correct key should verify
        assert verify_api_key(key, hashed)
        
        # Wrong key should not verify
        assert not verify_api_key(wrong_key, hashed)
        
        # Invalid hash format should not verify
        assert not verify_api_key(key, "invalid_hash")
    
    def test_hash_with_custom_salt(self):
        """Test hashing with custom salt."""
        key = "test_key"
        salt = "custom_salt"
        
        hashed1 = hash_api_key(key, salt)
        hashed2 = hash_api_key(key, salt)
        
        # Same salt should produce same hash
        assert hashed1 == hashed2
        assert verify_api_key(key, hashed1)


class TestMiddlewareIntegration:
    """Test middleware working together."""
    
    def setup_method(self):
        """Set up app with multiple middleware."""
        self.app = FastAPI()
        
        @self.app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        # Add multiple middleware (order matters)
        self.app.add_middleware(SecurityHeadersMiddleware)
        self.app.add_middleware(RateLimitMiddleware, requests_per_hour=10, use_redis=False)
        self.app.add_middleware(RequestLoggingMiddleware)
        
        self.client = TestClient(self.app)
    
    def test_all_middleware_active(self):
        """Test that all middleware is active and working together."""
        response = self.client.get("/test")
        assert response.status_code == 200
        
        headers = response.headers
        
        # Security headers should be present
        assert "x-content-type-options" in headers
        
        # Rate limit headers should be present
        assert "x-ratelimit-limit" in headers
        
        # Request logging headers should be present
        assert "x-request-id" in headers
        assert "x-process-time" in headers
    
    def test_middleware_order_preserved(self):
        """Test that middleware execution order is preserved."""
        # This is more of an integration test
        # The fact that we get all expected headers suggests correct order
        response = self.client.get("/test")
        assert response.status_code == 200
        
        # All expected functionality should work
        headers = response.headers
        expected_headers = [
            "x-content-type-options",
            "x-ratelimit-limit", 
            "x-request-id"
        ]
        
        for header in expected_headers:
            assert header in headers


class TestMiddlewareErrorHandling:
    """Test middleware error handling."""
    
    def test_middleware_handles_downstream_errors(self):
        """Test that middleware properly handles errors from downstream."""
        app = FastAPI()
        
        @app.get("/error")
        async def error_endpoint():
            raise Exception("Test error")
        
        app.add_middleware(SecurityHeadersMiddleware)
        app.add_middleware(RequestLoggingMiddleware)
        
        client = TestClient(app)
        
        response = client.get("/error")
        assert response.status_code == 500
        
        # Security headers should still be present on error responses
        headers = response.headers
        assert "x-content-type-options" in headers
    
    @patch('src.api.middleware.security.logging.getLogger')
    def test_middleware_error_logging(self, mock_get_logger):
        """Test that middleware errors are logged."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # This is harder to test directly, but we can verify logging setup
        middleware = RequestLoggingMiddleware(None)
        assert middleware.logger is not None


if __name__ == "__main__":
    pytest.main([__file__])