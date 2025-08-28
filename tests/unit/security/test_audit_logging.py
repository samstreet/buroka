"""
Comprehensive tests for audit logging system.
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timezone
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from src.utils.audit_logger import (
    AuditLogger,
    AuditEvent,
    AuditLevel,
    AuditCategory,
    get_audit_logger,
    init_audit_logger,
    audit_login_attempt,
    audit_api_call,
    audit_security_violation
)


class TestAuditEvent:
    """Test AuditEvent dataclass."""
    
    def test_audit_event_creation(self):
        """Test creating audit events."""
        event = AuditEvent(
            timestamp="2024-01-01T12:00:00Z",
            level=AuditLevel.INFO,
            category=AuditCategory.AUTHENTICATION,
            event_type="login",
            user_id="user123",
            result="success"
        )
        
        assert event.timestamp == "2024-01-01T12:00:00Z"
        assert event.level == AuditLevel.INFO
        assert event.category == AuditCategory.AUTHENTICATION
        assert event.event_type == "login"
        assert event.user_id == "user123"
        assert event.result == "success"
    
    def test_audit_event_optional_fields(self):
        """Test audit event with optional fields."""
        event = AuditEvent(
            timestamp="2024-01-01T12:00:00Z",
            level=AuditLevel.WARNING,
            category=AuditCategory.SECURITY_VIOLATION,
            event_type="brute_force"
        )
        
        # Optional fields should be None
        assert event.user_id is None
        assert event.details is None
        assert event.ip_address is None


class TestAuditLogger:
    """Test AuditLogger class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log')
        self.temp_file.close()
        
        self.logger = AuditLogger(
            log_file=self.temp_file.name,
            enable_file_logging=True,
            log_sensitive_data=False
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_logger_initialization(self):
        """Test audit logger initialization."""
        assert self.logger.log_file == self.temp_file.name
        assert self.logger.enable_file_logging is True
        assert self.logger.log_sensitive_data is False
        assert self.logger.logger is not None
    
    def test_integrity_hash_creation(self):
        """Test integrity hash creation."""
        event_data = "test data"
        hash1 = self.logger._create_integrity_hash(event_data)
        hash2 = self.logger._create_integrity_hash(event_data)
        
        # Same data should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest
    
    def test_sensitive_data_sanitization(self):
        """Test sensitive data is sanitized."""
        sensitive_details = {
            "password": "secret123",
            "api_key": "abc123xyz789",
            "user_id": "user123",  # Not sensitive
            "email": "test@example.com"  # Not sensitive
        }
        
        sanitized = self.logger._sanitize_details(sensitive_details)
        
        # Sensitive fields should be redacted
        assert "***REDACTED***" in sanitized["password"]
        assert "***REDACTED***" in sanitized["api_key"]
        
        # Non-sensitive fields should remain
        assert sanitized["user_id"] == "user123"
        assert sanitized["email"] == "test@example.com"
    
    def test_log_authentication_success(self):
        """Test logging successful authentication."""
        self.logger.log_authentication(
            event_type="login",
            user_id="user123",
            ip_address="192.168.1.1",
            result="success",
            details={"method": "password"}
        )
        
        # Check that event was logged to file
        with open(self.temp_file.name, 'r') as f:
            log_content = f.read()
            assert "login" in log_content
            assert "user123" in log_content
            assert "success" in log_content
            assert "AUTHENTICATION" in log_content
    
    def test_log_authentication_failure(self):
        """Test logging failed authentication."""
        self.logger.log_authentication(
            event_type="login_failed",
            user_id="user123",
            ip_address="192.168.1.1",
            result="failure",
            details={"reason": "invalid_password"}
        )
        
        with open(self.temp_file.name, 'r') as f:
            log_content = f.read()
            log_data = json.loads(log_content.strip())
            
            assert log_data["level"] == "WARNING"  # Failed auth should be WARNING
            assert log_data["event_type"] == "login_failed"
            assert log_data["result"] == "failure"
    
    def test_log_authorization(self):
        """Test logging authorization events."""
        self.logger.log_authorization(
            event_type="access_check",
            user_id="user123",
            resource="/api/sensitive",
            action="read",
            result="denied",
            details={"reason": "insufficient_permissions"}
        )
        
        with open(self.temp_file.name, 'r') as f:
            log_content = f.read()
            log_data = json.loads(log_content.strip())
            
            assert log_data["category"] == "AUTHORIZATION"
            assert log_data["resource"] == "/api/sensitive"
            assert log_data["action"] == "read"
            assert log_data["result"] == "denied"
    
    def test_log_data_access(self):
        """Test logging data access events."""
        self.logger.log_data_access(
            event_type="query",
            user_id="user123",
            resource="market_data",
            action="read",
            details={"symbol": "AAPL", "fields": ["price", "volume"]},
            data_hash="abc123"
        )
        
        with open(self.temp_file.name, 'r') as f:
            log_content = f.read()
            log_data = json.loads(log_content.strip())
            
            assert log_data["category"] == "DATA_ACCESS"
            assert log_data["resource"] == "market_data"
            assert log_data["data_hash"] == "abc123"
            assert "AAPL" in str(log_data["details"])
    
    def test_log_security_violation(self):
        """Test logging security violations."""
        self.logger.log_security_violation(
            event_type="sql_injection_attempt",
            ip_address="192.168.1.100",
            details={"payload": "'; DROP TABLE users; --", "endpoint": "/api/search"},
            severity="high"
        )
        
        with open(self.temp_file.name, 'r') as f:
            log_content = f.read()
            log_data = json.loads(log_content.strip())
            
            assert log_data["level"] == "ERROR"  # High severity should be ERROR
            assert log_data["category"] == "SECURITY_VIOLATION"
            assert log_data["ip_address"] == "192.168.1.100"
    
    def test_log_api_access(self):
        """Test logging API access events."""
        self.logger.log_api_access(
            event_type="api_call",
            user_id="user123",
            api_key_id="key456",
            resource="/api/market-data/AAPL",
            action="GET",
            result="200",
            duration_ms=150.5,
            details={"response_size": 1024}
        )
        
        with open(self.temp_file.name, 'r') as f:
            log_content = f.read()
            log_data = json.loads(log_content.strip())
            
            assert log_data["category"] == "API_ACCESS"
            assert log_data["api_key_id"] == "key456"
            assert log_data["duration_ms"] == 150.5
            assert log_data["result"] == "200"
    
    def test_log_user_management(self):
        """Test logging user management events."""
        self.logger.log_user_management(
            event_type="user_created",
            user_id="admin123",
            target_user_id="newuser456",
            action="create",
            result="success",
            details={"roles": ["user"], "email": "new@example.com"}
        )
        
        with open(self.temp_file.name, 'r') as f:
            log_content = f.read()
            log_data = json.loads(log_content.strip())
            
            assert log_data["category"] == "USER_MANAGEMENT"
            assert log_data["user_id"] == "admin123"
            assert log_data["details"]["target_user_id"] == "newuser456"
    
    def test_log_financial_data_access(self):
        """Test logging financial data access (compliance)."""
        self.logger.log_financial_data_access(
            event_type="stock_price_query",
            user_id="trader123",
            symbol="AAPL",
            data_type="real_time_quote",
            action="read",
            details={"price": 150.25, "volume": 1000000}
        )
        
        with open(self.temp_file.name, 'r') as f:
            log_content = f.read()
            log_data = json.loads(log_content.strip())
            
            assert log_data["category"] == "FINANCIAL_DATA"
            assert log_data["resource"] == "AAPL"
            assert log_data["details"]["symbol"] == "AAPL"
            assert log_data["details"]["data_type"] == "real_time_quote"
    
    def test_log_configuration_change(self):
        """Test logging configuration changes."""
        self.logger.log_configuration_change(
            event_type="rate_limit_updated",
            user_id="admin123",
            resource="rate_limit_config",
            action="modify",
            old_value="1000",
            new_value="2000",
            details={"requests_per_hour": True}
        )
        
        with open(self.temp_file.name, 'r') as f:
            log_content = f.read()
            log_data = json.loads(log_content.strip())
            
            assert log_data["category"] == "CONFIGURATION_CHANGE"
            assert log_data["level"] == "WARNING"  # Config changes are WARNING level
            assert log_data["details"]["old_value"] == "1000"
            assert log_data["details"]["new_value"] == "2000"
    
    def test_integrity_verification(self):
        """Test audit log integrity verification."""
        # Log an event
        self.logger.log_authentication(
            event_type="login",
            user_id="user123",
            result="success"
        )
        
        # Read the logged event
        with open(self.temp_file.name, 'r') as f:
            log_content = f.read().strip()
            log_data = json.loads(log_content)
        
        # Extract integrity hash
        integrity_hash = log_data.pop("integrity_hash")
        
        # Verify integrity
        event_json = json.dumps(log_data, default=str, separators=(',', ':'))
        assert self.logger.verify_integrity(event_json, integrity_hash)
        
        # Verify tampering detection
        tampered_json = event_json.replace("user123", "hacker")
        assert not self.logger.verify_integrity(tampered_json, integrity_hash)
    
    def test_search_events(self):
        """Test searching audit events."""
        # Log multiple events
        self.logger.log_authentication("login", user_id="user1", result="success")
        self.logger.log_authentication("login", user_id="user2", result="failure")
        self.logger.log_api_access("api_call", user_id="user1", resource="/api/test")
        
        # Search by category
        auth_events = self.logger.search_events(category=AuditCategory.AUTHENTICATION)
        assert len(auth_events) == 2
        
        # Search by user
        user1_events = self.logger.search_events(user_id="user1")
        assert len(user1_events) == 2  # 1 auth + 1 api
        
        # Search with limit
        limited_events = self.logger.search_events(limit=1)
        assert len(limited_events) == 1
    
    def test_no_file_logging(self):
        """Test logger with file logging disabled."""
        logger = AuditLogger(enable_file_logging=False)
        
        # Should not create file handler
        logger.log_authentication("test", user_id="user123")
        
        # No assertion needed - just shouldn't crash
    
    @patch('builtins.open', side_effect=PermissionError())
    def test_file_write_error_handling(self, mock_open):
        """Test handling of file write errors."""
        logger = AuditLogger(log_file="nonexistent/path/audit.log")
        
        # Should not crash even if file cannot be written
        logger.log_authentication("test", user_id="user123")


class TestConvenienceFunctions:
    """Test convenience functions for common audit events."""
    
    @patch('src.utils.audit_logger.get_audit_logger')
    def test_audit_login_attempt(self, mock_get_logger):
        """Test audit_login_attempt convenience function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        audit_login_attempt("user123", True, "192.168.1.1", {"method": "password"})
        
        mock_logger.log_authentication.assert_called_once_with(
            event_type="login_attempt",
            user_id="user123",
            ip_address="192.168.1.1",
            result="success",
            details={"method": "password"}
        )
    
    @patch('src.utils.audit_logger.get_audit_logger')
    def test_audit_api_call(self, mock_get_logger):
        """Test audit_api_call convenience function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        audit_api_call(
            user_id="user123",
            endpoint="/api/test",
            method="GET",
            status_code=200,
            duration_ms=150.0,
            ip_address="192.168.1.1",
            api_key_id="key456"
        )
        
        mock_logger.log_api_access.assert_called_once_with(
            event_type="api_call",
            user_id="user123",
            api_key_id="key456",
            resource="/api/test",
            action="GET",
            result="200",
            duration_ms=150.0,
            details={"ip_address": "192.168.1.1"}
        )
    
    @patch('src.utils.audit_logger.get_audit_logger')
    def test_audit_security_violation(self, mock_get_logger):
        """Test audit_security_violation convenience function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        audit_security_violation(
            "sql_injection",
            "192.168.1.100",
            {"payload": "'; DROP TABLE users;"}
        )
        
        mock_logger.log_security_violation.assert_called_once_with(
            event_type="sql_injection",
            ip_address="192.168.1.100",
            details={"payload": "'; DROP TABLE users;"},
            severity="medium"
        )


class TestGlobalLoggerManagement:
    """Test global logger instance management."""
    
    def test_get_audit_logger(self):
        """Test get_audit_logger returns singleton."""
        logger1 = get_audit_logger()
        logger2 = get_audit_logger()
        
        # Should return same instance
        assert logger1 is logger2
    
    def test_init_audit_logger(self):
        """Test initializing global logger with custom settings."""
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        
        try:
            logger = init_audit_logger(
                log_file=temp_file.name,
                log_sensitive_data=True
            )
            
            assert logger.log_file == temp_file.name
            assert logger.log_sensitive_data is True
            
            # Should be same as get_audit_logger after init
            assert get_audit_logger() is logger
            
        finally:
            os.unlink(temp_file.name)


class TestAuditLevelsAndCategories:
    """Test audit levels and categories enums."""
    
    def test_audit_levels(self):
        """Test all audit levels are defined."""
        levels = list(AuditLevel)
        expected_levels = ["INFO", "WARNING", "ERROR", "CRITICAL", "SECURITY"]
        
        assert len(levels) == len(expected_levels)
        for level in expected_levels:
            assert AuditLevel(level) in levels
    
    def test_audit_categories(self):
        """Test all audit categories are defined."""
        categories = list(AuditCategory)
        expected_categories = [
            "AUTHENTICATION", "AUTHORIZATION", "DATA_ACCESS",
            "DATA_MODIFICATION", "SYSTEM_ACCESS", "CONFIGURATION_CHANGE",
            "SECURITY_VIOLATION", "API_ACCESS", "USER_MANAGEMENT",
            "FINANCIAL_DATA"
        ]
        
        assert len(categories) >= len(expected_categories)
        for category in expected_categories:
            assert AuditCategory(category) in categories


class TestSensitiveDataHandling:
    """Test handling of sensitive data in audit logs."""
    
    def test_sensitive_data_logged_when_enabled(self):
        """Test sensitive data is logged when explicitly enabled."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log')
        temp_file.close()
        
        try:
            logger = AuditLogger(
                log_file=temp_file.name,
                log_sensitive_data=True
            )
            
            logger.log_authentication(
                "login",
                user_id="user123",
                details={"password": "secret123"}
            )
            
            with open(temp_file.name, 'r') as f:
                log_content = f.read()
                assert "secret123" in log_content  # Should be logged when enabled
                
        finally:
            os.unlink(temp_file.name)
    
    def test_sensitive_data_redacted_by_default(self):
        """Test sensitive data is redacted by default."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log')
        temp_file.close()
        
        try:
            logger = AuditLogger(
                log_file=temp_file.name,
                log_sensitive_data=False  # Default behavior
            )
            
            logger.log_authentication(
                "login",
                user_id="user123",
                details={"password": "secret123", "token": "abc123xyz"}
            )
            
            with open(temp_file.name, 'r') as f:
                log_content = f.read()
                assert "secret123" not in log_content  # Should be redacted
                assert "***REDACTED***" in log_content
                
        finally:
            os.unlink(temp_file.name)


if __name__ == "__main__":
    pytest.main([__file__])