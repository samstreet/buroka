"""
Comprehensive audit logging system for security and compliance.
Tracks all sensitive operations and security events.
"""

import json
import logging
import hashlib
import hmac
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, asdict
import os
import secrets


class AuditLevel(str, Enum):
    """Audit logging levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"


class AuditCategory(str, Enum):
    """Categories of audit events."""
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    DATA_ACCESS = "DATA_ACCESS"
    DATA_MODIFICATION = "DATA_MODIFICATION"
    SYSTEM_ACCESS = "SYSTEM_ACCESS"
    CONFIGURATION_CHANGE = "CONFIGURATION_CHANGE"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    API_ACCESS = "API_ACCESS"
    USER_MANAGEMENT = "USER_MANAGEMENT"
    FINANCIAL_DATA = "FINANCIAL_DATA"


@dataclass
class AuditEvent:
    """Represents an audit event."""
    timestamp: str
    level: AuditLevel
    category: AuditCategory
    event_type: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    api_key_id: Optional[str] = None
    duration_ms: Optional[float] = None
    data_hash: Optional[str] = None
    integrity_hash: Optional[str] = None
    

class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(
        self,
        log_file: str = "logs/audit.log",
        enable_file_logging: bool = True,
        enable_structured_logging: bool = True,
        log_sensitive_data: bool = False,
        integrity_key: Optional[str] = None,
        max_details_length: int = 1000
    ):
        self.log_file = log_file
        self.enable_file_logging = enable_file_logging
        self.enable_structured_logging = enable_structured_logging
        self.log_sensitive_data = log_sensitive_data
        self.integrity_key = integrity_key or os.getenv('AUDIT_INTEGRITY_KEY', secrets.token_hex(32))
        self.max_details_length = max_details_length
        
        # Set up logging
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)
        
        # Ensure log directory exists
        if self.enable_file_logging:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Set up file handler
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # Use structured JSON format
            formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)
            
            if not self.logger.handlers:
                self.logger.addHandler(file_handler)
        
        self.system_logger = logging.getLogger(__name__)
        self.system_logger.info(f"Audit logger initialized: {self.log_file}")
    
    def _create_integrity_hash(self, event_data: str) -> str:
        """Create integrity hash for audit event."""
        return hmac.new(
            self.integrity_key.encode(),
            event_data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def _sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize sensitive information from details."""
        if not self.log_sensitive_data:
            sensitive_keys = {
                'password', 'secret', 'key', 'token', 'credential',
                'auth', 'session', 'cookie', 'signature', 'hash',
                'card_number', 'ssn', 'social_security', 'api_key'
            }
            
            sanitized = {}
            for key, value in details.items():
                key_lower = key.lower()
                if any(sensitive in key_lower for sensitive in sensitive_keys):
                    if isinstance(value, str) and len(value) > 4:
                        sanitized[key] = f"***{value[-4:]}"  # Show last 4 chars
                    else:
                        sanitized[key] = "***REDACTED***"
                else:
                    if isinstance(value, str) and len(value) > self.max_details_length:
                        sanitized[key] = f"{value[:self.max_details_length]}...TRUNCATED"
                    else:
                        sanitized[key] = value
            return sanitized
        
        return details
    
    def _log_event(self, event: AuditEvent) -> None:
        """Log an audit event."""
        try:
            # Convert to dict and sanitize
            event_dict = asdict(event)
            if event_dict.get('details'):
                event_dict['details'] = self._sanitize_details(event_dict['details'])
            
            # Create JSON representation
            event_json = json.dumps(event_dict, default=str, separators=(',', ':'))
            
            # Add integrity hash
            if self.integrity_key:
                event_dict['integrity_hash'] = self._create_integrity_hash(event_json)
                event_json = json.dumps(event_dict, default=str, separators=(',', ':'))
            
            # Log to file
            if self.enable_file_logging:
                self.logger.info(event_json)
            
            # Log to structured logger if available
            if self.enable_structured_logging:
                try:
                    from .logging_config import get_audit_logger
                    structured_logger = get_audit_logger()
                    if structured_logger:
                        structured_logger.info("audit_event", extra=event_dict)
                except ImportError:
                    pass
            
        except Exception as e:
            self.system_logger.error(f"Failed to log audit event: {e}")
    
    def log_authentication(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        result: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Log authentication events."""
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=AuditLevel.INFO if result == "success" else AuditLevel.WARNING,
            category=AuditCategory.AUTHENTICATION,
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            result=result,
            details=details,
            **kwargs
        )
        self._log_event(event)
    
    def log_authorization(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        result: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Log authorization events."""
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=AuditLevel.INFO if result == "allowed" else AuditLevel.WARNING,
            category=AuditCategory.AUTHORIZATION,
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            details=details,
            **kwargs
        )
        self._log_event(event)
    
    def log_data_access(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: str = "read",
        details: Optional[Dict[str, Any]] = None,
        data_hash: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log data access events."""
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=AuditLevel.INFO,
            category=AuditCategory.DATA_ACCESS,
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            details=details,
            data_hash=data_hash,
            **kwargs
        )
        self._log_event(event)
    
    def log_data_modification(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: str = "modify",
        details: Optional[Dict[str, Any]] = None,
        data_hash: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log data modification events."""
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=AuditLevel.INFO,
            category=AuditCategory.DATA_MODIFICATION,
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            details=details,
            data_hash=data_hash,
            **kwargs
        )
        self._log_event(event)
    
    def log_security_violation(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "medium",
        **kwargs
    ) -> None:
        """Log security violations."""
        level_map = {
            "low": AuditLevel.INFO,
            "medium": AuditLevel.WARNING,
            "high": AuditLevel.ERROR,
            "critical": AuditLevel.CRITICAL
        }
        
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level_map.get(severity, AuditLevel.SECURITY),
            category=AuditCategory.SECURITY_VIOLATION,
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            details=details,
            **kwargs
        )
        self._log_event(event)
    
    def log_api_access(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        result: Optional[str] = None,
        duration_ms: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Log API access events."""
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=AuditLevel.INFO,
            category=AuditCategory.API_ACCESS,
            event_type=event_type,
            user_id=user_id,
            api_key_id=api_key_id,
            resource=resource,
            action=action,
            result=result,
            duration_ms=duration_ms,
            details=details,
            **kwargs
        )
        self._log_event(event)
    
    def log_user_management(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        target_user_id: Optional[str] = None,
        action: Optional[str] = None,
        result: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Log user management events."""
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=AuditLevel.INFO,
            category=AuditCategory.USER_MANAGEMENT,
            event_type=event_type,
            user_id=user_id,
            action=action,
            result=result,
            details={**(details or {}), "target_user_id": target_user_id},
            **kwargs
        )
        self._log_event(event)
    
    def log_financial_data_access(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        symbol: Optional[str] = None,
        data_type: Optional[str] = None,
        action: str = "read",
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Log financial data access (special category for compliance)."""
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=AuditLevel.INFO,
            category=AuditCategory.FINANCIAL_DATA,
            event_type=event_type,
            user_id=user_id,
            resource=symbol,
            action=action,
            details={**(details or {}), "data_type": data_type, "symbol": symbol},
            **kwargs
        )
        self._log_event(event)
    
    def log_configuration_change(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: str = "modify",
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Log configuration changes."""
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=AuditLevel.WARNING,
            category=AuditCategory.CONFIGURATION_CHANGE,
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            details={
                **(details or {}),
                "old_value": old_value,
                "new_value": new_value
            },
            **kwargs
        )
        self._log_event(event)
    
    def verify_integrity(self, event_json: str, provided_hash: str) -> bool:
        """Verify the integrity of an audit event."""
        if not self.integrity_key:
            return True
        
        expected_hash = self._create_integrity_hash(event_json)
        return hmac.compare_digest(expected_hash, provided_hash)
    
    def search_events(
        self,
        category: Optional[AuditCategory] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search audit events (basic file-based implementation)."""
        if not self.enable_file_logging or not Path(self.log_file).exists():
            return []
        
        events = []
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        
                        # Apply filters
                        if category and event.get('category') != category:
                            continue
                        
                        if user_id and event.get('user_id') != user_id:
                            continue
                        
                        if start_time or end_time:
                            event_time = datetime.fromisoformat(event.get('timestamp', ''))
                            if start_time and event_time < start_time:
                                continue
                            if end_time and event_time > end_time:
                                continue
                        
                        events.append(event)
                        
                        if len(events) >= limit:
                            break
                            
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            self.system_logger.error(f"Error searching audit events: {e}")
        
        return events


# Global audit logger instance
_audit_logger = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def init_audit_logger(
    log_file: str = "logs/audit.log",
    enable_file_logging: bool = True,
    log_sensitive_data: bool = False,
    integrity_key: Optional[str] = None
) -> AuditLogger:
    """Initialize the global audit logger."""
    global _audit_logger
    _audit_logger = AuditLogger(
        log_file=log_file,
        enable_file_logging=enable_file_logging,
        log_sensitive_data=log_sensitive_data,
        integrity_key=integrity_key
    )
    return _audit_logger


# Convenience functions for common audit events
def audit_login_attempt(user_id: str, success: bool, ip_address: str, details: Dict[str, Any] = None):
    """Audit user login attempt."""
    logger = get_audit_logger()
    logger.log_authentication(
        event_type="login_attempt",
        user_id=user_id,
        ip_address=ip_address,
        result="success" if success else "failure",
        details=details
    )


def audit_api_call(
    user_id: str,
    endpoint: str,
    method: str,
    status_code: int,
    duration_ms: float,
    ip_address: str = None,
    api_key_id: str = None
):
    """Audit API call."""
    logger = get_audit_logger()
    logger.log_api_access(
        event_type="api_call",
        user_id=user_id,
        api_key_id=api_key_id,
        resource=endpoint,
        action=method,
        result=str(status_code),
        duration_ms=duration_ms,
        details={"ip_address": ip_address}
    )


def audit_security_violation(violation_type: str, ip_address: str, details: Dict[str, Any]):
    """Audit security violation."""
    logger = get_audit_logger()
    logger.log_security_violation(
        event_type=violation_type,
        ip_address=ip_address,
        details=details,
        severity="medium"
    )