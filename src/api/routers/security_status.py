"""
Security status and monitoring endpoints.
Provides security health checks, configuration status, and audit information.
"""

from fastapi import APIRouter, HTTPException, Depends, status, Request
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
import logging

from ...api.middleware.security_config import get_security_config, SecurityAuditor
from ...api.middleware.api_monitoring import get_usage_tracker
from ...utils.audit_logger import get_audit_logger, AuditCategory

router = APIRouter(
    prefix="/api/v1/security",
    tags=["Security"],
    responses={404: {"description": "Not found"}}
)

logger = logging.getLogger(__name__)


@router.get("/status")
async def get_security_status(request: Request) -> Dict[str, Any]:
    """Get overall security status."""
    security_config = get_security_config()
    auditor = SecurityAuditor(security_config)
    usage_tracker = get_usage_tracker()
    
    # Audit the current configuration
    audit_result = auditor.audit_configuration()
    
    # Get recent security metrics
    recent_stats = usage_tracker.get_usage_statistics(hours=24)
    performance_alerts = usage_tracker.get_performance_alerts(severity="high", limit=10)
    rate_limit_analytics = usage_tracker.get_rate_limit_analytics()
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": "secure" if audit_result["audit_passed"] else "needs_attention",
        "security_score": audit_result["score"],
        "security_grade": audit_result.get("grade", "Unknown"),
        "configuration_audit": audit_result,
        "recent_metrics": {
            "total_requests_24h": recent_stats.total_requests,
            "error_rate_24h": recent_stats.error_requests / max(recent_stats.total_requests, 1),
            "unique_users_24h": recent_stats.unique_users,
            "unique_ips_24h": recent_stats.unique_ips,
        },
        "security_incidents": {
            "high_severity_alerts": len([a for a in performance_alerts if a.get("severity") == "high"]),
            "rate_limit_violations": rate_limit_analytics.get("total_violations", 0),
            "recent_violations": rate_limit_analytics.get("recent_violations", 0)
        },
        "security_features": {
            "rate_limiting": security_config.settings.rate_limit_enabled,
            "input_validation": security_config.settings.enable_input_sanitization,
            "audit_logging": security_config.settings.enable_audit_logging,
            "https_enforcement": security_config.settings.https_only,
            "jwt_authentication": True,  # Always enabled
        }
    }


@router.get("/config")
async def get_security_configuration(request: Request) -> Dict[str, Any]:
    """Get current security configuration (sanitized)."""
    security_config = get_security_config()
    
    # Return sanitized configuration (no secrets)
    config = {
        "rate_limiting": {
            "enabled": security_config.settings.rate_limit_enabled,
            "requests_per_hour": security_config.settings.rate_limit_requests_per_hour,
            "burst_size": security_config.settings.rate_limit_burst_size
        },
        "authentication": {
            "jwt_algorithm": security_config.settings.jwt_algorithm,
            "access_token_expire_minutes": security_config.settings.jwt_access_token_expire_minutes,
            "refresh_token_expire_days": security_config.settings.jwt_refresh_token_expire_days
        },
        "password_policy": {
            "min_length": security_config.settings.min_password_length,
            "require_uppercase": security_config.settings.require_password_uppercase,
            "require_lowercase": security_config.settings.require_password_lowercase,
            "require_digit": security_config.settings.require_password_digit,
            "require_special": security_config.settings.require_password_special
        },
        "security_headers": {
            "hsts_enabled": security_config.settings.enable_hsts,
            "hsts_max_age": security_config.settings.hsts_max_age,
            "csp_enabled": security_config.settings.enable_csp,
        },
        "input_validation": {
            "enabled": security_config.settings.enable_input_sanitization,
            "max_request_size": security_config.settings.max_request_size,
            "max_string_length": security_config.settings.max_string_length
        },
        "audit_logging": {
            "enabled": security_config.settings.enable_audit_logging,
            "retention_days": security_config.settings.audit_log_retention_days
        },
        "https": {
            "enforced": security_config.settings.https_only,
            "port": security_config.settings.https_port
        }
    }
    
    # Log configuration access
    audit_logger = get_audit_logger()
    audit_logger.log_data_access(
        event_type="security_config_view",
        user_id=getattr(request.state, 'user_id', None),
        resource="security_configuration",
        action="read",
        details={"endpoint": "/api/v1/security/config"}
    )
    
    return config


@router.get("/audit")
async def run_security_audit(request: Request) -> Dict[str, Any]:
    """Run comprehensive security audit."""
    security_config = get_security_config()
    auditor = SecurityAuditor(security_config)
    
    # Run the audit
    audit_result = auditor.audit_configuration()
    
    # Log the audit
    audit_logger = get_audit_logger()
    audit_logger.log_data_access(
        event_type="security_audit_run",
        user_id=getattr(request.state, 'user_id', None),
        resource="security_audit",
        action="execute",
        details={"score": audit_result["score"], "grade": audit_result.get("grade")}
    )
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audit_result": audit_result,
        "recommendations": audit_result.get("recommendations", []),
        "next_audit_recommended": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    }


@router.get("/metrics")
async def get_security_metrics(
    request: Request,
    hours: int = 24
) -> Dict[str, Any]:
    """Get security-related metrics."""
    if hours > 168:  # Max 1 week
        hours = 168
    
    usage_tracker = get_usage_tracker()
    
    # Get usage statistics
    stats = usage_tracker.get_usage_statistics(hours=hours)
    
    # Get performance alerts
    alerts = usage_tracker.get_performance_alerts(limit=50)
    
    # Get rate limiting analytics
    rate_limit_analytics = usage_tracker.get_rate_limit_analytics()
    
    # Calculate security metrics
    error_rate = stats.error_requests / max(stats.total_requests, 1)
    avg_response_time = stats.avg_response_time
    
    security_score = 100
    if error_rate > 0.1:  # > 10% error rate
        security_score -= 20
    if avg_response_time > 1000:  # > 1 second average
        security_score -= 10
    if rate_limit_analytics.get("recent_violations", 0) > 10:
        security_score -= 15
    
    return {
        "period_hours": hours,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "security_score": max(0, security_score),
        "traffic_metrics": {
            "total_requests": stats.total_requests,
            "successful_requests": stats.successful_requests,
            "error_requests": stats.error_requests,
            "error_rate": error_rate,
            "avg_response_time_ms": avg_response_time,
            "unique_users": stats.unique_users,
            "unique_ips": stats.unique_ips
        },
        "security_incidents": {
            "total_alerts": len(alerts),
            "high_severity_alerts": len([a for a in alerts if a.get("severity") == "high"]),
            "medium_severity_alerts": len([a for a in alerts if a.get("severity") == "medium"]),
            "rate_limit_violations": rate_limit_analytics.get("total_violations", 0),
            "recent_violations": rate_limit_analytics.get("recent_violations", 0)
        },
        "top_endpoints": stats.top_endpoints,
        "error_breakdown": stats.error_breakdown,
        "rate_limiting": rate_limit_analytics
    }


@router.get("/alerts")
async def get_security_alerts(
    request: Request,
    severity: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """Get recent security alerts."""
    usage_tracker = get_usage_tracker()
    
    # Get performance alerts
    alerts = usage_tracker.get_performance_alerts(severity=severity, limit=limit)
    
    # Log alert access
    audit_logger = get_audit_logger()
    audit_logger.log_data_access(
        event_type="security_alerts_view",
        user_id=getattr(request.state, 'user_id', None),
        resource="security_alerts",
        action="read",
        details={"severity": severity, "limit": limit, "alert_count": len(alerts)}
    )
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_alerts": len(alerts),
        "severity_filter": severity,
        "alerts": alerts,
        "summary": {
            "high": len([a for a in alerts if a.get("severity") == "high"]),
            "medium": len([a for a in alerts if a.get("severity") == "medium"]),
            "low": len([a for a in alerts if a.get("severity") == "low"])
        }
    }


@router.get("/audit-logs")
async def get_audit_logs(
    request: Request,
    category: Optional[str] = None,
    user_id: Optional[str] = None,
    hours: int = 24,
    limit: int = 100
) -> Dict[str, Any]:
    """Get recent audit log entries."""
    audit_logger = get_audit_logger()
    
    # Parse category if provided
    audit_category = None
    if category:
        try:
            audit_category = AuditCategory(category.upper())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid category: {category}"
            )
    
    # Calculate time range
    start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    
    # Search audit logs
    events = audit_logger.search_events(
        category=audit_category,
        user_id=user_id,
        start_time=start_time,
        limit=limit
    )
    
    # Log this audit log access
    audit_logger.log_data_access(
        event_type="audit_logs_view",
        user_id=getattr(request.state, 'user_id', None),
        resource="audit_logs",
        action="read",
        details={
            "category": category,
            "target_user_id": user_id,
            "hours": hours,
            "results_count": len(events)
        }
    )
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "category": category,
            "user_id": user_id,
            "hours": hours,
            "limit": limit
        },
        "total_events": len(events),
        "events": events
    }


@router.get("/health")
async def security_health_check() -> Dict[str, Any]:
    """Quick security health check."""
    security_config = get_security_config()
    auditor = SecurityAuditor(security_config)
    
    # Quick audit
    audit_result = auditor.audit_configuration()
    
    # Check if critical security features are enabled
    critical_checks = {
        "https_enforced": security_config.settings.https_only,
        "rate_limiting_enabled": security_config.settings.rate_limit_enabled,
        "audit_logging_enabled": security_config.settings.enable_audit_logging,
        "input_validation_enabled": security_config.settings.enable_input_sanitization,
    }
    
    all_critical_passed = all(critical_checks.values())
    
    return {
        "status": "healthy" if audit_result["audit_passed"] and all_critical_passed else "degraded",
        "security_score": audit_result["score"],
        "critical_checks": critical_checks,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }