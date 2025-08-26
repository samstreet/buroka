"""
Authentication dependencies for FastAPI endpoints.
"""

from typing import Optional, List
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .jwt_handler import JWTHandler, TokenData, UserProfile, InMemoryUserStore
import os


# Global instances
_jwt_handler: Optional[JWTHandler] = None
_user_store: Optional[InMemoryUserStore] = None

# Security scheme
security = HTTPBearer(auto_error=False)


def get_jwt_handler() -> JWTHandler:
    """Get JWT handler instance."""
    global _jwt_handler
    if _jwt_handler is None:
        secret_key = os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
        _jwt_handler = JWTHandler(
            secret_key=secret_key,
            algorithm="HS256",
            access_token_expire_minutes=int(os.getenv("JWT_EXPIRATION_MINUTES", "30")),
            refresh_token_expire_days=int(os.getenv("JWT_REFRESH_DAYS", "7"))
        )
    return _jwt_handler


def get_user_store() -> InMemoryUserStore:
    """Get user store instance."""
    global _user_store
    if _user_store is None:
        _user_store = InMemoryUserStore()
    return _user_store


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    jwt_handler: JWTHandler = Depends(get_jwt_handler),
    user_store: InMemoryUserStore = Depends(get_user_store)
) -> UserProfile:
    """Get current authenticated user."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify token
    token_data = jwt_handler.verify_token(credentials.credentials)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from store
    user = user_store.get_user_by_id(token_data.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_current_active_user(
    current_user: UserProfile = Depends(get_current_user)
) -> UserProfile:
    """Get current active user (alias for backward compatibility)."""
    return current_user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    jwt_handler: JWTHandler = Depends(get_jwt_handler),
    user_store: InMemoryUserStore = Depends(get_user_store)
) -> Optional[UserProfile]:
    """Get current user if authenticated, None if not."""
    if not credentials:
        return None
    
    try:
        token_data = jwt_handler.verify_token(credentials.credentials)
        if not token_data:
            return None
        
        user = user_store.get_user_by_id(token_data.user_id)
        if not user or not user.is_active:
            return None
        
        return user
    except Exception:
        return None


def require_roles(required_roles: List[str]):
    """Create a dependency that requires specific roles."""
    async def check_roles(current_user: UserProfile = Depends(get_current_user)) -> UserProfile:
        user_roles = set(current_user.roles)
        required_roles_set = set(required_roles)
        
        if not required_roles_set.intersection(user_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {required_roles}"
            )
        
        return current_user
    
    return check_roles


def require_admin():
    """Dependency that requires admin role."""
    return require_roles(["admin"])


async def get_admin_user(current_user: UserProfile = Depends(require_admin())) -> UserProfile:
    """Get current user if they have admin role."""
    return current_user


class RoleChecker:
    """Role checker for more complex role requirements."""
    
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, current_user: UserProfile = Depends(get_current_user)) -> UserProfile:
        user_roles = set(current_user.roles)
        allowed_roles_set = set(self.allowed_roles)
        
        if not allowed_roles_set.intersection(user_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden. Required roles: {self.allowed_roles}"
            )
        
        return current_user


# Common role checkers
require_user_role = RoleChecker(["user", "admin"])
require_admin_role = RoleChecker(["admin"])