"""
Authentication API endpoints.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import Dict, Any, List
from datetime import datetime, timezone

from ..auth.jwt_handler import (
    JWTHandler, UserCredentials, UserRegistration, UserProfile, 
    TokenResponse, InMemoryUserStore
)
from ..auth.dependencies import (
    get_jwt_handler, get_user_store, get_current_user, get_admin_user
)

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


@router.post("/register", summary="Register New User", response_model=UserProfile)
async def register_user(
    registration: UserRegistration,
    user_store: InMemoryUserStore = Depends(get_user_store)
) -> UserProfile:
    """
    Register a new user account.
    
    Args:
        registration: User registration data
        
    Returns:
        UserProfile containing user information
        
    Raises:
        HTTPException: If email is already registered
    """
    # Check if email already exists
    existing_user = user_store.get_user_by_email(registration.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email address is already registered"
        )
    
    # Create user
    user = user_store.create_user(registration)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user account"
        )
    
    return user


@router.post("/login", summary="User Login", response_model=TokenResponse)
async def login_user(
    credentials: UserCredentials,
    jwt_handler: JWTHandler = Depends(get_jwt_handler),
    user_store: InMemoryUserStore = Depends(get_user_store)
) -> TokenResponse:
    """
    Authenticate user and return access tokens.
    
    Args:
        credentials: User email and password
        
    Returns:
        TokenResponse with access and refresh tokens
        
    Raises:
        HTTPException: If credentials are invalid
    """
    # Authenticate user
    user = user_store.authenticate_user(credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Create tokens
    user_data = {
        "user_id": user.user_id,
        "email": user.email,
        "username": user.username,
        "roles": user.roles
    }
    
    access_token = jwt_handler.create_access_token(user_data)
    refresh_token = jwt_handler.create_refresh_token(user_data)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=jwt_handler.access_token_expire_minutes * 60,
        user_info=user
    )


@router.post("/refresh", summary="Refresh Access Token", response_model=TokenResponse)
async def refresh_token(
    refresh_token: str,
    jwt_handler: JWTHandler = Depends(get_jwt_handler),
    user_store: InMemoryUserStore = Depends(get_user_store)
) -> TokenResponse:
    """
    Refresh access token using refresh token.
    
    Args:
        refresh_token: Valid refresh token
        
    Returns:
        TokenResponse with new access and refresh tokens
        
    Raises:
        HTTPException: If refresh token is invalid
    """
    # Verify refresh token
    token_data = jwt_handler.verify_refresh_token(refresh_token)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    # Get user
    user = user_store.get_user_by_id(token_data["user_id"])
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Revoke old refresh token
    jwt_handler.revoke_token(token_data["jti"])
    
    # Create new tokens
    user_data = {
        "user_id": user.user_id,
        "email": user.email,
        "username": user.username,
        "roles": user.roles
    }
    
    new_access_token = jwt_handler.create_access_token(user_data)
    new_refresh_token = jwt_handler.create_refresh_token(user_data)
    
    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        expires_in=jwt_handler.access_token_expire_minutes * 60,
        user_info=user
    )


@router.post("/logout", summary="User Logout")
async def logout_user(
    current_user: UserProfile = Depends(get_current_user),
    jwt_handler: JWTHandler = Depends(get_jwt_handler)
) -> Dict[str, str]:
    """
    Logout user and revoke tokens.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Success message
    """
    # In a full implementation, we would revoke all user tokens
    # For now, just log the logout
    jwt_handler.revoke_all_user_tokens(current_user.user_id)
    
    return {"message": "Successfully logged out"}


@router.get("/me", summary="Get Current User", response_model=UserProfile)
async def get_current_user_info(
    current_user: UserProfile = Depends(get_current_user)
) -> UserProfile:
    """
    Get current user information.
    
    Returns:
        UserProfile of the authenticated user
    """
    return current_user


@router.put("/me", summary="Update Current User", response_model=UserProfile)
async def update_current_user(
    updates: Dict[str, Any],
    current_user: UserProfile = Depends(get_current_user),
    user_store: InMemoryUserStore = Depends(get_user_store)
) -> UserProfile:
    """
    Update current user information.
    
    Args:
        updates: Fields to update
        current_user: Current authenticated user
        
    Returns:
        Updated UserProfile
    """
    # Only allow certain fields to be updated
    allowed_fields = {"username", "full_name"}
    filtered_updates = {
        key: value for key, value in updates.items() 
        if key in allowed_fields
    }
    
    if not filtered_updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid fields to update"
        )
    
    updated_user = user_store.update_user(current_user.user_id, filtered_updates)
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )
    
    return updated_user


@router.get("/users", summary="List Users (Admin Only)", response_model=List[UserProfile])
async def list_users(
    admin_user: UserProfile = Depends(get_admin_user),
    user_store: InMemoryUserStore = Depends(get_user_store)
) -> List[UserProfile]:
    """
    List all users (admin only).
    
    Returns:
        List of all user profiles
    """
    return user_store.list_users()


@router.get("/users/{user_id}", summary="Get User by ID (Admin Only)", response_model=UserProfile)
async def get_user_by_id(
    user_id: str,
    admin_user: UserProfile = Depends(get_admin_user),
    user_store: InMemoryUserStore = Depends(get_user_store)
) -> UserProfile:
    """
    Get user by ID (admin only).
    
    Args:
        user_id: User ID to retrieve
        
    Returns:
        UserProfile of the requested user
        
    Raises:
        HTTPException: If user not found
    """
    user = user_store.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user


@router.put("/users/{user_id}", summary="Update User (Admin Only)", response_model=UserProfile)
async def update_user(
    user_id: str,
    updates: Dict[str, Any],
    admin_user: UserProfile = Depends(get_admin_user),
    user_store: InMemoryUserStore = Depends(get_user_store)
) -> UserProfile:
    """
    Update user information (admin only).
    
    Args:
        user_id: User ID to update
        updates: Fields to update
        
    Returns:
        Updated UserProfile
        
    Raises:
        HTTPException: If user not found or update fails
    """
    # Admin can update more fields
    allowed_fields = {"username", "full_name", "is_active", "roles"}
    filtered_updates = {
        key: value for key, value in updates.items() 
        if key in allowed_fields
    }
    
    if not filtered_updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid fields to update"
        )
    
    updated_user = user_store.update_user(user_id, filtered_updates)
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return updated_user


@router.get("/status", summary="Authentication Status")
async def auth_status() -> Dict[str, Any]:
    """
    Get authentication system status.
    
    Returns:
        Authentication system status information
    """
    return {
        "status": "active",
        "jwt_enabled": True,
        "registration_enabled": True,
        "default_token_expiry_minutes": 30,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }