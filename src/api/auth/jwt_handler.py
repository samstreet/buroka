"""
JWT token handling for authentication.
"""

import jwt
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import secrets
from pydantic import BaseModel, EmailStr, Field
import logging


class TokenData(BaseModel):
    """JWT token data structure."""
    user_id: str
    email: str
    username: str
    roles: list[str] = Field(default_factory=list)
    exp: datetime
    iat: datetime
    jti: str  # JWT ID for token revocation


class UserCredentials(BaseModel):
    """User credentials for authentication."""
    email: EmailStr
    password: str = Field(min_length=8, max_length=100)


class UserRegistration(BaseModel):
    """User registration data."""
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=8, max_length=100)
    full_name: Optional[str] = Field(default=None, max_length=100)


class UserProfile(BaseModel):
    """User profile information."""
    user_id: str
    username: str
    email: str
    full_name: Optional[str] = None
    roles: list[str] = Field(default_factory=list)
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True


class TokenResponse(BaseModel):
    """Token response structure."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user_info: UserProfile


class JWTHandler:
    """JWT token handler for authentication."""
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.logger = logging.getLogger(__name__)
        
        # In production, this would be stored in Redis or database
        self.revoked_tokens: set[str] = set()
    
    def create_access_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT access token."""
        now = datetime.now(timezone.utc)
        expire = now + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "user_id": user_data["user_id"],
            "email": user_data["email"],
            "username": user_data["username"],
            "roles": user_data.get("roles", []),
            "exp": expire,
            "iat": now,
            "jti": secrets.token_urlsafe(16),
            "type": "access"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT refresh token."""
        now = datetime.now(timezone.utc)
        expire = now + timedelta(days=self.refresh_token_expire_days)
        
        payload = {
            "user_id": user_data["user_id"],
            "email": user_data["email"],
            "exp": expire,
            "iat": now,
            "jti": secrets.token_urlsafe(16),
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is revoked
            jti = payload.get("jti")
            if jti in self.revoked_tokens:
                self.logger.warning(f"Revoked token used: {jti}")
                return None
            
            # Check token type (should be access token for API calls)
            if payload.get("type") != "access":
                self.logger.warning(f"Wrong token type: {payload.get('type')}")
                return None
            
            # Convert datetime fields
            exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
            iat = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
            
            return TokenData(
                user_id=payload["user_id"],
                email=payload["email"],
                username=payload["username"],
                roles=payload.get("roles", []),
                exp=exp,
                iat=iat,
                jti=jti
            )
            
        except jwt.ExpiredSignatureError:
            self.logger.info("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {e}")
            return None
    
    def verify_refresh_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify refresh token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is revoked
            jti = payload.get("jti")
            if jti in self.revoked_tokens:
                return None
            
            # Check token type
            if payload.get("type") != "refresh":
                return None
            
            return {
                "user_id": payload["user_id"],
                "email": payload["email"],
                "jti": jti
            }
            
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None
    
    def revoke_token(self, jti: str) -> None:
        """Revoke a token by its JTI."""
        self.revoked_tokens.add(jti)
        self.logger.info(f"Token revoked: {jti}")
    
    def revoke_all_user_tokens(self, user_id: str) -> None:
        """Revoke all tokens for a user."""
        # In production, this would query the database for user tokens
        self.logger.info(f"All tokens revoked for user: {user_id}")


class PasswordHandler:
    """Password hashing and verification."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(
            password.encode('utf-8'), 
            hashed_password.encode('utf-8')
        )
    
    @staticmethod
    def generate_password(length: int = 12) -> str:
        """Generate a secure random password."""
        return secrets.token_urlsafe(length)


class InMemoryUserStore:
    """In-memory user store for development/testing."""
    
    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {}
        self.users_by_email: Dict[str, str] = {}  # email -> user_id mapping
        self.logger = logging.getLogger(__name__)
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self) -> None:
        """Create default admin user."""
        admin_id = "admin-001"
        admin_password = "admin123456"  # In production, this would be configurable
        
        self.users[admin_id] = {
            "user_id": admin_id,
            "username": "admin",
            "email": "admin@market-analysis.com",
            "password_hash": PasswordHandler.hash_password(admin_password),
            "full_name": "System Administrator",
            "roles": ["admin", "user"],
            "created_at": datetime.now(timezone.utc),
            "last_login": None,
            "is_active": True
        }
        
        self.users_by_email["admin@market-analysis.com"] = admin_id
        self.logger.info("Default admin user created")
    
    def create_user(self, registration: UserRegistration) -> Optional[UserProfile]:
        """Create a new user."""
        # Check if email already exists
        if registration.email in self.users_by_email:
            return None
        
        # Generate user ID
        user_id = f"user-{secrets.token_hex(8)}"
        
        # Hash password
        password_hash = PasswordHandler.hash_password(registration.password)
        
        # Create user record
        user_data = {
            "user_id": user_id,
            "username": registration.username,
            "email": registration.email,
            "password_hash": password_hash,
            "full_name": registration.full_name,
            "roles": ["user"],
            "created_at": datetime.now(timezone.utc),
            "last_login": None,
            "is_active": True
        }
        
        # Store user
        self.users[user_id] = user_data
        self.users_by_email[registration.email] = user_id
        
        self.logger.info(f"User created: {registration.username} ({registration.email})")
        
        return UserProfile(
            user_id=user_id,
            username=registration.username,
            email=registration.email,
            full_name=registration.full_name,
            roles=["user"],
            created_at=user_data["created_at"],
            is_active=True
        )
    
    def authenticate_user(self, credentials: UserCredentials) -> Optional[UserProfile]:
        """Authenticate user with email and password."""
        # Find user by email
        user_id = self.users_by_email.get(credentials.email)
        if not user_id:
            return None
        
        user_data = self.users.get(user_id)
        if not user_data or not user_data["is_active"]:
            return None
        
        # Verify password
        if not PasswordHandler.verify_password(credentials.password, user_data["password_hash"]):
            return None
        
        # Update last login
        user_data["last_login"] = datetime.now(timezone.utc)
        
        self.logger.info(f"User authenticated: {user_data['username']}")
        
        return UserProfile(
            user_id=user_data["user_id"],
            username=user_data["username"],
            email=user_data["email"],
            full_name=user_data["full_name"],
            roles=user_data["roles"],
            created_at=user_data["created_at"],
            last_login=user_data["last_login"],
            is_active=user_data["is_active"]
        )
    
    def get_user_by_id(self, user_id: str) -> Optional[UserProfile]:
        """Get user by ID."""
        user_data = self.users.get(user_id)
        if not user_data:
            return None
        
        return UserProfile(
            user_id=user_data["user_id"],
            username=user_data["username"],
            email=user_data["email"],
            full_name=user_data["full_name"],
            roles=user_data["roles"],
            created_at=user_data["created_at"],
            last_login=user_data["last_login"],
            is_active=user_data["is_active"]
        )
    
    def get_user_by_email(self, email: str) -> Optional[UserProfile]:
        """Get user by email."""
        user_id = self.users_by_email.get(email)
        if user_id:
            return self.get_user_by_id(user_id)
        return None
    
    def update_user(self, user_id: str, updates: Dict[str, Any]) -> Optional[UserProfile]:
        """Update user information."""
        if user_id not in self.users:
            return None
        
        user_data = self.users[user_id]
        
        # Update allowed fields
        allowed_fields = {"username", "full_name", "is_active", "roles"}
        for field, value in updates.items():
            if field in allowed_fields:
                user_data[field] = value
        
        return self.get_user_by_id(user_id)
    
    def list_users(self) -> list[UserProfile]:
        """List all users."""
        return [self.get_user_by_id(user_id) for user_id in self.users.keys()]