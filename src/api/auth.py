"""
Module d'authentification et d'autorisation pour l'API
Implémentation JWT avec gestion des utilisateurs et des rôles
"""

from datetime import datetime, timedelta
from typing import Optional, Union, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Contexte de chiffrement des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Schéma d'authentification
security = HTTPBearer()

class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"

class UserInDB(BaseModel):
    username: str
    email: str
    hashed_password: str
    role: UserRole
    is_active: bool = True
    created_at: datetime = datetime.utcnow()

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    role: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: UserRole = UserRole.VIEWER

class UserResponse(BaseModel):
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime

# Base de données utilisateurs en mémoire (à remplacer par une vraie DB)
fake_users_db = {
    "admin": {
        "username": "admin",
        "email": "admin@company.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "role": "admin",
        "is_active": True,
        "created_at": datetime.utcnow()
    },
    "analyst": {
        "username": "analyst",
        "email": "analyst@company.com", 
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "role": "analyst",
        "is_active": True,
        "created_at": datetime.utcnow()
    }
}

class AuthService:
    """Service d'authentification et d'autorisation"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Vérifie un mot de passe contre son hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Génère le hash d'un mot de passe"""
        return pwd_context.hash(password)
    
    @staticmethod
    def get_user(username: str) -> Optional[UserInDB]:
        """Récupère un utilisateur par son nom"""
        if username in fake_users_db:
            user_dict = fake_users_db[username]
            return UserInDB(**user_dict)
        return None
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> Union[UserInDB, bool]:
        """Authentifie un utilisateur"""
        user = AuthService.get_user(username)
        if not user:
            return False
        if not AuthService.verify_password(password, user.hashed_password):
            return False
        return user
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Crée un token JWT"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> TokenData:
        """Vérifie et décode un token JWT"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            role: str = payload.get("role")
            
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token invalide",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            token_data = TokenData(username=username, role=role)
            return token_data
        
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token invalide",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    @staticmethod
    def create_user(user_data: UserCreate) -> UserInDB:
        """Crée un nouvel utilisateur"""
        if user_data.username in fake_users_db:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Nom d'utilisateur déjà utilisé"
            )
        
        hashed_password = AuthService.get_password_hash(user_data.password)
        user_dict = {
            "username": user_data.username,
            "email": user_data.email,
            "hashed_password": hashed_password,
            "role": user_data.role,
            "is_active": True,
            "created_at": datetime.utcnow()
        }
        
        fake_users_db[user_data.username] = user_dict
        return UserInDB(**user_dict)

# Dépendances pour FastAPI
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserInDB:
    """Récupère l'utilisateur actuel à partir du token"""
    token_data = AuthService.verify_token(credentials.credentials)
    user = AuthService.get_user(username=token_data.username)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Utilisateur non trouvé",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Utilisateur inactif"
        )
    
    return user

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    """Récupère l'utilisateur actuel et vérifie qu'il est actif"""
    return current_user

def require_role(required_role: UserRole):
    """Décorateur pour vérifier le rôle d'un utilisateur"""
    def role_checker(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
        if current_user.role != required_role and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Rôle {required_role} requis"
            )
        return current_user
    return role_checker

def require_admin(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    """Vérifie que l'utilisateur est admin"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Droits administrateur requis"
        )
    return current_user

def require_analyst_or_admin(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    """Vérifie que l'utilisateur est analyst ou admin"""
    if current_user.role not in [UserRole.ANALYST, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Droits analyst ou admin requis"
        )
    return current_user

# Logging des accès
def log_access(user: UserInDB, endpoint: str, method: str):
    """Log les accès à l'API"""
    logger.info(f"Access: {user.username} ({user.role}) - {method} {endpoint}")