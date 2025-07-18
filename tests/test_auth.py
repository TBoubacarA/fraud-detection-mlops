"""
Tests pour le module d'authentification
"""

import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from jose import jwt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.api.auth import AuthService, UserInDB, UserCreate, UserRole, SECRET_KEY, ALGORITHM
from src.api.scoring_api import app

client = TestClient(app)

class TestAuthService:
    """Tests pour AuthService"""
    
    def test_password_hashing(self):
        """Test du hachage des mots de passe"""
        password = "test_password_123"
        hashed = AuthService.get_password_hash(password)
        
        assert hashed != password
        assert AuthService.verify_password(password, hashed)
        assert not AuthService.verify_password("wrong_password", hashed)
    
    def test_get_user_existing(self):
        """Test de récupération d'un utilisateur existant"""
        user = AuthService.get_user("admin")
        
        assert user is not None
        assert user.username == "admin"
        assert user.role == "admin"
        assert user.is_active is True
    
    def test_get_user_nonexistent(self):
        """Test de récupération d'un utilisateur inexistant"""
        user = AuthService.get_user("nonexistent_user")
        assert user is None
    
    def test_authenticate_user_success(self):
        """Test d'authentification réussie"""
        user = AuthService.authenticate_user("admin", "secret")
        
        assert user is not False
        assert isinstance(user, UserInDB)
        assert user.username == "admin"
    
    def test_authenticate_user_wrong_password(self):
        """Test d'authentification avec mauvais mot de passe"""
        user = AuthService.authenticate_user("admin", "wrong_password")
        assert user is False
    
    def test_authenticate_user_nonexistent(self):
        """Test d'authentification avec utilisateur inexistant"""
        user = AuthService.authenticate_user("nonexistent", "password")
        assert user is False
    
    def test_create_access_token(self):
        """Test de création de token JWT"""
        data = {"sub": "testuser", "role": "analyst"}
        token = AuthService.create_access_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Vérifier le contenu du token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "testuser"
        assert payload["role"] == "analyst"
        assert "exp" in payload
    
    def test_create_access_token_with_expiration(self):
        """Test de création de token avec expiration personnalisée"""
        data = {"sub": "testuser", "role": "analyst"}
        expires_delta = timedelta(minutes=15)
        token = AuthService.create_access_token(data, expires_delta)
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        exp_time = datetime.fromtimestamp(payload["exp"])
        expected_time = datetime.utcnow() + expires_delta
        
        # Vérifier que l'expiration est proche de l'attendu (± 1 minute)
        assert abs((exp_time - expected_time).total_seconds()) < 60
    
    def test_verify_token_valid(self):
        """Test de vérification d'un token valide"""
        data = {"sub": "testuser", "role": "analyst"}
        token = AuthService.create_access_token(data)
        
        token_data = AuthService.verify_token(token)
        assert token_data.username == "testuser"
        assert token_data.role == "analyst"
    
    def test_verify_token_invalid(self):
        """Test de vérification d'un token invalide"""
        with pytest.raises(Exception):  # HTTPException
            AuthService.verify_token("invalid_token")
    
    def test_verify_token_expired(self):
        """Test de vérification d'un token expiré"""
        data = {"sub": "testuser", "role": "analyst"}
        expires_delta = timedelta(seconds=-1)  # Expiré
        token = AuthService.create_access_token(data, expires_delta)
        
        with pytest.raises(Exception):  # HTTPException
            AuthService.verify_token(token)
    
    def test_create_user_success(self):
        """Test de création d'utilisateur réussi"""
        user_data = UserCreate(
            username="newuser",
            email="newuser@test.com",
            password="NewPassword123!",
            role=UserRole.ANALYST
        )
        
        user = AuthService.create_user(user_data)
        
        assert user.username == "newuser"
        assert user.email == "newuser@test.com"
        assert user.role == UserRole.ANALYST
        assert user.is_active is True
        assert user.hashed_password != "NewPassword123!"
    
    def test_create_user_duplicate_username(self):
        """Test de création d'utilisateur avec nom existant"""
        user_data = UserCreate(
            username="admin",  # Utilisateur existant
            email="admin2@test.com",
            password="NewPassword123!",
            role=UserRole.ANALYST
        )
        
        with pytest.raises(Exception):  # HTTPException
            AuthService.create_user(user_data)

class TestAuthAPI:
    """Tests pour les endpoints d'authentification"""
    
    def test_login_success(self):
        """Test de connexion réussie"""
        response = client.post("/auth/login", params={
            "username": "admin",
            "password": "secret"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["role"] == "admin"
        assert "expires_in" in data
    
    def test_login_wrong_credentials(self):
        """Test de connexion avec mauvais identifiants"""
        response = client.post("/auth/login", params={
            "username": "admin",
            "password": "wrong_password"
        })
        
        assert response.status_code == 401
    
    def test_login_nonexistent_user(self):
        """Test de connexion avec utilisateur inexistant"""
        response = client.post("/auth/login", params={
            "username": "nonexistent",
            "password": "password"
        })
        
        assert response.status_code == 401
    
    def test_register_success_as_admin(self):
        """Test de création d'utilisateur par admin"""
        # D'abord, se connecter en tant qu'admin
        login_response = client.post("/auth/login", params={
            "username": "admin",
            "password": "secret"
        })
        token = login_response.json()["access_token"]
        
        # Créer un nouvel utilisateur
        user_data = {
            "username": "testuser2",
            "email": "testuser2@test.com",
            "password": "TestPassword123!",
            "role": "analyst"
        }
        
        response = client.post(
            "/auth/register",
            json=user_data,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["username"] == "testuser2"
        assert data["email"] == "testuser2@test.com"
        assert data["role"] == "analyst"
        assert data["is_active"] is True
    
    def test_register_without_admin_role(self):
        """Test de création d'utilisateur sans droits admin"""
        # Se connecter en tant qu'analyst
        login_response = client.post("/auth/login", params={
            "username": "analyst",
            "password": "secret"
        })
        token = login_response.json()["access_token"]
        
        user_data = {
            "username": "testuser3",
            "email": "testuser3@test.com",
            "password": "TestPassword123!",
            "role": "analyst"
        }
        
        response = client.post(
            "/auth/register",
            json=user_data,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 403
    
    def test_register_without_token(self):
        """Test de création d'utilisateur sans token"""
        user_data = {
            "username": "testuser4",
            "email": "testuser4@test.com",
            "password": "TestPassword123!",
            "role": "analyst"
        }
        
        response = client.post("/auth/register", json=user_data)
        assert response.status_code == 403  # Pas de token
    
    def test_protected_endpoint_with_valid_token(self):
        """Test d'accès à un endpoint protégé avec token valide"""
        # Se connecter
        login_response = client.post("/auth/login", params={
            "username": "admin",
            "password": "secret"
        })
        token = login_response.json()["access_token"]
        
        # Accéder à un endpoint protégé
        response = client.get(
            "/model/info",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code in [200, 503]  # 503 si modèle non chargé
    
    def test_protected_endpoint_without_token(self):
        """Test d'accès à un endpoint protégé sans token"""
        response = client.get("/model/info")
        assert response.status_code == 403
    
    def test_protected_endpoint_with_invalid_token(self):
        """Test d'accès à un endpoint protégé avec token invalide"""
        response = client.get(
            "/model/info",
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == 401
    
    def test_admin_only_endpoint_with_admin(self):
        """Test d'accès à un endpoint admin avec compte admin"""
        # Se connecter en tant qu'admin
        login_response = client.post("/auth/login", params={
            "username": "admin",
            "password": "secret"
        })
        token = login_response.json()["access_token"]
        
        # Accéder à un endpoint admin
        response = client.get(
            "/admin/users",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert "total_users" in data
    
    def test_admin_only_endpoint_with_analyst(self):
        """Test d'accès à un endpoint admin avec compte analyst"""
        # Se connecter en tant qu'analyst
        login_response = client.post("/auth/login", params={
            "username": "analyst",
            "password": "secret"
        })
        token = login_response.json()["access_token"]
        
        # Tenter d'accéder à un endpoint admin
        response = client.get(
            "/admin/users",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 403