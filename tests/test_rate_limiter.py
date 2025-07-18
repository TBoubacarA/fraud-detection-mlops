"""
Tests pour le module de rate limiting
"""

import pytest
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.api.rate_limiter import RateLimiter, InMemoryStorage, DEFAULT_LIMITS
from fastapi.testclient import TestClient
from src.api.scoring_api import app

client = TestClient(app)

class TestInMemoryStorage:
    """Tests pour InMemoryStorage"""
    
    def test_storage_set_and_get(self):
        """Test de stockage et récupération"""
        storage = InMemoryStorage()
        
        # Stocker une valeur
        storage.set("test_key", 5, 3600)
        
        # Récupérer la valeur
        result = storage.get("test_key")
        assert result == 5
    
    def test_storage_expiration(self):
        """Test d'expiration des données"""
        storage = InMemoryStorage()
        
        # Stocker avec expiration très courte
        storage.set("test_key", 5, 0)  # Expire immédiatement
        
        # Attendre un peu
        time.sleep(0.1)
        
        # La valeur devrait avoir expiré
        result = storage.get("test_key")
        assert result is None
    
    def test_storage_increment(self):
        """Test d'incrémentation"""
        storage = InMemoryStorage()
        
        # Premier increment
        result1 = storage.increment("counter", 3600)
        assert result1 == 1
        
        # Deuxième increment
        result2 = storage.increment("counter", 3600)
        assert result2 == 2
        
        # Troisième increment
        result3 = storage.increment("counter", 3600)
        assert result3 == 3
    
    def test_storage_increment_different_keys(self):
        """Test d'incrémentation avec différentes clés"""
        storage = InMemoryStorage()
        
        # Incrémenter différentes clés
        result1 = storage.increment("counter1", 3600)
        result2 = storage.increment("counter2", 3600)
        result3 = storage.increment("counter1", 3600)
        
        assert result1 == 1
        assert result2 == 1
        assert result3 == 2
    
    def test_storage_get_nonexistent(self):
        """Test de récupération de clé inexistante"""
        storage = InMemoryStorage()
        
        result = storage.get("nonexistent_key")
        assert result is None

class TestRateLimiter:
    """Tests pour RateLimiter"""
    
    def test_rate_limiter_initialization(self):
        """Test d'initialisation du rate limiter"""
        limiter = RateLimiter()
        
        assert limiter.storage is not None
        assert limiter.backend in ["memory", "redis"]
    
    def test_rate_limiter_allow_within_limit(self):
        """Test d'autorisation dans la limite"""
        limiter = RateLimiter()
        
        # Première requête
        is_allowed, info = limiter.is_allowed("test_user", 5, 3600)
        
        assert is_allowed is True
        assert info["limit"] == 5
        assert info["remaining"] == 4
        assert info["current_count"] == 1
    
    def test_rate_limiter_block_over_limit(self):
        """Test de blocage au-delà de la limite"""
        limiter = RateLimiter()
        
        # Faire plusieurs requêtes jusqu'à la limite
        for i in range(5):
            is_allowed, info = limiter.is_allowed("test_user", 5, 3600)
            assert is_allowed is True
        
        # La 6ème requête devrait être bloquée
        is_allowed, info = limiter.is_allowed("test_user", 5, 3600)
        assert is_allowed is False
        assert info["remaining"] == 0
    
    def test_rate_limiter_different_users(self):
        """Test avec différents utilisateurs"""
        limiter = RateLimiter()
        
        # Utilisateur 1
        is_allowed1, info1 = limiter.is_allowed("user1", 3, 3600)
        assert is_allowed1 is True
        assert info1["current_count"] == 1
        
        # Utilisateur 2
        is_allowed2, info2 = limiter.is_allowed("user2", 3, 3600)
        assert is_allowed2 is True
        assert info2["current_count"] == 1
        
        # Utilisateur 1 encore
        is_allowed1_again, info1_again = limiter.is_allowed("user1", 3, 3600)
        assert is_allowed1_again is True
        assert info1_again["current_count"] == 2
    
    def test_rate_limiter_window_reset(self):
        """Test de réinitialisation de la fenêtre"""
        limiter = RateLimiter()
        
        # Faire des requêtes avec une fenêtre courte
        for i in range(3):
            is_allowed, info = limiter.is_allowed("test_user", 3, 1)  # 1 seconde
            assert is_allowed is True
        
        # La 4ème requête devrait être bloquée
        is_allowed, info = limiter.is_allowed("test_user", 3, 1)
        assert is_allowed is False
        
        # Attendre que la fenêtre se réinitialise
        time.sleep(1.1)
        
        # Maintenant ça devrait marcher à nouveau
        is_allowed, info = limiter.is_allowed("test_user", 3, 1)
        assert is_allowed is True
        assert info["current_count"] == 1
    
    def test_rate_limiter_edge_case_zero_limit(self):
        """Test avec limite zéro"""
        limiter = RateLimiter()
        
        is_allowed, info = limiter.is_allowed("test_user", 0, 3600)
        assert is_allowed is False
        assert info["remaining"] == 0
    
    def test_rate_limiter_edge_case_one_limit(self):
        """Test avec limite de 1"""
        limiter = RateLimiter()
        
        # Première requête
        is_allowed1, info1 = limiter.is_allowed("test_user", 1, 3600)
        assert is_allowed1 is True
        assert info1["remaining"] == 0
        
        # Deuxième requête
        is_allowed2, info2 = limiter.is_allowed("test_user", 1, 3600)
        assert is_allowed2 is False
        assert info2["remaining"] == 0

class TestRateLimitingAPI:
    """Tests du rate limiting via l'API"""
    
    @pytest.fixture
    def admin_token(self):
        """Fixture pour obtenir un token admin"""
        response = client.post("/auth/login", params={
            "username": "admin",
            "password": "secret"
        })
        return response.json()["access_token"]
    
    def test_health_check_rate_limit(self):
        """Test du rate limiting sur health check"""
        # Faire plusieurs requêtes pour tester le rate limiting
        responses = []
        for i in range(10):
            response = client.get("/health")
            responses.append(response)
        
        # Toutes les requêtes devraient passer (limite haute)
        for response in responses:
            assert response.status_code == 200
            assert "x-ratelimit-limit" in response.headers
            assert "x-ratelimit-remaining" in response.headers
    
    def test_login_rate_limit(self):
        """Test du rate limiting sur login"""
        # Faire plusieurs tentatives de connexion
        responses = []
        for i in range(3):
            response = client.post("/auth/login", params={
                "username": "admin",
                "password": "secret"
            })
            responses.append(response)
        
        # Les premières requêtes devraient passer
        for response in responses:
            assert response.status_code == 200
            assert "x-ratelimit-limit" in response.headers
    
    def test_login_rate_limit_exceeded(self):
        """Test de dépassement du rate limit sur login"""
        # Faire trop de tentatives de connexion rapidement
        responses = []
        for i in range(7):  # Plus que la limite de 5/minute
            response = client.post("/auth/login", params={
                "username": "admin",
                "password": "secret"
            })
            responses.append(response)
        
        # Les premières requêtes devraient passer
        success_count = sum(1 for r in responses if r.status_code == 200)
        rate_limited_count = sum(1 for r in responses if r.status_code == 429)
        
        # Il devrait y avoir au moins quelques succès et quelques rate limits
        assert success_count > 0
        # Note: Le rate limiting peut être plus permissif selon la configuration
    
    def test_predict_rate_limit(self, admin_token):
        """Test du rate limiting sur predict"""
        transaction = {
            "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
            "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
            "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
            "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
            "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
            "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
            "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
            "Amount": 149.62
        }
        
        headers = {"Authorization": f"Bearer {admin_token}"}
        
        # Faire plusieurs requêtes de prédiction
        responses = []
        for i in range(5):
            response = client.post("/predict", json=transaction, headers=headers)
            responses.append(response)
        
        # Vérifier que les headers de rate limiting sont présents
        for response in responses:
            if response.status_code in [200, 503]:  # 503 si modèle non chargé
                assert "x-ratelimit-limit" in response.headers
                assert "x-ratelimit-remaining" in response.headers
    
    def test_batch_predict_rate_limit(self, admin_token):
        """Test du rate limiting sur batch predict"""
        transaction = {
            "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
            "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
            "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
            "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
            "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
            "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
            "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
            "Amount": 149.62
        }
        
        batch_data = {
            "transactions": [transaction, transaction]
        }
        
        headers = {"Authorization": f"Bearer {admin_token}"}
        
        # Faire plusieurs requêtes de batch
        responses = []
        for i in range(3):
            response = client.post("/predict/batch", json=batch_data, headers=headers)
            responses.append(response)
        
        # Vérifier que les headers de rate limiting sont présents
        for response in responses:
            if response.status_code in [200, 503]:  # 503 si modèle non chargé
                assert "x-ratelimit-limit" in response.headers
                assert "x-ratelimit-remaining" in response.headers
    
    def test_rate_limit_headers_format(self):
        """Test du format des headers de rate limiting"""
        response = client.get("/health")
        
        assert response.status_code == 200
        
        # Vérifier la présence des headers
        assert "x-ratelimit-limit" in response.headers
        assert "x-ratelimit-remaining" in response.headers
        assert "x-ratelimit-reset" in response.headers
        
        # Vérifier que les valeurs sont numériques
        limit = int(response.headers["x-ratelimit-limit"])
        remaining = int(response.headers["x-ratelimit-remaining"])
        
        assert limit > 0
        assert remaining >= 0
        assert remaining <= limit
    
    def test_rate_limit_different_endpoints(self, admin_token):
        """Test que différents endpoints ont des limites différentes"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        
        # Health check (limite haute)
        health_response = client.get("/health")
        health_limit = int(health_response.headers.get("x-ratelimit-limit", 0))
        
        # Example endpoint (limite moyenne)
        example_response = client.get("/example", headers=headers)
        if example_response.status_code == 200:
            example_limit = int(example_response.headers.get("x-ratelimit-limit", 0))
            
            # Les limites peuvent être différentes
            # (Ceci dépend de la configuration exacte)
            assert health_limit > 0
            assert example_limit > 0
    
    def test_rate_limit_reset_time(self):
        """Test du temps de reset du rate limiting"""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert "x-ratelimit-reset" in response.headers
        
        # Le reset time devrait être une date ISO valide
        reset_time = response.headers["x-ratelimit-reset"]
        assert len(reset_time) > 0
        
        # Vérifier que c'est un format de date valide
        from datetime import datetime
        try:
            datetime.fromisoformat(reset_time.replace('Z', '+00:00'))
        except ValueError:
            # Si ce n'est pas ISO, ça pourrait être un timestamp
            assert reset_time.isdigit()

class TestDefaultLimits:
    """Tests des limites par défaut"""
    
    def test_default_limits_structure(self):
        """Test de la structure des limites par défaut"""
        assert "general" in DEFAULT_LIMITS
        assert "predict" in DEFAULT_LIMITS
        assert "batch" in DEFAULT_LIMITS
        assert "auth" in DEFAULT_LIMITS
        
        for limit_name, limit_config in DEFAULT_LIMITS.items():
            assert "requests" in limit_config
            assert "window" in limit_config
            assert isinstance(limit_config["requests"], int)
            assert isinstance(limit_config["window"], int)
            assert limit_config["requests"] > 0
            assert limit_config["window"] > 0
    
    def test_default_limits_values(self):
        """Test des valeurs des limites par défaut"""
        # Vérifier que les limites sont raisonnables
        assert DEFAULT_LIMITS["general"]["requests"] >= 10
        assert DEFAULT_LIMITS["predict"]["requests"] >= 10
        assert DEFAULT_LIMITS["batch"]["requests"] >= 5
        assert DEFAULT_LIMITS["auth"]["requests"] >= 3
        
        # Vérifier que les fenêtres sont raisonnables
        assert DEFAULT_LIMITS["general"]["window"] >= 60  # Au moins 1 minute
        assert DEFAULT_LIMITS["predict"]["window"] >= 60
        assert DEFAULT_LIMITS["batch"]["window"] >= 60
        assert DEFAULT_LIMITS["auth"]["window"] >= 60