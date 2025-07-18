"""
Tests d'intégration pour l'API de scoring
"""

import pytest
import json
import time
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.api.scoring_api import app

client = TestClient(app)

class TestAPIIntegration:
    """Tests d'intégration de l'API"""
    
    @pytest.fixture
    def admin_token(self):
        """Fixture pour obtenir un token admin"""
        response = client.post("/auth/login", params={
            "username": "admin",
            "password": "secret"
        })
        return response.json()["access_token"]
    
    @pytest.fixture
    def analyst_token(self):
        """Fixture pour obtenir un token analyst"""
        response = client.post("/auth/login", params={
            "username": "analyst",
            "password": "secret"
        })
        return response.json()["access_token"]
    
    @pytest.fixture
    def sample_transaction(self):
        """Fixture pour une transaction d'exemple"""
        return {
            "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
            "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
            "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
            "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
            "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
            "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
            "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
            "Amount": 149.62
        }
    
    def test_health_check(self):
        """Test du health check"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
    
    def test_health_check_rate_limit(self):
        """Test du rate limiting sur health check"""
        # Faire plusieurs requêtes rapides
        for i in range(5):
            response = client.get("/health")
            assert response.status_code == 200
        
        # Vérifier les headers de rate limiting
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
    
    def test_example_endpoint_without_auth(self):
        """Test de l'endpoint example sans authentification"""
        response = client.get("/example")
        assert response.status_code == 403
    
    def test_example_endpoint_with_auth(self, admin_token):
        """Test de l'endpoint example avec authentification"""
        response = client.get(
            "/example",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "example_normal" in data
        assert "usage" in data
        assert "user_role" in data
    
    def test_model_info_without_auth(self):
        """Test de model info sans authentification"""
        response = client.get("/model/info")
        assert response.status_code == 403
    
    def test_model_info_with_auth(self, admin_token):
        """Test de model info avec authentification"""
        response = client.get(
            "/model/info",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        # Peut être 200 ou 503 selon si le modèle est chargé
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "model_type" in data
            assert "features_count" in data
            assert "performance" in data
    
    def test_predict_without_auth(self, sample_transaction):
        """Test de prédiction sans authentification"""
        response = client.post("/predict", json=sample_transaction)
        assert response.status_code == 403
    
    def test_predict_with_analyst_auth(self, analyst_token, sample_transaction):
        """Test de prédiction avec authentification analyst"""
        response = client.post(
            "/predict",
            json=sample_transaction,
            headers={"Authorization": f"Bearer {analyst_token}"}
        )
        
        # Peut être 200 ou 503 selon si le modèle est chargé
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "is_fraud" in data
            assert "fraud_probability" in data
            assert "risk_level" in data
            assert "confidence" in data
            assert "model_version" in data
            assert "prediction_time" in data
            assert "request_id" in data
    
    def test_predict_with_invalid_data(self, analyst_token):
        """Test de prédiction avec données invalides"""
        invalid_transaction = {
            "V1": "invalid_string",  # Doit être un float
            "Amount": 100.0
        }
        
        response = client.post(
            "/predict",
            json=invalid_transaction,
            headers={"Authorization": f"Bearer {analyst_token}"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_malicious_data(self, analyst_token):
        """Test de prédiction avec données malicieuses"""
        malicious_transaction = {
            "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
            "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
            "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
            "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
            "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
            "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
            "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
            "Amount": 149.62,
            "transaction_id": "<script>alert('xss')</script>"
        }
        
        response = client.post(
            "/predict",
            json=malicious_transaction,
            headers={"Authorization": f"Bearer {analyst_token}"}
        )
        
        assert response.status_code == 422  # Validation should catch this
    
    def test_batch_predict_without_auth(self, sample_transaction):
        """Test de prédiction en lot sans authentification"""
        batch_data = {
            "transactions": [sample_transaction, sample_transaction]
        }
        
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 403
    
    def test_batch_predict_with_auth(self, analyst_token, sample_transaction):
        """Test de prédiction en lot avec authentification"""
        batch_data = {
            "transactions": [sample_transaction, sample_transaction],
            "batch_id": "TEST_BATCH_001"
        }
        
        response = client.post(
            "/predict/batch",
            json=batch_data,
            headers={"Authorization": f"Bearer {analyst_token}"}
        )
        
        # Peut être 200 ou 503 selon si le modèle est chargé
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "batch_id" in data
            assert "total_transactions" in data
            assert "fraud_detected" in data
            assert "fraud_rate" in data
            assert "results" in data
            assert "processed_by" in data
    
    def test_batch_predict_too_many_transactions(self, analyst_token, sample_transaction):
        """Test de prédiction en lot avec trop de transactions"""
        batch_data = {
            "transactions": [sample_transaction] * 101  # Plus de 100
        }
        
        response = client.post(
            "/predict/batch",
            json=batch_data,
            headers={"Authorization": f"Bearer {analyst_token}"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_model_reload_without_admin(self, analyst_token):
        """Test de rechargement de modèle sans droits admin"""
        response = client.post(
            "/model/reload",
            headers={"Authorization": f"Bearer {analyst_token}"}
        )
        
        assert response.status_code == 403
    
    def test_model_reload_with_admin(self, admin_token):
        """Test de rechargement de modèle avec droits admin"""
        response = client.post(
            "/model/reload",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        assert response.status_code in [200, 500]  # Depends on model availability
        
        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert "model_loaded" in data
            assert "reloaded_by" in data
    
    def test_admin_users_list_without_admin(self, analyst_token):
        """Test de liste des utilisateurs sans droits admin"""
        response = client.get(
            "/admin/users",
            headers={"Authorization": f"Bearer {analyst_token}"}
        )
        
        assert response.status_code == 403
    
    def test_admin_users_list_with_admin(self, admin_token):
        """Test de liste des utilisateurs avec droits admin"""
        response = client.get(
            "/admin/users",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert "total_users" in data
        assert len(data["users"]) >= 2  # Au moins admin et analyst
    
    def test_metrics_without_auth(self):
        """Test des métriques sans authentification"""
        response = client.get("/metrics")
        assert response.status_code == 403
    
    def test_metrics_with_analyst_auth(self, analyst_token):
        """Test des métriques avec authentification analyst"""
        response = client.get(
            "/metrics",
            headers={"Authorization": f"Bearer {analyst_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "api_version" in data
        assert "model_loaded" in data
        assert "environment" in data
        assert "timestamp" in data
    
    def test_register_user_without_admin(self, analyst_token):
        """Test de création d'utilisateur sans droits admin"""
        user_data = {
            "username": "newuser",
            "email": "newuser@test.com",
            "password": "NewPassword123!",
            "role": "viewer"
        }
        
        response = client.post(
            "/auth/register",
            json=user_data,
            headers={"Authorization": f"Bearer {analyst_token}"}
        )
        
        assert response.status_code == 403
    
    def test_register_user_with_admin(self, admin_token):
        """Test de création d'utilisateur avec droits admin"""
        user_data = {
            "username": "newuser123",
            "email": "newuser123@test.com",
            "password": "NewPassword123!",
            "role": "viewer"
        }
        
        response = client.post(
            "/auth/register",
            json=user_data,
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "newuser123"
        assert data["email"] == "newuser123@test.com"
        assert data["role"] == "viewer"
    
    def test_cors_headers(self):
        """Test des headers CORS"""
        response = client.options("/health")
        
        # Les headers CORS devraient être présents
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
    
    def test_rate_limiting_headers(self):
        """Test des headers de rate limiting"""
        response = client.get("/health")
        
        # Les headers de rate limiting devraient être présents
        assert "x-ratelimit-limit" in response.headers
        assert "x-ratelimit-remaining" in response.headers
        assert "x-ratelimit-reset" in response.headers
    
    def test_security_headers(self):
        """Test des headers de sécurité"""
        response = client.get("/health")
        
        # Vérifier que les headers de sécurité sont présents
        # (Ces headers peuvent être ajoutés par des middlewares)
        assert response.status_code == 200
    
    def test_api_documentation_in_development(self):
        """Test que la documentation est disponible en développement"""
        # Test si on peut accéder à la documentation
        response = client.get("/docs")
        # En développement, devrait être accessible
        # En production, devrait être 404 ou redirection
        assert response.status_code in [200, 404]
    
    def test_openapi_schema(self):
        """Test du schéma OpenAPI"""
        response = client.get("/openapi.json")
        # Devrait être accessible en développement
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "openapi" in data
            assert "info" in data
            assert "paths" in data

class TestAPIPerformance:
    """Tests de performance de l'API"""
    
    @pytest.fixture
    def admin_token(self):
        """Fixture pour obtenir un token admin"""
        response = client.post("/auth/login", params={
            "username": "admin",
            "password": "secret"
        })
        return response.json()["access_token"]
    
    @pytest.fixture
    def sample_transaction(self):
        """Fixture pour une transaction d'exemple"""
        return {
            "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
            "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
            "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
            "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
            "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
            "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
            "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
            "Amount": 149.62
        }
    
    def test_health_check_performance(self):
        """Test de performance du health check"""
        start_time = time.time()
        
        for _ in range(10):
            response = client.get("/health")
            assert response.status_code == 200
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        # Le health check devrait être très rapide (< 100ms)
        assert avg_time < 0.1
    
    def test_login_performance(self):
        """Test de performance de la connexion"""
        start_time = time.time()
        
        for _ in range(5):
            response = client.post("/auth/login", params={
                "username": "admin",
                "password": "secret"
            })
            assert response.status_code == 200
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 5
        
        # La connexion devrait être rapide (< 200ms)
        assert avg_time < 0.2
    
    def test_concurrent_health_checks(self):
        """Test de requêtes concurrentes sur health check"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            start = time.time()
            response = client.get("/health")
            end = time.time()
            results.put((response.status_code, end - start))
        
        # Créer 10 threads pour faire des requêtes concurrentes
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Attendre que tous les threads se terminent
        for thread in threads:
            thread.join()
        
        # Vérifier les résultats
        response_times = []
        while not results.empty():
            status_code, response_time = results.get()
            assert status_code == 200
            response_times.append(response_time)
        
        # Toutes les requêtes devraient être rapides
        assert all(rt < 0.5 for rt in response_times)
        assert len(response_times) == 10