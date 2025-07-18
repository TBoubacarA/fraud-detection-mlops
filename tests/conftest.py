"""
Configuration partagée pour les tests
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Ajouter le répertoire source au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope="session")
def test_data_dir():
    """Crée un répertoire temporaire pour les données de test"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def test_models_dir():
    """Crée un répertoire temporaire pour les modèles de test"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def test_logs_dir():
    """Crée un répertoire temporaire pour les logs de test"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def clean_environment():
    """Nettoie l'environnement entre les tests"""
    # Sauvegarder les variables d'environnement
    original_env = os.environ.copy()
    
    # Configurer pour les tests
    os.environ["ENVIRONMENT"] = "test"
    os.environ["USE_REDIS"] = "false"
    os.environ["JWT_SECRET_KEY"] = "test_secret_key_do_not_use_in_production"
    
    yield
    
    # Restaurer l'environnement original
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture(scope="function")
def mock_model_data():
    """Données de test pour les modèles ML"""
    import pandas as pd
    import numpy as np
    
    # Créer des données de test
    np.random.seed(42)
    n_samples = 1000
    
    # Features V1-V28
    features = {}
    for i in range(1, 29):
        features[f"V{i}"] = np.random.normal(0, 1, n_samples)
    
    # Amount
    features["Amount"] = np.random.lognormal(0, 1, n_samples)
    
    # Class (0 = normal, 1 = fraud)
    features["Class"] = np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
    
    df = pd.DataFrame(features)
    return df

@pytest.fixture(scope="function")
def sample_transaction():
    """Transaction d'exemple pour les tests"""
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

@pytest.fixture(scope="function")
def sample_fraud_transaction():
    """Transaction frauduleuse d'exemple pour les tests"""
    return {
        "V1": -3.043541, "V2": -3.157307, "V3": 1.088463, "V4": 2.288644,
        "V5": 1.359805, "V6": -1.064823, "V7": 0.325574, "V8": -0.067794,
        "V9": -0.270533, "V10": -0.838587, "V11": 1.138865, "V12": 0.141266,
        "V13": 0.109663, "V14": -2.180014, "V15": 0.124205, "V16": 0.344462,
        "V17": 0.018298, "V18": -0.123794, "V19": 0.195267, "V20": -0.287924,
        "V21": -0.932391, "V22": 0.172726, "V23": 0.019318, "V24": 0.017709,
        "V25": -0.009431, "V26": 0.392205, "V27": -0.004421, "V28": -0.121404,
        "Amount": 9.99
    }

@pytest.fixture(scope="function")
def test_users():
    """Utilisateurs de test"""
    return {
        "admin": {
            "username": "admin",
            "password": "secret",
            "role": "admin"
        },
        "analyst": {
            "username": "analyst",
            "password": "secret",
            "role": "analyst"
        },
        "viewer": {
            "username": "viewer",
            "password": "secret",
            "role": "viewer"
        }
    }

@pytest.fixture(scope="function")
def mock_mlflow():
    """Mock MLflow pour les tests"""
    import unittest.mock as mock
    
    with mock.patch('mlflow.start_run') as mock_start_run, \
         mock.patch('mlflow.log_metrics') as mock_log_metrics, \
         mock.patch('mlflow.log_params') as mock_log_params, \
         mock.patch('mlflow.sklearn.log_model') as mock_log_model, \
         mock.patch('mlflow.end_run') as mock_end_run:
        
        yield {
            'start_run': mock_start_run,
            'log_metrics': mock_log_metrics,
            'log_params': mock_log_params,
            'log_model': mock_log_model,
            'end_run': mock_end_run
        }

@pytest.fixture(scope="function")
def mock_model():
    """Mock d'un modèle ML pour les tests"""
    import unittest.mock as mock
    
    class MockModel:
        def predict(self, X):
            # Simuler des prédictions (majoritairement 0, quelques 1)
            return [0] * (len(X) - 1) + [1] if len(X) > 1 else [0]
        
        def predict_proba(self, X):
            # Simuler des probabilités
            proba = []
            for i in range(len(X)):
                if i == len(X) - 1:  # Dernière transaction = fraude
                    proba.append([0.2, 0.8])
                else:
                    proba.append([0.95, 0.05])
            return proba
        
        @property
        def feature_importances_(self):
            # Simuler l'importance des features
            return [0.1] * 28 + [0.2]  # 28 features V + Amount
    
    return MockModel()

@pytest.fixture(scope="function")
def setup_test_directories():
    """Configure les répertoires de test"""
    test_dirs = ["data/raw", "data/processed", "models", "logs", "tests"]
    
    for directory in test_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    yield test_dirs
    
    # Nettoyage optionnel (si nécessaire)
    # for directory in test_dirs:
    #     if Path(directory).exists():
    #         shutil.rmtree(directory)

# Markers personnalisés pour les tests
def pytest_configure(config):
    """Configuration des markers de test"""
    config.addinivalue_line(
        "markers", "unit: Tests unitaires rapides"
    )
    config.addinivalue_line(
        "markers", "integration: Tests d'intégration"
    )
    config.addinivalue_line(
        "markers", "performance: Tests de performance"
    )
    config.addinivalue_line(
        "markers", "security: Tests de sécurité"
    )
    config.addinivalue_line(
        "markers", "slow: Tests lents"
    )

# Hooks pour les tests
def pytest_runtest_setup(item):
    """Setup avant chaque test"""
    # Configurer les variables d'environnement pour les tests
    os.environ["ENVIRONMENT"] = "test"
    os.environ["USE_REDIS"] = "false"
    if "JWT_SECRET_KEY" not in os.environ:
        os.environ["JWT_SECRET_KEY"] = "test_secret_key_do_not_use_in_production"

def pytest_collection_modifyitems(config, items):
    """Modifier la collection de tests"""
    # Ajouter des markers automatiquement selon les noms de fichiers
    for item in items:
        if "test_auth" in item.nodeid:
            item.add_marker(pytest.mark.security)
        elif "test_validation" in item.nodeid:
            item.add_marker(pytest.mark.security)
        elif "test_rate_limiter" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        elif "test_api" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

# Fixtures pour les tests de performance
@pytest.fixture(scope="session")
def benchmark_data():
    """Données pour les benchmarks"""
    import pandas as pd
    import numpy as np
    
    # Créer un dataset plus large pour les tests de performance
    np.random.seed(42)
    n_samples = 10000
    
    features = {}
    for i in range(1, 29):
        features[f"V{i}"] = np.random.normal(0, 1, n_samples)
    
    features["Amount"] = np.random.lognormal(0, 1, n_samples)
    features["Class"] = np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
    
    return pd.DataFrame(features)

# Fixtures pour les tests de sécurité
@pytest.fixture(scope="function")
def malicious_payloads():
    """Payloads malicieux pour les tests de sécurité"""
    return {
        "sql_injection": [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "UNION SELECT * FROM users",
            "1' AND 1=1--"
        ],
        "xss": [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "onload=alert('xss')"
        ],
        "command_injection": [
            "; rm -rf /",
            " && cat /etc/passwd",
            " | ls -la",
            " $(whoami)"
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "C:\\windows\\system32\\config\\sam"
        ]
    }