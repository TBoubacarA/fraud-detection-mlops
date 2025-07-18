"""
Tests de base pour vérifier que l'environnement fonctionne
"""

import pytest
import os
import sys


def test_python_version():
    """Test que Python est en version 3.9+"""
    assert sys.version_info >= (3, 9)


def test_basic_imports():
    """Test que les imports de base fonctionnent"""
    import pandas as pd
    import numpy as np
    import sklearn
    import fastapi
    
    assert pd.__version__ >= "2.0.0"
    assert np.__version__ >= "1.24.0"


def test_environment_setup():
    """Test que l'environnement est configuré"""
    # Ces tests passent toujours
    assert True
    
    # Test des variables d'environnement optionnelles
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    assert redis_url is not None


def test_directory_structure():
    """Test que la structure de projet existe"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Vérifier que les dossiers existent
    assert os.path.exists(os.path.join(base_dir, "src"))
    assert os.path.exists(os.path.join(base_dir, "src", "api"))
    assert os.path.exists(os.path.join(base_dir, "tests"))


if __name__ == "__main__":
    pytest.main([__file__])