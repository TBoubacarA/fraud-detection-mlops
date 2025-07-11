#!/usr/bin/env python3
"""
Script pour créer la structure du projet de détection de fraude
"""

import os
from pathlib import Path

def create_project_structure():
    """Création de la structure complète du projet"""
    
    # Structure des dossiers
    folders = [
        "data/raw",
        "data/processed", 
        "data/external",
        "src/data",
        "src/features",
        "src/models", 
        "src/monitoring",
        "src/api",
        "notebooks",
        "tests",
        "configs",
        "docker",
        "scripts",
        "logs",
        "models",
        "reports"
    ]
    
    # Créer les dossiers
    print("🏗️  Création de la structure des dossiers...")
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {folder}/")
    
    # Créer les fichiers __init__.py
    python_packages = [
        "src",
        "src/data", 
        "src/features",
        "src/models",
        "src/monitoring", 
        "src/api",
        "tests"
    ]
    
    print("\n🐍 Création des packages Python...")
    for package in python_packages:
        init_file = Path(package) / "__init__.py"
        init_file.touch()
        print(f"   ✅ {init_file}")
    
    # Créer les fichiers .gitkeep
    gitkeep_folders = [
        "data/raw",
        "data/processed",
        "data/external", 
        "logs",
        "models",
        "reports"
    ]
    
    print("\n📌 Création des fichiers .gitkeep...")
    for folder in gitkeep_folders:
        gitkeep_file = Path(folder) / ".gitkeep"
        gitkeep_file.touch()
        print(f"   ✅ {gitkeep_file}")
    
    # Créer le fichier requirements.txt
    print("\n📦 Création du requirements.txt...")
    requirements_content = """# MLflow et tracking
mlflow==2.8.1

# Data processing
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
scipy==1.11.4

# Machine Learning
xgboost==2.0.3
lightgbm==4.1.0
imbalanced-learn==0.11.0
optuna==3.4.0

# API et déploiement
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0

# Base de données
psycopg2-binary==2.9.9
sqlalchemy==2.0.23

# Monitoring et drift detection
evidently==0.4.11

# Explicabilité
shap==0.44.0

# Visualisation
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0

# Utilities
python-dotenv==1.0.0
pyyaml==6.0.1
joblib==1.3.2
tqdm==4.66.1
loguru==0.7.2

# Tests
pytest==7.4.3
pytest-cov==4.1.0

# Development
black==23.11.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    print("   ✅ requirements.txt")
    
    # Créer le fichier .env.example
    print("\n🔧 Création du fichier .env.example...")
    env_content = """# Configuration de la base de données
POSTGRES_USER=Boubacar
POSTGRES_PASSWORD=Toubacar
POSTGRES_DB=Boubacardb
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Configuration MLflow
MLFLOW_TRACKING_URI=http://localhost:5001
MLFLOW_BACKEND_STORE_URI=postgresql://Boubacar:Toubacar@localhost:5432/Boubacardb

# Configuration pgAdmin
PGADMIN_DEFAULT_EMAIL=tboubacaraliou@gmail.com
PGADMIN_DEFAULT_PASSWORD=Tboubacar1

# Configuration API
API_SECRET_KEY=boubacar-fraud-detection-secure-key-2025-xyz789
API_HOST=0.0.0.0
API_PORT=8000

# Configuration des alertes
ALERT_EMAIL=tboubacaraliou@gmail.com
"""
    
    with open(".env.example", "w") as f:
        f.write(env_content)
    print("   ✅ .env.example")
    
    print("\n🎉 Structure du projet créée avec succès!")
    print("\n📋 Prochaines étapes:")
    print("1. Copier .env.example vers .env et ajuster si nécessaire")
    print("2. Installer les dépendances: pip install -r requirements.txt")
    print("3. Télécharger le dataset de fraude dans data/raw/")
    print("4. Lancer le premier pipeline")

if __name__ == "__main__":
    create_project_structure()