"""
Tests pour les modèles ML et l'entraînement
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.training import RobustFraudTrainer
from src.data.Data_ingestion import SimpleDataIngestion

class TestRobustFraudTrainer:
    """Tests pour RobustFraudTrainer"""
    
    @pytest.fixture
    def trainer(self):
        """Fixture pour créer un trainer"""
        return RobustFraudTrainer()
    
    @pytest.fixture
    def sample_data(self):
        """Fixture pour créer des données d'exemple"""
        np.random.seed(42)
        n_samples = 1000
        
        # Créer des features
        data = {}
        for i in range(1, 29):
            data[f"V{i}"] = np.random.normal(0, 1, n_samples)
        
        data["Amount"] = np.random.lognormal(0, 1, n_samples)
        data["Class"] = np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
        data["Time"] = np.arange(n_samples)
        
        return pd.DataFrame(data)
    
    def test_trainer_initialization(self, trainer):
        """Test d'initialisation du trainer"""
        assert trainer is not None
        assert hasattr(trainer, 'mlflow_uri')
        assert hasattr(trainer, 'mlflow_enabled')
    
    def test_prepare_features(self, trainer, sample_data):
        """Test de préparation des features"""
        X, y = trainer.prepare_features(sample_data)
        
        # Vérifier que Time est exclue
        assert 'Time' not in X.columns
        assert 'Class' not in X.columns
        
        # Vérifier que toutes les features sont présentes
        expected_features = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        assert all(feature in X.columns for feature in expected_features)
        
        # Vérifier les dimensions
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        assert len(X.columns) == 29  # 28 V features + Amount
    
    def test_prepare_features_with_missing_values(self, trainer, sample_data):
        """Test de préparation des features avec valeurs manquantes"""
        # Ajouter des valeurs manquantes
        sample_data_with_nan = sample_data.copy()
        sample_data_with_nan.loc[0, 'V1'] = np.nan
        sample_data_with_nan.loc[1, 'V2'] = np.inf
        sample_data_with_nan.loc[2, 'V3'] = -np.inf
        
        X, y = trainer.prepare_features(sample_data_with_nan)
        
        # Vérifier qu'il n'y a plus de valeurs manquantes ou infinies
        assert not X.isnull().any().any()
        assert not np.isinf(X.values).any()
    
    def test_calculate_metrics(self, trainer):
        """Test de calcul des métriques"""
        # Créer des données de test
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.4, 0.3, 0.9, 0.6, 0.7])
        
        metrics = trainer.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Vérifier que toutes les métriques sont présentes
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        assert all(metric in metrics for metric in expected_metrics)
        
        # Vérifier que les valeurs sont raisonnables
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_calculate_metrics_without_proba(self, trainer):
        """Test de calcul des métriques sans probabilités"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0])
        
        metrics = trainer.calculate_metrics(y_true, y_pred)
        
        # Vérifier que ROC-AUC n'est pas présent
        assert 'roc_auc' not in metrics or metrics['roc_auc'] == 0.5
        
        # Vérifier les autres métriques
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
    
    def test_calculate_metrics_edge_cases(self, trainer):
        """Test de calcul des métriques avec cas limites"""
        # Cas où il n'y a que des vrais négatifs
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        
        metrics = trainer.calculate_metrics(y_true, y_pred)
        
        # Precision et recall devraient être 0 (pas de vrais positifs)
        assert metrics['precision'] == 0
        assert metrics['recall'] == 0
        assert metrics['f1'] == 0
        assert metrics['accuracy'] == 1
    
    def test_safe_mlflow_log(self, trainer):
        """Test de logging MLflow sécurisé"""
        # Simuler une fonction qui lève une exception
        def failing_function(*args, **kwargs):
            raise Exception("MLflow error")
        
        # Vérifier que l'exception est capturée
        result = trainer.safe_mlflow_log(failing_function, "test")
        assert result is None  # Ne devrait pas lever d'exception
    
    @pytest.mark.slow
    def test_train_random_forest_simple(self, trainer, sample_data):
        """Test d'entraînement d'un Random Forest simple"""
        # Préparer les données
        X, y = trainer.prepare_features(sample_data)
        
        # Diviser en train/val
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Entraîner le modèle
        model, metrics, importance = trainer.train_random_forest_simple(
            X_train, y_train, X_val, y_val
        )
        
        # Vérifier que le modèle est entraîné
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        
        # Vérifier les métriques
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        
        # Vérifier l'importance des features
        assert isinstance(importance, pd.DataFrame)
        assert len(importance) == 29  # 28 V features + Amount
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
    
    def test_model_file_saving(self, trainer, sample_data, tmp_path):
        """Test de sauvegarde des modèles"""
        # Changer le répertoire de travail temporairement
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Préparer les données
            X, y = trainer.prepare_features(sample_data)
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Entraîner le modèle
            model, metrics, importance = trainer.train_random_forest_simple(
                X_train, y_train, X_val, y_val
            )
            
            # Vérifier que le fichier modèle est créé
            model_path = Path("models/random_forest_model.pkl")
            assert model_path.exists()
            
            # Vérifier qu'on peut charger le modèle
            import joblib
            loaded_model = joblib.load(model_path)
            assert loaded_model is not None
            
            # Vérifier que le modèle fonctionne
            predictions = loaded_model.predict(X_val)
            assert len(predictions) == len(y_val)
            
        finally:
            os.chdir(original_cwd)

class TestSimpleDataIngestion:
    """Tests pour SimpleDataIngestion"""
    
    @pytest.fixture
    def ingestion(self):
        """Fixture pour créer une instance d'ingestion"""
        return SimpleDataIngestion()
    
    @pytest.fixture
    def sample_csv_data(self, tmp_path):
        """Fixture pour créer un fichier CSV de test"""
        np.random.seed(42)
        n_samples = 1000
        
        # Créer des données
        data = {}
        for i in range(1, 29):
            data[f"V{i}"] = np.random.normal(0, 1, n_samples)
        
        data["Amount"] = np.random.lognormal(0, 1, n_samples)
        data["Class"] = np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
        data["Time"] = np.arange(n_samples)
        
        df = pd.DataFrame(data)
        
        # Créer la structure de répertoires
        data_dir = tmp_path / "data" / "raw"
        data_dir.mkdir(parents=True)
        
        # Sauvegarder le fichier
        csv_path = data_dir / "creditcard.csv"
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    def test_ingestion_initialization(self, ingestion):
        """Test d'initialisation de l'ingestion"""
        assert ingestion is not None
        assert hasattr(ingestion, 'mlflow_uri')
        assert hasattr(ingestion, 'experiment_name')
    
    def test_check_data_file_exists(self, ingestion, sample_csv_data):
        """Test de vérification d'existence du fichier"""
        result = ingestion.check_data_file(str(sample_csv_data))
        assert result is True
    
    def test_check_data_file_not_exists(self, ingestion):
        """Test de vérification d'un fichier inexistant"""
        result = ingestion.check_data_file("nonexistent/file.csv")
        assert result is False
    
    def test_load_and_validate_data(self, ingestion, sample_csv_data):
        """Test de chargement et validation des données"""
        df = ingestion.load_and_validate_data(str(sample_csv_data))
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Vérifier que toutes les colonnes attendues sont présentes
        expected_columns = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class", "Time"]
        assert all(col in df.columns for col in expected_columns)
    
    def test_load_and_validate_data_missing_class(self, ingestion, tmp_path):
        """Test avec données sans colonne Class"""
        # Créer des données sans colonne Class
        data = {f"V{i}": np.random.normal(0, 1, 100) for i in range(1, 29)}
        data["Amount"] = np.random.lognormal(0, 1, 100)
        df = pd.DataFrame(data)
        
        # Créer la structure de répertoires
        data_dir = tmp_path / "data" / "raw"
        data_dir.mkdir(parents=True)
        
        # Sauvegarder le fichier
        csv_path = data_dir / "test.csv"
        df.to_csv(csv_path, index=False)
        
        # Tenter de charger
        result = ingestion.load_and_validate_data(str(csv_path))
        assert result is None  # Devrait échouer
    
    def test_split_data(self, ingestion, sample_csv_data):
        """Test de division des données"""
        df = ingestion.load_and_validate_data(str(sample_csv_data))
        train_df, val_df, test_df = ingestion.split_data(df)
        
        # Vérifier les dimensions
        total_samples = len(df)
        assert len(train_df) + len(val_df) + len(test_df) == total_samples
        
        # Vérifier les proportions approximatives
        assert len(train_df) / total_samples > 0.5  # Plus de 50% pour train
        assert len(val_df) / total_samples > 0.15   # Environ 20% pour val
        assert len(test_df) / total_samples > 0.15  # Environ 20% pour test
        
        # Vérifier que toutes les colonnes sont présentes
        for df_split in [train_df, val_df, test_df]:
            assert "Class" in df_split.columns
            assert all(f"V{i}" in df_split.columns for i in range(1, 29))
            assert "Amount" in df_split.columns
    
    def test_split_data_stratification(self, ingestion, sample_csv_data):
        """Test de la stratification lors de la division"""
        df = ingestion.load_and_validate_data(str(sample_csv_data))
        train_df, val_df, test_df = ingestion.split_data(df)
        
        # Calculer les taux de fraude
        original_fraud_rate = df["Class"].mean()
        train_fraud_rate = train_df["Class"].mean()
        val_fraud_rate = val_df["Class"].mean()
        test_fraud_rate = test_df["Class"].mean()
        
        # Vérifier que les taux sont similaires (±5%)
        tolerance = 0.05
        assert abs(train_fraud_rate - original_fraud_rate) < tolerance
        assert abs(val_fraud_rate - original_fraud_rate) < tolerance
        assert abs(test_fraud_rate - original_fraud_rate) < tolerance
    
    def test_save_processed_data(self, ingestion, sample_csv_data, tmp_path):
        """Test de sauvegarde des données traitées"""
        # Changer le répertoire de travail temporairement
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Charger et diviser les données
            df = ingestion.load_and_validate_data(str(sample_csv_data))
            train_df, val_df, test_df = ingestion.split_data(df)
            
            # Sauvegarder
            ingestion.save_processed_data(train_df, val_df, test_df)
            
            # Vérifier que les fichiers sont créés
            processed_dir = Path("data/processed")
            assert processed_dir.exists()
            assert (processed_dir / "train.csv").exists()
            assert (processed_dir / "validation.csv").exists()
            assert (processed_dir / "test.csv").exists()
            
            # Vérifier que les fichiers peuvent être chargés
            loaded_train = pd.read_csv(processed_dir / "train.csv")
            loaded_val = pd.read_csv(processed_dir / "validation.csv")
            loaded_test = pd.read_csv(processed_dir / "test.csv")
            
            assert len(loaded_train) > 0
            assert len(loaded_val) > 0
            assert len(loaded_test) > 0
            
        finally:
            os.chdir(original_cwd)

class TestModelPerformance:
    """Tests de performance des modèles"""
    
    @pytest.fixture
    def large_dataset(self):
        """Dataset plus large pour les tests de performance"""
        np.random.seed(42)
        n_samples = 10000
        
        data = {}
        for i in range(1, 29):
            data[f"V{i}"] = np.random.normal(0, 1, n_samples)
        
        data["Amount"] = np.random.lognormal(0, 1, n_samples)
        data["Class"] = np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
        data["Time"] = np.arange(n_samples)
        
        return pd.DataFrame(data)
    
    @pytest.mark.performance
    def test_model_training_time(self, large_dataset):
        """Test du temps d'entraînement des modèles"""
        import time
        
        trainer = RobustFraudTrainer()
        X, y = trainer.prepare_features(large_dataset)
        
        # Diviser en train/val
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Mesurer le temps d'entraînement
        start_time = time.time()
        model, metrics, importance = trainer.train_random_forest_simple(
            X_train, y_train, X_val, y_val
        )
        end_time = time.time()
        
        training_time = end_time - start_time
        
        # L'entraînement devrait prendre moins de 60 secondes
        assert training_time < 60
        
        # Vérifier que le modèle fonctionne
        assert model is not None
        assert metrics is not None
    
    @pytest.mark.performance
    def test_model_prediction_time(self, large_dataset):
        """Test du temps de prédiction des modèles"""
        import time
        
        trainer = RobustFraudTrainer()
        X, y = trainer.prepare_features(large_dataset)
        
        # Diviser en train/val
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Entraîner le modèle
        model, metrics, importance = trainer.train_random_forest_simple(
            X_train, y_train, X_val, y_val
        )
        
        # Mesurer le temps de prédiction
        start_time = time.time()
        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)
        end_time = time.time()
        
        prediction_time = end_time - start_time
        
        # La prédiction devrait être très rapide
        assert prediction_time < 1  # Moins d'1 seconde
        
        # Vérifier les résultats
        assert len(predictions) == len(y_val)
        assert len(probabilities) == len(y_val)
    
    @pytest.mark.performance
    def test_memory_usage(self, large_dataset):
        """Test d'utilisation mémoire"""
        import psutil
        import os
        
        # Mesurer la mémoire avant
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        trainer = RobustFraudTrainer()
        X, y = trainer.prepare_features(large_dataset)
        
        # Diviser en train/val
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Entraîner le modèle
        model, metrics, importance = trainer.train_random_forest_simple(
            X_train, y_train, X_val, y_val
        )
        
        # Mesurer la mémoire après
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # L'augmentation de mémoire devrait être raisonnable (< 500MB)
        assert memory_increase < 500
        
        # Nettoyer
        del model, X, y, X_train, X_val, y_train, y_val