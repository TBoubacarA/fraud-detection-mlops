"""
Script d'entraînement corrigé pour la détection de fraude
Version robuste avec gestion d'erreurs
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustFraudTrainer:
    """Entraîneur robuste pour la détection de fraude"""
    
    def __init__(self):
        # Configuration MLflow
        self.mlflow_uri = "http://localhost:5001"
        mlflow.set_tracking_uri(self.mlflow_uri)
        
        try:
            mlflow.set_experiment("fraud_detection_pipeline")
            self.mlflow_enabled = True
            logger.info(f"✅ MLflow configuré: {self.mlflow_uri}")
        except Exception as e:
            logger.warning(f"⚠️ MLflow désactivé: {e}")
            self.mlflow_enabled = False
    
    def load_data(self):
        """Charge les données prétraitées"""
        logger.info("📊 Chargement des données...")
        
        try:
            train_df = pd.read_csv("data/processed/train.csv")
            val_df = pd.read_csv("data/processed/validation.csv")
            test_df = pd.read_csv("data/processed/test.csv")
            
            logger.info(f"   • Train: {len(train_df)} lignes")
            logger.info(f"   • Validation: {len(val_df)} lignes") 
            logger.info(f"   • Test: {len(test_df)} lignes")
            
            return train_df, val_df, test_df
            
        except FileNotFoundError as e:
            logger.error(f"❌ Fichiers de données non trouvés: {e}")
            logger.error("💡 Exécutez d'abord l'ingestion des données")
            return None, None, None
    
    def prepare_features(self, df):
        """Prépare les features et la target"""
        # Exclure la colonne Time qui peut causer des problèmes numériques
        feature_cols = [col for col in df.columns if col not in ['Class', 'Time']]
        X = df[feature_cols]
        y = df['Class']
        
        # Nettoyer les valeurs inf et nan
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        return X, y
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calcule les métriques d'évaluation"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                metrics['roc_auc'] = 0.5
        
        return metrics
    
    def safe_mlflow_log(self, func, *args, **kwargs):
        """Wrapper sécurisé pour MLflow"""
        if not self.mlflow_enabled:
            return
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"⚠️ MLflow warning: {e}")
    
    def train_random_forest_simple(self, X_train, y_train, X_val, y_val):
        """Entraîne un Random Forest simple et robuste"""
        
        if self.mlflow_enabled:
            run = mlflow.start_run(run_name=f"random_forest_{datetime.now().strftime('%H%M%S')}")
        
        try:
            logger.info("🧠 Entraînement - Random Forest (Simple)")
            
            # Modèle robuste avec gestion du déséquilibre
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced',  # Gestion automatique du déséquilibre
                n_jobs=-1
            )
            
            # Entraînement
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Métriques
            metrics = self.calculate_metrics(y_val, y_pred, y_pred_proba)
            
            # Log MLflow (sécurisé)
            self.safe_mlflow_log(mlflow.log_params, {
                'algorithm': 'random_forest_simple',
                'n_estimators': 100,
                'max_depth': 15,
                'class_weight': 'balanced'
            })
            
            for metric_name, metric_value in metrics.items():
                self.safe_mlflow_log(mlflow.log_metric, f"val_{metric_name}", metric_value)
            
            # Sauvegarder localement (backup)
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            joblib.dump(model, model_dir / "random_forest_model.pkl")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Log model (sécurisé)
            self.safe_mlflow_log(mlflow.sklearn.log_model, model, "model")
            
            logger.info(f"   ✅ ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"   ✅ Precision: {metrics['precision']:.4f}")
            logger.info(f"   ✅ Recall: {metrics['recall']:.4f}")
            logger.info(f"   ✅ F1-Score: {metrics['f1']:.4f}")
            
            return model, metrics, feature_importance
            
        finally:
            if self.mlflow_enabled:
                mlflow.end_run()
    
    def train_random_forest_with_smote(self, X_train, y_train, X_val, y_val):
        """Entraîne un Random Forest avec SMOTE"""
        
        if self.mlflow_enabled:
            run = mlflow.start_run(run_name=f"random_forest_smote_{datetime.now().strftime('%H%M%S')}")
        
        try:
            logger.info("🧠 Entraînement - Random Forest (avec SMOTE)")
            
            # SMOTE pour équilibrer
            smote = SMOTE(random_state=42, k_neighbors=3)  # Réduire k_neighbors
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            logger.info(f"   • Après SMOTE: {len(y_train_balanced)} échantillons")
            logger.info(f"   • Nouvelle distribution: {y_train_balanced.mean():.3%} fraudes")
            
            # Modèle
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            # Entraînement
            model.fit(X_train_balanced, y_train_balanced)
            
            # Prédictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Métriques
            metrics = self.calculate_metrics(y_val, y_pred, y_pred_proba)
            
            # Log MLflow (sécurisé)
            self.safe_mlflow_log(mlflow.log_params, {
                'algorithm': 'random_forest_smote',
                'n_estimators': 100,
                'max_depth': 12,
                'sampling': 'smote'
            })
            
            for metric_name, metric_value in metrics.items():
                self.safe_mlflow_log(mlflow.log_metric, f"val_{metric_name}", metric_value)
            
            # Sauvegarder localement
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            joblib.dump(model, model_dir / "random_forest_smote_model.pkl")
            
            # Log model (sécurisé)
            self.safe_mlflow_log(mlflow.sklearn.log_model, model, "model")
            
            logger.info(f"   ✅ ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"   ✅ Precision: {metrics['precision']:.4f}")
            logger.info(f"   ✅ Recall: {metrics['recall']:.4f}")
            logger.info(f"   ✅ F1-Score: {metrics['f1']:.4f}")
            
            return model, metrics
            
        finally:
            if self.mlflow_enabled:
                mlflow.end_run()
    
    def train_logistic_regression_robust(self, X_train, y_train, X_val, y_val):
        """Entraîne une régression logistique robuste"""
        
        if self.mlflow_enabled:
            run = mlflow.start_run(run_name=f"logistic_robust_{datetime.now().strftime('%H%M%S')}")
        
        try:
            logger.info("🧠 Entraînement - Régression Logistique (Robuste)")
            
            # Scaler robuste pour éviter les problèmes numériques
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Modèle robuste
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=0.1,  # Régularisation forte
                solver='liblinear',  # Solver robuste
                class_weight='balanced'
            )
            
            # Entraînement
            model.fit(X_train_scaled, y_train)
            
            # Prédictions
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            
            # Métriques
            metrics = self.calculate_metrics(y_val, y_pred, y_pred_proba)
            
            # Log MLflow (sécurisé)
            self.safe_mlflow_log(mlflow.log_params, {
                'algorithm': 'logistic_regression_robust',
                'C': 0.1,
                'solver': 'liblinear',
                'scaling': 'robust'
            })
            
            for metric_name, metric_value in metrics.items():
                self.safe_mlflow_log(mlflow.log_metric, f"val_{metric_name}", metric_value)
            
            # Sauvegarder localement
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            joblib.dump(model, model_dir / "logistic_robust_model.pkl")
            joblib.dump(scaler, model_dir / "robust_scaler.pkl")
            
            # Log model (sécurisé)
            self.safe_mlflow_log(mlflow.sklearn.log_model, model, "model")
            self.safe_mlflow_log(mlflow.sklearn.log_model, scaler, "scaler")
            
            logger.info(f"   ✅ ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"   ✅ Precision: {metrics['precision']:.4f}")
            logger.info(f"   ✅ Recall: {metrics['recall']:.4f}")
            logger.info(f"   ✅ F1-Score: {metrics['f1']:.4f}")
            
            return model, scaler, metrics
            
        finally:
            if self.mlflow_enabled:
                mlflow.end_run()
    
    def run_training_pipeline(self):
        """Exécute l'entraînement complet"""
        logger.info("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT")
        
        # Charger les données
        train_df, val_df, test_df = self.load_data()
        if train_df is None:
            return False
        
        # Préparer les features (sans Time)
        X_train, y_train = self.prepare_features(train_df)
        X_val, y_val = self.prepare_features(val_df)
        X_test, y_test = self.prepare_features(test_df)
        
        logger.info(f"📊 Features utilisées: {len(X_train.columns)}")
        logger.info(f"📊 Déséquilibre des classes:")
        logger.info(f"   • Train - Fraudes: {y_train.mean():.3%}")
        logger.info(f"   • Validation - Fraudes: {y_val.mean():.3%}")
        
        results = {}
        
        try:
            # 1. Random Forest simple (plus robuste)
            model1, metrics1, importance1 = self.train_random_forest_simple(X_train, y_train, X_val, y_val)
            results['random_forest_simple'] = metrics1
            
            # 2. Random Forest avec SMOTE
            model2, metrics2 = self.train_random_forest_with_smote(X_train, y_train, X_val, y_val)
            results['random_forest_smote'] = metrics2
            
            # 3. Régression logistique robuste
            model3, scaler3, metrics3 = self.train_logistic_regression_robust(X_train, y_train, X_val, y_val)
            results['logistic_robust'] = metrics3
            
            # Résumé
            logger.info("\n📊 RÉSUMÉ DES PERFORMANCES:")
            for model_name, metrics in results.items():
                logger.info(f"   {model_name}:")
                logger.info(f"     • ROC-AUC: {metrics['roc_auc']:.4f}")
                logger.info(f"     • Precision: {metrics['precision']:.4f}")
                logger.info(f"     • Recall: {metrics['recall']:.4f}")
                logger.info(f"     • F1-Score: {metrics['f1']:.4f}")
            
            # Meilleur modèle
            best_model = max(results.items(), key=lambda x: x[1]['roc_auc'])
            logger.info(f"\n🏆 MEILLEUR MODÈLE: {best_model[0]} (ROC-AUC: {best_model[1]['roc_auc']:.4f})")
            
            # Sauvegarder le résumé
            results_df = pd.DataFrame(results).T
            results_df.to_csv("models/training_results.csv")
            logger.info("📄 Résultats sauvegardés dans models/training_results.csv")
            
            logger.info("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'entraînement: {e}")
            return False

def main():
    """Fonction principale"""
    
    print("🧠 ENTRAÎNEMENT ROBUSTE - DÉTECTION DE FRAUDE")
    print("=" * 50)
    
    # Vérifier les packages
    try:
        import sklearn
        import mlflow
        from imblearn.over_sampling import SMOTE
        print("✅ Packages ML OK")
    except ImportError as e:
        print(f"❌ Package manquant: {e}")
        print("💡 Installez avec: pip3 install imbalanced-learn")
        return
    
    # Entraîner les modèles
    trainer = RobustFraudTrainer()
    success = trainer.run_training_pipeline()
    
    if success:
        print("\n🎉 ENTRAÎNEMENT RÉUSSI!")
        print("📊 Consultez MLflow: http://localhost:5001")
        print("💾 Modèles sauvegardés dans models/")
        print("📄 Résultats dans models/training_results.csv")
    else:
        print("\n❌ ÉCHEC DE L'ENTRAÎNEMENT")
        print("💡 Vérifiez que l'ingestion des données a été faite")

if __name__ == "__main__":
    main()