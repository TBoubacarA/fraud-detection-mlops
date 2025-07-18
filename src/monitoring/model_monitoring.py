"""
Module de monitoring avancé des modèles ML
Détection de dérive des données et des performances
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from collections import deque
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class DriftMetrics:
    """Métriques de dérive"""
    feature_name: str
    drift_score: float
    p_value: float
    threshold: float
    is_drift: bool
    method: str
    timestamp: datetime

@dataclass
class ModelPerformanceMetrics:
    """Métriques de performance du modèle"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    timestamp: datetime
    sample_size: int

class DataDriftDetector:
    """Détecteur de dérive des données"""
    
    def __init__(self, reference_data: pd.DataFrame, drift_threshold: float = 0.05):
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.feature_stats = self._calculate_reference_stats()
        
    def _calculate_reference_stats(self) -> Dict[str, Dict[str, float]]:
        """Calcule les statistiques de référence"""
        stats = {}
        
        for column in self.reference_data.select_dtypes(include=[np.number]).columns:
            if column != 'Class':  # Exclure la variable cible
                stats[column] = {
                    'mean': self.reference_data[column].mean(),
                    'std': self.reference_data[column].std(),
                    'min': self.reference_data[column].min(),
                    'max': self.reference_data[column].max(),
                    'q25': self.reference_data[column].quantile(0.25),
                    'q50': self.reference_data[column].quantile(0.50),
                    'q75': self.reference_data[column].quantile(0.75)
                }
        
        return stats
    
    def detect_drift_ks(self, current_data: pd.DataFrame) -> List[DriftMetrics]:
        """Détection de dérive avec le test de Kolmogorov-Smirnov"""
        drift_results = []
        
        for column in self.feature_stats.keys():
            if column in current_data.columns:
                # Test KS
                ks_stat, p_value = stats.ks_2samp(
                    self.reference_data[column].dropna(),
                    current_data[column].dropna()
                )
                
                is_drift = p_value < self.drift_threshold
                
                drift_results.append(DriftMetrics(
                    feature_name=column,
                    drift_score=ks_stat,
                    p_value=p_value,
                    threshold=self.drift_threshold,
                    is_drift=is_drift,
                    method='kolmogorov_smirnov',
                    timestamp=datetime.now()
                ))
        
        return drift_results
    
    def detect_drift_psi(self, current_data: pd.DataFrame, bins: int = 10) -> List[DriftMetrics]:
        """Détection de dérive avec l'indice PSI (Population Stability Index)"""
        drift_results = []
        
        for column in self.feature_stats.keys():
            if column in current_data.columns:
                try:
                    # Créer des bins basés sur les données de référence
                    ref_data = self.reference_data[column].dropna()
                    current_data_col = current_data[column].dropna()
                    
                    # Calculer les quantiles pour les bins
                    bin_edges = np.quantile(ref_data, np.linspace(0, 1, bins + 1))
                    bin_edges = np.unique(bin_edges)  # Éviter les doublons
                    
                    if len(bin_edges) > 1:
                        # Calculer les distributions
                        ref_counts, _ = np.histogram(ref_data, bins=bin_edges)
                        current_counts, _ = np.histogram(current_data_col, bins=bin_edges)
                        
                        # Éviter les divisions par zéro
                        ref_counts = ref_counts + 1e-10
                        current_counts = current_counts + 1e-10
                        
                        # Normaliser
                        ref_pct = ref_counts / ref_counts.sum()
                        current_pct = current_counts / current_counts.sum()
                        
                        # Calculer PSI
                        psi = np.sum((current_pct - ref_pct) * np.log(current_pct / ref_pct))
                        
                        # Seuils PSI standards
                        # < 0.1: pas de dérive
                        # 0.1-0.2: dérive modérée
                        # > 0.2: dérive significative
                        is_drift = psi > 0.1
                        
                        drift_results.append(DriftMetrics(
                            feature_name=column,
                            drift_score=psi,
                            p_value=0.0,  # PSI n'a pas de p-value
                            threshold=0.1,
                            is_drift=is_drift,
                            method='psi',
                            timestamp=datetime.now()
                        ))
                        
                except Exception as e:
                    logger.warning(f"Erreur calcul PSI pour {column}: {e}")
        
        return drift_results
    
    def detect_drift_statistical(self, current_data: pd.DataFrame) -> List[DriftMetrics]:
        """Détection de dérive basée sur les statistiques descriptives"""
        drift_results = []
        
        for column, ref_stats in self.feature_stats.items():
            if column in current_data.columns:
                current_col = current_data[column].dropna()
                
                if len(current_col) > 0:
                    current_mean = current_col.mean()
                    current_std = current_col.std()
                    
                    # Calcul de l'écart standardisé
                    mean_diff = abs(current_mean - ref_stats['mean']) / ref_stats['std']
                    std_diff = abs(current_std - ref_stats['std']) / ref_stats['std']
                    
                    # Score de dérive combiné
                    drift_score = max(mean_diff, std_diff)
                    
                    # Seuil de dérive (2 écarts-types)
                    threshold = 2.0
                    is_drift = drift_score > threshold
                    
                    drift_results.append(DriftMetrics(
                        feature_name=column,
                        drift_score=drift_score,
                        p_value=0.0,  # Pas de p-value pour cette méthode
                        threshold=threshold,
                        is_drift=is_drift,
                        method='statistical',
                        timestamp=datetime.now()
                    ))
        
        return drift_results

class ModelPerformanceMonitor:
    """Moniteur de performance du modèle"""
    
    def __init__(self, model_path: str, window_size: int = 1000):
        self.model_path = model_path
        self.window_size = window_size
        self.model = None
        self.predictions_history = deque(maxlen=window_size)
        self.performance_history = deque(maxlen=100)  # Garder 100 évaluations
        self.baseline_performance = None
        
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle ML"""
        try:
            if Path(self.model_path).exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"Modèle chargé: {self.model_path}")
            else:
                logger.warning(f"Modèle non trouvé: {self.model_path}")
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
    
    def record_prediction(self, features: np.ndarray, prediction: int, 
                         probability: float, true_label: Optional[int] = None):
        """Enregistre une prédiction"""
        self.predictions_history.append({
            'timestamp': datetime.now(),
            'features': features,
            'prediction': prediction,
            'probability': probability,
            'true_label': true_label
        })
    
    def evaluate_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> ModelPerformanceMetrics:
        """Évalue la performance du modèle"""
        if self.model is None:
            raise ValueError("Modèle non chargé")
        
        # Prédictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calcul des métriques
        metrics = ModelPerformanceMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1_score=f1_score(y_test, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y_test, y_pred_proba),
            timestamp=datetime.now(),
            sample_size=len(y_test)
        )
        
        # Stocker dans l'historique
        self.performance_history.append(metrics)
        
        # Définir la baseline si c'est la première évaluation
        if self.baseline_performance is None:
            self.baseline_performance = metrics
            logger.info("Performance baseline définie")
        
        return metrics
    
    def detect_performance_drift(self, current_metrics: ModelPerformanceMetrics, 
                                threshold: float = 0.05) -> Dict[str, bool]:
        """Détecte une dérive de performance"""
        if self.baseline_performance is None:
            return {}
        
        drift_detected = {}
        
        # Comparer chaque métrique avec la baseline
        metrics_to_check = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for metric in metrics_to_check:
            baseline_value = getattr(self.baseline_performance, metric)
            current_value = getattr(current_metrics, metric)
            
            # Calculer la dégradation relative
            if baseline_value > 0:
                degradation = (baseline_value - current_value) / baseline_value
                drift_detected[metric] = degradation > threshold
            else:
                drift_detected[metric] = False
        
        return drift_detected
    
    def get_prediction_distribution(self) -> Dict[str, Any]:
        """Analyse la distribution des prédictions récentes"""
        if not self.predictions_history:
            return {}
        
        predictions = [p['prediction'] for p in self.predictions_history]
        probabilities = [p['probability'] for p in self.predictions_history]
        
        return {
            'fraud_rate': np.mean(predictions),
            'avg_probability': np.mean(probabilities),
            'std_probability': np.std(probabilities),
            'total_predictions': len(predictions),
            'timestamp': datetime.now()
        }

class ModelMonitoringSystem:
    """Système de monitoring complet des modèles"""
    
    def __init__(self, model_path: str, reference_data: pd.DataFrame, 
                 monitoring_interval: int = 3600):
        self.model_path = model_path
        self.reference_data = reference_data
        self.monitoring_interval = monitoring_interval
        
        # Composants de monitoring
        self.drift_detector = DataDriftDetector(reference_data)
        self.performance_monitor = ModelPerformanceMonitor(model_path)
        
        # Historique des alertes
        self.drift_alerts = deque(maxlen=1000)
        self.performance_alerts = deque(maxlen=1000)
        
        # État du monitoring
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Callbacks pour les alertes
        self.alert_callbacks = []
    
    def add_alert_callback(self, callback):
        """Ajoute un callback pour les alertes"""
        self.alert_callbacks.append(callback)
    
    def _trigger_alert(self, alert_type: str, message: str, details: Dict[str, Any]):
        """Déclenche une alerte"""
        alert = {
            'type': alert_type,
            'message': message,
            'details': details,
            'timestamp': datetime.now()
        }
        
        if alert_type == 'drift':
            self.drift_alerts.append(alert)
        elif alert_type == 'performance':
            self.performance_alerts.append(alert)
        
        # Appeler les callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Erreur callback alerte: {e}")
        
        logger.warning(f"Alerte {alert_type}: {message}")
    
    def check_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Vérifie la dérive des données"""
        # Différentes méthodes de détection
        ks_results = self.drift_detector.detect_drift_ks(current_data)
        psi_results = self.drift_detector.detect_drift_psi(current_data)
        stat_results = self.drift_detector.detect_drift_statistical(current_data)
        
        # Compiler les résultats
        drift_summary = {
            'timestamp': datetime.now(),
            'total_features': len(ks_results),
            'ks_drifts': sum(1 for r in ks_results if r.is_drift),
            'psi_drifts': sum(1 for r in psi_results if r.is_drift),
            'stat_drifts': sum(1 for r in stat_results if r.is_drift),
            'features_with_drift': []
        }
        
        # Identifier les features avec dérive
        all_results = ks_results + psi_results + stat_results
        drift_features = set()
        
        for result in all_results:
            if result.is_drift:
                drift_features.add(result.feature_name)
        
        drift_summary['features_with_drift'] = list(drift_features)
        
        # Déclencher des alertes si nécessaire
        if len(drift_features) > 0:
            self._trigger_alert(
                'drift',
                f"Dérive détectée sur {len(drift_features)} features",
                {
                    'features': list(drift_features),
                    'ks_drifts': drift_summary['ks_drifts'],
                    'psi_drifts': drift_summary['psi_drifts'],
                    'stat_drifts': drift_summary['stat_drifts']
                }
            )
        
        return drift_summary
    
    def check_model_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Vérifie la performance du modèle"""
        current_metrics = self.performance_monitor.evaluate_performance(X_test, y_test)
        drift_detected = self.performance_monitor.detect_performance_drift(current_metrics)
        
        performance_summary = {
            'timestamp': datetime.now(),
            'metrics': {
                'accuracy': current_metrics.accuracy,
                'precision': current_metrics.precision,
                'recall': current_metrics.recall,
                'f1_score': current_metrics.f1_score,
                'roc_auc': current_metrics.roc_auc
            },
            'sample_size': current_metrics.sample_size,
            'drift_detected': drift_detected,
            'degraded_metrics': [k for k, v in drift_detected.items() if v]
        }
        
        # Déclencher des alertes pour les dégradations
        if any(drift_detected.values()):
            degraded_metrics = [k for k, v in drift_detected.items() if v]
            self._trigger_alert(
                'performance',
                f"Dégradation de performance détectée: {', '.join(degraded_metrics)}",
                {
                    'degraded_metrics': degraded_metrics,
                    'current_metrics': performance_summary['metrics']
                }
            )
        
        return performance_summary
    
    def start_monitoring(self):
        """Démarre le monitoring continu"""
        if self.is_monitoring:
            logger.warning("Monitoring déjà en cours")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Monitoring des modèles démarré")
    
    def stop_monitoring(self):
        """Arrête le monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        logger.info("Monitoring des modèles arrêté")
    
    def _monitoring_loop(self):
        """Boucle de monitoring continue"""
        while self.is_monitoring:
            try:
                # Ici, vous pourriez implémenter la logique pour:
                # - Récupérer les nouvelles données
                # - Vérifier la dérive
                # - Évaluer la performance
                # - Déclencher des alertes
                
                # Pour l'instant, on fait juste une pause
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle de monitoring: {e}")
                time.sleep(60)  # Pause plus longue en cas d'erreur
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Génère un rapport de monitoring"""
        return {
            'timestamp': datetime.now(),
            'monitoring_active': self.is_monitoring,
            'model_path': self.model_path,
            'reference_data_size': len(self.reference_data),
            'drift_alerts_count': len(self.drift_alerts),
            'performance_alerts_count': len(self.performance_alerts),
            'recent_drift_alerts': list(self.drift_alerts)[-10:],
            'recent_performance_alerts': list(self.performance_alerts)[-10:],
            'prediction_distribution': self.performance_monitor.get_prediction_distribution()
        }
    
    def export_monitoring_data(self, filepath: str):
        """Exporte les données de monitoring"""
        try:
            monitoring_data = {
                'report': self.get_monitoring_report(),
                'drift_alerts': list(self.drift_alerts),
                'performance_alerts': list(self.performance_alerts),
                'performance_history': [
                    {
                        'accuracy': p.accuracy,
                        'precision': p.precision,
                        'recall': p.recall,
                        'f1_score': p.f1_score,
                        'roc_auc': p.roc_auc,
                        'timestamp': p.timestamp.isoformat(),
                        'sample_size': p.sample_size
                    }
                    for p in self.performance_monitor.performance_history
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(monitoring_data, f, indent=2, default=str)
            
            logger.info(f"Données de monitoring exportées: {filepath}")
            
        except Exception as e:
            logger.error(f"Erreur export monitoring: {e}")

# Fonction utilitaire pour créer un système de monitoring
def create_monitoring_system(model_path: str, reference_data_path: str) -> ModelMonitoringSystem:
    """Crée un système de monitoring à partir des fichiers"""
    try:
        # Charger les données de référence
        reference_data = pd.read_csv(reference_data_path)
        
        # Créer le système de monitoring
        monitoring_system = ModelMonitoringSystem(model_path, reference_data)
        
        logger.info("Système de monitoring créé avec succès")
        return monitoring_system
        
    except Exception as e:
        logger.error(f"Erreur création système monitoring: {e}")
        raise

# Exemple d'utilisation
if __name__ == "__main__":
    # Créer un système de monitoring
    monitoring = create_monitoring_system(
        model_path="models/random_forest_model.pkl",
        reference_data_path="data/processed/train.csv"
    )
    
    # Démarrer le monitoring
    monitoring.start_monitoring()
    
    # Exemple de vérification de dérive
    # new_data = pd.read_csv("data/processed/test.csv")
    # drift_summary = monitoring.check_data_drift(new_data)
    # print(f"Dérive détectée: {drift_summary}")
    
    # Arrêter le monitoring
    # monitoring.stop_monitoring()