"""
Module de monitoring et métriques pour l'API de détection de fraude
Collecte et exposition des métriques système et métier
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class APIMetrics:
    """Métriques de l'API"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    requests_per_second: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ModelMetrics:
    """Métriques des modèles ML"""
    total_predictions: int = 0
    fraud_predictions: int = 0
    normal_predictions: int = 0
    avg_prediction_time: float = 0.0
    fraud_rate: float = 0.0
    avg_fraud_probability: float = 0.0
    high_risk_predictions: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SystemMetrics:
    """Métriques système"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    active_connections: int = 0
    load_average: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SecurityMetrics:
    """Métriques de sécurité"""
    failed_authentications: int = 0
    rate_limited_requests: int = 0
    malicious_requests: int = 0
    blocked_ips: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class MetricsCollector:
    """Collecteur de métriques centralisé"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.lock = threading.Lock()
        
        # Métriques
        self.api_metrics = APIMetrics()
        self.model_metrics = ModelMetrics()
        self.system_metrics = SystemMetrics()
        self.security_metrics = SecurityMetrics()
        
        # Historique des temps de réponse
        self.response_times = deque(maxlen=window_size)
        self.prediction_times = deque(maxlen=window_size)
        self.fraud_probabilities = deque(maxlen=window_size)
        
        # Compteurs par endpoint
        self.endpoint_counters = defaultdict(int)
        self.endpoint_errors = defaultdict(int)
        self.endpoint_response_times = defaultdict(lambda: deque(maxlen=100))
        
        # Historique temporel
        self.hourly_stats = defaultdict(lambda: defaultdict(int))
        self.daily_stats = defaultdict(lambda: defaultdict(int))
        
        # Démarrer la collecte système
        self.start_system_monitoring()
    
    def start_system_monitoring(self):
        """Démarre le monitoring système en arrière-plan"""
        def monitor_system():
            while True:
                try:
                    self._collect_system_metrics()
                    time.sleep(30)  # Collecte toutes les 30 secondes
                except Exception as e:
                    logger.error(f"Erreur monitoring système: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=monitor_system, daemon=True)
        thread.start()
    
    def _collect_system_metrics(self):
        """Collecte les métriques système"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Mémoire
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disque
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Connexions réseau
            connections = len(psutil.net_connections())
            
            # Load average (Unix seulement)
            try:
                load_avg = psutil.getloadavg()[0]
            except (AttributeError, OSError):
                load_avg = 0.0
            
            with self.lock:
                self.system_metrics.cpu_usage = cpu_percent
                self.system_metrics.memory_usage = memory_percent
                self.system_metrics.disk_usage = disk_percent
                self.system_metrics.active_connections = connections
                self.system_metrics.load_average = load_avg
        
        except Exception as e:
            logger.error(f"Erreur collecte métriques système: {e}")
    
    def record_request(self, endpoint: str, response_time: float, status_code: int, user_role: str = None):
        """Enregistre une requête API"""
        with self.lock:
            # Métriques générales
            self.api_metrics.total_requests += 1
            if 200 <= status_code < 400:
                self.api_metrics.successful_requests += 1
            else:
                self.api_metrics.failed_requests += 1
            
            # Temps de réponse
            self.response_times.append(response_time)
            self._update_response_time_stats()
            
            # Métriques par endpoint
            self.endpoint_counters[endpoint] += 1
            if status_code >= 400:
                self.endpoint_errors[endpoint] += 1
            
            self.endpoint_response_times[endpoint].append(response_time)
            
            # Statistiques temporelles
            current_hour = datetime.now().strftime('%Y-%m-%d-%H')
            current_day = datetime.now().strftime('%Y-%m-%d')
            
            self.hourly_stats[current_hour]['requests'] += 1
            self.daily_stats[current_day]['requests'] += 1
            
            if user_role:
                self.hourly_stats[current_hour][f'requests_{user_role}'] += 1
                self.daily_stats[current_day][f'requests_{user_role}'] += 1
    
    def record_prediction(self, prediction_time: float, is_fraud: bool, fraud_probability: float, risk_level: str):
        """Enregistre une prédiction"""
        with self.lock:
            # Métriques de prédiction
            self.model_metrics.total_predictions += 1
            if is_fraud:
                self.model_metrics.fraud_predictions += 1
            else:
                self.model_metrics.normal_predictions += 1
            
            if risk_level == "HIGH":
                self.model_metrics.high_risk_predictions += 1
            
            # Temps de prédiction
            self.prediction_times.append(prediction_time)
            self._update_prediction_time_stats()
            
            # Probabilités de fraude
            self.fraud_probabilities.append(fraud_probability)
            self._update_fraud_stats()
            
            # Statistiques temporelles
            current_hour = datetime.now().strftime('%Y-%m-%d-%H')
            current_day = datetime.now().strftime('%Y-%m-%d')
            
            self.hourly_stats[current_hour]['predictions'] += 1
            self.daily_stats[current_day]['predictions'] += 1
            
            if is_fraud:
                self.hourly_stats[current_hour]['fraud_detected'] += 1
                self.daily_stats[current_day]['fraud_detected'] += 1
    
    def record_security_event(self, event_type: str, details: Dict[str, Any] = None):
        """Enregistre un événement de sécurité"""
        with self.lock:
            if event_type == "failed_auth":
                self.security_metrics.failed_authentications += 1
            elif event_type == "rate_limited":
                self.security_metrics.rate_limited_requests += 1
            elif event_type == "malicious_request":
                self.security_metrics.malicious_requests += 1
            elif event_type == "blocked_ip":
                self.security_metrics.blocked_ips += 1
            
            # Log détaillé
            logger.warning(f"Événement de sécurité: {event_type}", extra={
                "event_type": event_type,
                "details": details or {},
                "timestamp": datetime.now().isoformat()
            })
    
    def _update_response_time_stats(self):
        """Met à jour les statistiques de temps de réponse"""
        if self.response_times:
            times = list(self.response_times)
            self.api_metrics.avg_response_time = np.mean(times)
            self.api_metrics.p95_response_time = np.percentile(times, 95)
            self.api_metrics.p99_response_time = np.percentile(times, 99)
            
            # Calcul des requêtes par seconde
            if len(times) > 1:
                window_duration = len(times) * self.api_metrics.avg_response_time
                if window_duration > 0:
                    self.api_metrics.requests_per_second = len(times) / window_duration
    
    def _update_prediction_time_stats(self):
        """Met à jour les statistiques de temps de prédiction"""
        if self.prediction_times:
            self.model_metrics.avg_prediction_time = np.mean(list(self.prediction_times))
    
    def _update_fraud_stats(self):
        """Met à jour les statistiques de fraude"""
        if self.fraud_probabilities:
            self.model_metrics.avg_fraud_probability = np.mean(list(self.fraud_probabilities))
        
        if self.model_metrics.total_predictions > 0:
            self.model_metrics.fraud_rate = self.model_metrics.fraud_predictions / self.model_metrics.total_predictions
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne toutes les métriques"""
        with self.lock:
            return {
                "api": self.api_metrics.to_dict(),
                "model": self.model_metrics.to_dict(),
                "system": self.system_metrics.to_dict(),
                "security": self.security_metrics.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_endpoint_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques par endpoint"""
        with self.lock:
            stats = {}
            for endpoint in self.endpoint_counters:
                response_times = list(self.endpoint_response_times[endpoint])
                stats[endpoint] = {
                    "requests": self.endpoint_counters[endpoint],
                    "errors": self.endpoint_errors[endpoint],
                    "error_rate": self.endpoint_errors[endpoint] / self.endpoint_counters[endpoint],
                    "avg_response_time": np.mean(response_times) if response_times else 0,
                    "p95_response_time": np.percentile(response_times, 95) if response_times else 0
                }
            return stats
    
    def get_hourly_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Retourne les statistiques horaires"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            relevant_stats = {}
            
            for hour_key, stats in self.hourly_stats.items():
                try:
                    hour_time = datetime.strptime(hour_key, '%Y-%m-%d-%H')
                    if hour_time >= cutoff_time:
                        relevant_stats[hour_key] = stats
                except ValueError:
                    continue
            
            return relevant_stats
    
    def get_daily_stats(self, days: int = 7) -> Dict[str, Any]:
        """Retourne les statistiques journalières"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(days=days)
            relevant_stats = {}
            
            for day_key, stats in self.daily_stats.items():
                try:
                    day_time = datetime.strptime(day_key, '%Y-%m-%d')
                    if day_time >= cutoff_time:
                        relevant_stats[day_key] = stats
                except ValueError:
                    continue
            
            return relevant_stats
    
    def export_metrics(self, filepath: str):
        """Exporte les métriques vers un fichier"""
        try:
            metrics_data = {
                "metrics": self.get_metrics(),
                "endpoint_stats": self.get_endpoint_stats(),
                "hourly_stats": self.get_hourly_stats(24),
                "daily_stats": self.get_daily_stats(7),
                "export_time": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.info(f"Métriques exportées vers {filepath}")
            
        except Exception as e:
            logger.error(f"Erreur export métriques: {e}")
    
    def reset_metrics(self):
        """Remet à zéro les métriques"""
        with self.lock:
            self.api_metrics = APIMetrics()
            self.model_metrics = ModelMetrics()
            self.security_metrics = SecurityMetrics()
            
            self.response_times.clear()
            self.prediction_times.clear()
            self.fraud_probabilities.clear()
            
            self.endpoint_counters.clear()
            self.endpoint_errors.clear()
            self.endpoint_response_times.clear()
            
            logger.info("Métriques remises à zéro")

# Instance globale du collecteur
metrics_collector = MetricsCollector()

class AlertManager:
    """Gestionnaire d'alertes"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules = []
        self.alert_history = deque(maxlen=1000)
        self.webhook_urls = []
        self.email_config = None
        
        # Règles d'alertes par défaut
        self._setup_default_alerts()
        
        # Démarrer le monitoring d'alertes
        self.start_alert_monitoring()
    
    def _setup_default_alerts(self):
        """Configure les alertes par défaut"""
        self.add_alert_rule(
            name="high_error_rate",
            condition=lambda m: m["api"]["failed_requests"] / max(m["api"]["total_requests"], 1) > 0.1,
            message="Taux d'erreur élevé détecté",
            severity="warning"
        )
        
        self.add_alert_rule(
            name="high_response_time",
            condition=lambda m: m["api"]["p95_response_time"] > 1.0,
            message="Temps de réponse élevé détecté",
            severity="warning"
        )
        
        self.add_alert_rule(
            name="high_cpu_usage",
            condition=lambda m: m["system"]["cpu_usage"] > 80,
            message="Utilisation CPU élevée",
            severity="critical"
        )
        
        self.add_alert_rule(
            name="high_memory_usage",
            condition=lambda m: m["system"]["memory_usage"] > 85,
            message="Utilisation mémoire élevée",
            severity="critical"
        )
        
        self.add_alert_rule(
            name="high_fraud_rate",
            condition=lambda m: m["model"]["fraud_rate"] > 0.05,
            message="Taux de fraude anormalement élevé",
            severity="warning"
        )
        
        self.add_alert_rule(
            name="security_threats",
            condition=lambda m: m["security"]["malicious_requests"] > 10,
            message="Menaces de sécurité détectées",
            severity="critical"
        )
    
    def add_alert_rule(self, name: str, condition: callable, message: str, severity: str = "warning"):
        """Ajoute une règle d'alerte"""
        self.alert_rules.append({
            "name": name,
            "condition": condition,
            "message": message,
            "severity": severity,
            "last_triggered": None
        })
    
    def start_alert_monitoring(self):
        """Démarre le monitoring d'alertes"""
        def monitor_alerts():
            while True:
                try:
                    self._check_alerts()
                    time.sleep(60)  # Vérification toutes les minutes
                except Exception as e:
                    logger.error(f"Erreur monitoring alertes: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=monitor_alerts, daemon=True)
        thread.start()
    
    def _check_alerts(self):
        """Vérifie les conditions d'alerte"""
        metrics = self.metrics_collector.get_metrics()
        
        for rule in self.alert_rules:
            try:
                if rule["condition"](metrics):
                    # Éviter les alertes répétées (cooldown de 10 minutes)
                    if rule["last_triggered"] is None or \
                       datetime.now() - rule["last_triggered"] > timedelta(minutes=10):
                        
                        self._trigger_alert(rule, metrics)
                        rule["last_triggered"] = datetime.now()
            
            except Exception as e:
                logger.error(f"Erreur évaluation règle {rule['name']}: {e}")
    
    def _trigger_alert(self, rule: Dict[str, Any], metrics: Dict[str, Any]):
        """Déclenche une alerte"""
        alert = {
            "name": rule["name"],
            "message": rule["message"],
            "severity": rule["severity"],
            "timestamp": datetime.now().isoformat(),
            "metrics_snapshot": metrics
        }
        
        self.alert_history.append(alert)
        
        # Log l'alerte
        logger.error(f"ALERTE {rule['severity'].upper()}: {rule['message']}", extra=alert)
        
        # Envoyer notifications
        self._send_notifications(alert)
    
    def _send_notifications(self, alert: Dict[str, Any]):
        """Envoie les notifications d'alerte"""
        try:
            # Webhook notifications
            for webhook_url in self.webhook_urls:
                self._send_webhook(webhook_url, alert)
            
            # Email notifications
            if self.email_config:
                self._send_email(alert)
        
        except Exception as e:
            logger.error(f"Erreur envoi notifications: {e}")
    
    def _send_webhook(self, url: str, alert: Dict[str, Any]):
        """Envoie une notification webhook"""
        try:
            import requests
            response = requests.post(url, json=alert, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Erreur webhook {url}: {e}")
    
    def _send_email(self, alert: Dict[str, Any]):
        """Envoie une notification email"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from']
            msg['To'] = ', '.join(self.email_config['to'])
            msg['Subject'] = f"[ALERTE] {alert['name']}"
            
            body = f"""
            Alerte: {alert['message']}
            Sévérité: {alert['severity']}
            Timestamp: {alert['timestamp']}
            
            Métriques:
            - Requêtes totales: {alert['metrics_snapshot']['api']['total_requests']}
            - Taux d'erreur: {alert['metrics_snapshot']['api']['failed_requests'] / max(alert['metrics_snapshot']['api']['total_requests'], 1):.2%}
            - CPU: {alert['metrics_snapshot']['system']['cpu_usage']:.1f}%
            - Mémoire: {alert['metrics_snapshot']['system']['memory_usage']:.1f}%
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Erreur envoi email: {e}")
    
    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retourne l'historique des alertes"""
        return list(self.alert_history)[-limit:]
    
    def configure_webhook(self, url: str):
        """Configure un webhook pour les notifications"""
        self.webhook_urls.append(url)
    
    def configure_email(self, smtp_server: str, smtp_port: int, username: str, 
                       password: str, from_email: str, to_emails: List[str]):
        """Configure les notifications email"""
        self.email_config = {
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'from': from_email,
            'to': to_emails
        }

# Instance globale du gestionnaire d'alertes
alert_manager = AlertManager(metrics_collector)