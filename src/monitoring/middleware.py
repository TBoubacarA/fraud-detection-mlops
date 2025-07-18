"""
Middleware de monitoring pour FastAPI
Collecte automatique des métriques et monitoring des performances
"""

import time
import logging
from typing import Callable, Dict, Any, Optional
from datetime import datetime
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .metrics import metrics_collector, alert_manager

logger = logging.getLogger(__name__)

class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware de monitoring pour l'API"""
    
    def __init__(self, app: ASGIApp, excluded_paths: Optional[list] = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Traite la requête et collecte les métriques"""
        
        # Vérifier si le path doit être exclu du monitoring
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Démarrer le chronométrage
        start_time = time.time()
        
        # Extraire les informations de la requête
        endpoint = request.url.path
        method = request.method
        user_agent = request.headers.get("user-agent", "")
        user_role = None
        
        # Extraire le rôle utilisateur si disponible
        if hasattr(request.state, 'user_role'):
            user_role = request.state.user_role
        
        # Variables pour stocker les informations de réponse
        response = None
        status_code = 500
        
        try:
            # Traiter la requête
            response = await call_next(request)
            status_code = response.status_code
            
            # Collecter les métriques de sécurité
            self._collect_security_metrics(request, response)
            
        except HTTPException as e:
            status_code = e.status_code
            
            # Enregistrer les événements de sécurité
            if status_code == 401:
                metrics_collector.record_security_event("failed_auth", {
                    "endpoint": endpoint,
                    "user_agent": user_agent,
                    "ip": request.client.host if request.client else "unknown"
                })
            elif status_code == 429:
                metrics_collector.record_security_event("rate_limited", {
                    "endpoint": endpoint,
                    "user_agent": user_agent,
                    "ip": request.client.host if request.client else "unknown"
                })
            
            raise e
            
        except Exception as e:
            logger.error(f"Erreur dans le middleware de monitoring: {e}")
            status_code = 500
            raise e
            
        finally:
            # Calculer le temps de réponse
            response_time = time.time() - start_time
            
            # Enregistrer les métriques
            metrics_collector.record_request(
                endpoint=endpoint,
                response_time=response_time,
                status_code=status_code,
                user_role=user_role
            )
            
            # Ajouter les headers de monitoring
            if response:
                response.headers["X-Response-Time"] = f"{response_time:.3f}s"
                response.headers["X-Request-ID"] = f"req_{int(time.time() * 1000)}"
        
        return response
    
    def _collect_security_metrics(self, request: Request, response: Response):
        """Collecte les métriques de sécurité"""
        try:
            # Analyser les headers suspects
            suspicious_headers = [
                "x-forwarded-for",
                "x-real-ip",
                "x-originating-ip",
                "x-remote-ip",
                "x-client-ip"
            ]
            
            suspicious_user_agents = [
                "bot", "crawler", "spider", "scraper",
                "scanner", "exploit", "hack", "attack"
            ]
            
            user_agent = request.headers.get("user-agent", "").lower()
            
            # Détecter les tentatives de scan
            if any(agent in user_agent for agent in suspicious_user_agents):
                metrics_collector.record_security_event("malicious_request", {
                    "type": "suspicious_user_agent",
                    "user_agent": user_agent,
                    "endpoint": request.url.path,
                    "ip": request.client.host if request.client else "unknown"
                })
            
            # Détecter les tentatives d'injection dans les paramètres
            query_params = str(request.query_params).lower()
            malicious_patterns = [
                "script", "javascript", "vbscript", "onload", "onerror",
                "union", "select", "drop", "insert", "delete", "update",
                "exec", "system", "cmd", "shell", "eval"
            ]
            
            if any(pattern in query_params for pattern in malicious_patterns):
                metrics_collector.record_security_event("malicious_request", {
                    "type": "injection_attempt",
                    "query_params": query_params,
                    "endpoint": request.url.path,
                    "ip": request.client.host if request.client else "unknown"
                })
        
        except Exception as e:
            logger.error(f"Erreur collecte métriques sécurité: {e}")

class PredictionMonitoringMiddleware:
    """Middleware spécialisé pour le monitoring des prédictions"""
    
    @staticmethod
    def record_prediction_metrics(
        prediction_time: float,
        is_fraud: bool,
        fraud_probability: float,
        risk_level: str,
        user_role: str,
        model_version: str = "1.0"
    ):
        """Enregistre les métriques de prédiction"""
        try:
            # Enregistrer dans le collecteur principal
            metrics_collector.record_prediction(
                prediction_time=prediction_time,
                is_fraud=is_fraud,
                fraud_probability=fraud_probability,
                risk_level=risk_level
            )
            
            # Logging détaillé
            logger.info(
                f"Prédiction enregistrée: fraude={is_fraud}, "
                f"probabilité={fraud_probability:.3f}, "
                f"risque={risk_level}, "
                f"temps={prediction_time:.3f}s",
                extra={
                    "event_type": "prediction",
                    "is_fraud": is_fraud,
                    "fraud_probability": fraud_probability,
                    "risk_level": risk_level,
                    "prediction_time": prediction_time,
                    "user_role": user_role,
                    "model_version": model_version,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Alertes spécifiques aux prédictions
            if fraud_probability > 0.9:
                alert_manager._trigger_alert(
                    {
                        "name": "high_fraud_probability",
                        "message": f"Probabilité de fraude très élevée: {fraud_probability:.3f}",
                        "severity": "warning"
                    },
                    metrics_collector.get_metrics()
                )
        
        except Exception as e:
            logger.error(f"Erreur enregistrement métriques prédiction: {e}")

class HealthCheckMiddleware:
    """Middleware pour les vérifications de santé"""
    
    def __init__(self):
        self.start_time = datetime.now()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Retourne le statut de santé du système"""
        try:
            current_metrics = metrics_collector.get_metrics()
            
            # Calculer l'uptime
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            # Évaluer la santé
            health_status = "healthy"
            issues = []
            
            # Vérifier les métriques système
            if current_metrics["system"]["cpu_usage"] > 90:
                health_status = "degraded"
                issues.append("CPU usage critical")
            
            if current_metrics["system"]["memory_usage"] > 90:
                health_status = "degraded"
                issues.append("Memory usage critical")
            
            # Vérifier les métriques API
            if current_metrics["api"]["total_requests"] > 0:
                error_rate = current_metrics["api"]["failed_requests"] / current_metrics["api"]["total_requests"]
                if error_rate > 0.2:
                    health_status = "degraded"
                    issues.append(f"High error rate: {error_rate:.2%}")
            
            # Vérifier les temps de réponse
            if current_metrics["api"]["p95_response_time"] > 2.0:
                health_status = "degraded"
                issues.append("High response time")
            
            return {
                "status": health_status,
                "uptime_seconds": uptime,
                "issues": issues,
                "metrics": current_metrics,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Erreur vérification santé: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

class MetricsExportMiddleware:
    """Middleware pour l'export des métriques"""
    
    def __init__(self, export_interval: int = 300):  # 5 minutes
        self.export_interval = export_interval
        self.last_export = time.time()
    
    def maybe_export_metrics(self):
        """Exporte les métriques si nécessaire"""
        current_time = time.time()
        
        if current_time - self.last_export > self.export_interval:
            try:
                # Créer le nom de fichier avec timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"logs/metrics_{timestamp}.json"
                
                # Exporter les métriques
                metrics_collector.export_metrics(filename)
                
                # Nettoyer les anciens fichiers (garder les 24 derniers)
                self._cleanup_old_exports()
                
                self.last_export = current_time
                
            except Exception as e:
                logger.error(f"Erreur export métriques: {e}")
    
    def _cleanup_old_exports(self):
        """Nettoie les anciens fichiers d'export"""
        try:
            import os
            import glob
            
            # Lister tous les fichiers de métriques
            metrics_files = glob.glob("logs/metrics_*.json")
            
            # Trier par date de modification
            metrics_files.sort(key=os.path.getmtime, reverse=True)
            
            # Supprimer les fichiers excédentaires
            for file_path in metrics_files[24:]:  # Garder les 24 derniers
                os.remove(file_path)
                logger.info(f"Fichier de métriques supprimé: {file_path}")
        
        except Exception as e:
            logger.error(f"Erreur nettoyage exports: {e}")

# Instances globales des middlewares
health_check_middleware = HealthCheckMiddleware()
metrics_export_middleware = MetricsExportMiddleware()
prediction_monitoring = PredictionMonitoringMiddleware()

# Decorator pour monitorer les fonctions
def monitor_function(func_name: str = None):
    """Décorateur pour monitorer les fonctions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func_name or func.__name__
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(f"Fonction {function_name} exécutée en {execution_time:.3f}s")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Erreur dans {function_name} après {execution_time:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator