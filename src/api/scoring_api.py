"""
API de scoring pour la détection de fraude en temps réel
Version simple utilisant le meilleur modèle entraîné
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import os

# Imports sécurisés
from .auth import (
    AuthService, get_current_user, require_analyst_or_admin, require_admin,
    UserInDB, Token, UserCreate, UserResponse, log_access, fake_users_db,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from .validation import (
    SecureTransactionRequest, SecureBatchRequest, 
    validate_request_data, ValidationError
)
from .rate_limiter import (
    limiter, rate_limit_check, rate_limit_middleware
)
from ..monitoring.middleware import (
    MonitoringMiddleware, health_check_middleware, 
    metrics_export_middleware, prediction_monitoring
)
from ..monitoring.metrics import metrics_collector, alert_manager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from ..cache.caching_system import get_cache, cache_prediction, cache_model_info, cache_metrics
from ..cache.cache_routes import cache_router
from ..optimization.performance_optimizer import (
    ModelOptimizer, FastPredictor, BatchProcessor, PerformanceMonitor,
    MemoryOptimizer, performance_monitor
)

# Configuration du logging sécurisé
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration de sécurité
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")

# Modèles de données sécurisés (utilisation des modèles validés)
class TransactionRequest(SecureTransactionRequest):
    """Modèle de transaction avec validation de sécurité"""
    pass

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    confidence: str
    model_version: str = "1.0"
    prediction_time: datetime
    request_id: str

# Initialiser FastAPI avec sécurité
app = FastAPI(
    title="API de Détection de Fraude - Sécurisée",
    description="API sécurisée pour scorer des transactions bancaires en temps réel",
    version="2.0.0",
    docs_url="/docs" if ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if ENVIRONMENT == "development" else None,
    openapi_url="/openapi.json" if ENVIRONMENT == "development" else None
)

# Middleware de sécurité
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
)

# Middleware de trusted hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=ALLOWED_HOSTS
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware personnalisé
app.middleware("http")(rate_limit_middleware)
app.add_middleware(MonitoringMiddleware)

# Inclure les routes de cache
app.include_router(cache_router)

# Variables globales pour le modèle
model = None
feature_columns = None
fast_predictor = None
batch_processor = None
performance_monitor = None

def load_model():
    """Charge le meilleur modèle entraîné avec optimisations"""
    global model, feature_columns, fast_predictor, batch_processor, performance_monitor
    
    try:
        model_path = Path("models/random_forest_model.pkl")
        if model_path.exists():
            model = joblib.load(model_path)
            
            # Colonnes features (sans Time et Class)
            feature_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
            
            # Initialiser les optimiseurs
            model_optimizer = ModelOptimizer(str(model_path))
            model_optimizer.optimize_model_memory()
            
            # Créer le prédicteur optimisé
            fast_predictor = FastPredictor(model)
            
            # Créer le processeur de batch
            batch_processor = BatchProcessor(fast_predictor, batch_size=32, max_workers=4)
            
            # Créer le moniteur de performance
            performance_monitor = PerformanceMonitor(window_size=1000)
            
            logger.info("✅ Modèle Random Forest chargé avec optimisations")
        else:
            logger.error("❌ Modèle non trouvé. Exécutez d'abord l'entraînement.")
            
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle: {e}")

# Charger le modèle au démarrage
@app.on_event("startup")
async def startup_event():
    """Initialisation sécurisée de l'application"""
    logger.info("🚀 Démarrage de l'API sécurisée")
    load_model()
    
    # Créer les dossiers nécessaires
    Path("logs").mkdir(exist_ok=True)
    
    logger.info(f"🔒 Environnement: {ENVIRONMENT}")
    logger.info(f"🌐 Hosts autorisés: {ALLOWED_HOSTS}")
    logger.info(f"📋 Documentation: {'Activée' if ENVIRONMENT == 'development' else 'Désactivée'}")

# Routes d'authentification
@app.post("/auth/login", response_model=Token)
@limiter.limit("5/minute")
async def login(request: Request, username: str, password: str):
    """Authentification utilisateur"""
    user = AuthService.authenticate_user(username, password)
    if not user:
        logger.warning(f"Tentative de connexion échouée pour: {username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nom d'utilisateur ou mot de passe incorrect"
        )
    
    access_token = AuthService.create_access_token(data={"sub": user.username, "role": user.role})
    
    logger.info(f"Connexion réussie: {user.username} ({user.role})")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        role=user.role
    )

@app.post("/auth/register", response_model=UserResponse)
@limiter.limit("3/minute")
async def register(request: Request, user_data: UserCreate, current_user: UserInDB = Depends(require_admin)):
    """Création d'un nouvel utilisateur (admin seulement)"""
    try:
        new_user = AuthService.create_user(user_data)
        logger.info(f"Nouvel utilisateur créé: {new_user.username} par {current_user.username}")
        
        return UserResponse(
            username=new_user.username,
            email=new_user.email,
            role=new_user.role,
            is_active=new_user.is_active,
            created_at=new_user.created_at
        )
    except Exception as e:
        logger.error(f"Erreur création utilisateur: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Route de santé avancée
@app.get("/health")
@limiter.limit("100/minute")
async def health_check(request: Request):
    """Vérification de santé complète de l'API"""
    health_status = health_check_middleware.get_health_status()
    health_status.update({
        "model_loaded": model is not None,
        "version": "2.0.0",
        "environment": ENVIRONMENT
    })
    return health_status

# Route d'information sur le modèle avec cache
@app.get("/model/info")
@limiter.limit("30/minute")
@cache_model_info(ttl=7200)  # Cache pendant 2 heures
async def model_info(request: Request, current_user: UserInDB = Depends(get_current_user)):
    """Informations sur le modèle chargé (utilisateur authentifié)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    log_access(current_user, "/model/info", "GET")
    
    # Statistiques de performance si disponibles
    perf_stats = {}
    if performance_monitor:
        perf_stats = performance_monitor.get_average_metrics()
    
    return {
        "model_type": "RandomForestClassifier",
        "model_version": "1.0",
        "features_count": len(feature_columns),
        "features": feature_columns,
        "performance": {
            "roc_auc": 0.929,
            "precision": 0.867,
            "recall": 0.727,
            "f1_score": 0.791
        },
        "optimization": {
            "fast_predictor_enabled": fast_predictor is not None,
            "batch_processor_enabled": batch_processor is not None,
            "performance_monitoring": perf_stats
        },
        "last_updated": datetime.now().isoformat()
    }

# Route principale de scoring
@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("50/minute")
async def predict_fraud(
    request: Request,
    transaction: TransactionRequest,
    current_user: UserInDB = Depends(require_analyst_or_admin)
):
    """
    Prédit si une transaction est frauduleuse (analyst+ seulement)
    
    Args:
        transaction: Données de la transaction
        current_user: Utilisateur authentifié
        
    Returns:
        Prédiction avec probabilité et niveau de risque
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    # Générer un ID de requête unique
    request_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(transaction.dict())) % 10000}"
    
    try:
        # Démarrer le monitoring de performance
        if performance_monitor:
            performance_monitor.start_monitoring("fraud_prediction")
        
        # Validation sécurisée des données
        validated_features = transaction.validate_all_features()
        
        # Convertir en DataFrame
        df = pd.DataFrame([validated_features])
        
        # Vérifier que toutes les colonnes sont présentes
        missing_features = set(feature_columns) - set(df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Features manquantes: {list(missing_features)}"
            )
        
        # Réorganiser les colonnes dans le bon ordre
        df = df[feature_columns]
        
        # Prédiction optimisée avec cache
        if fast_predictor:
            # Utiliser le prédicteur optimisé
            prediction = fast_predictor.predict(df.values)[0]
            probability = fast_predictor.predict_proba(df.values)[0, 1]
        else:
            # Fallback vers le modèle standard
            prediction = model.predict(df)[0]
            probability = model.predict_proba(df)[0, 1]
        
        # Déterminer le niveau de risque
        if probability < 0.3:
            risk_level = "LOW"
            confidence = "HIGH"
        elif probability < 0.7:
            risk_level = "MEDIUM" 
            confidence = "MEDIUM"
        else:
            risk_level = "HIGH"
            confidence = "HIGH"
        
        # Si le modèle prédit une fraude mais avec faible probabilité
        if prediction == 1 and probability < 0.5:
            confidence = "LOW"
        
        # Mesurer le temps de prédiction
        prediction_end_time = datetime.now()
        prediction_duration = (prediction_end_time - datetime.now()).total_seconds()
        
        # Logging sécurisé
        log_access(current_user, "/predict", "POST")
        logger.info(f"Prédiction: {request_id} - Utilisateur: {current_user.username} - Fraude: {bool(prediction)} - Probabilité: {probability:.3f}")
        
        # Terminer le monitoring de performance
        if performance_monitor:
            performance_monitor.end_monitoring(1)
        
        # Enregistrer les métriques de prédiction
        prediction_monitoring.record_prediction_metrics(
            prediction_time=abs(prediction_duration),
            is_fraud=bool(prediction),
            fraud_probability=float(probability),
            risk_level=risk_level,
            user_role=current_user.role
        )
        
        return PredictionResponse(
            is_fraud=bool(prediction),
            fraud_probability=float(probability),
            risk_level=risk_level,
            confidence=confidence,
            model_version="1.0",
            prediction_time=prediction_end_time,
            request_id=request_id
        )
        
    except ValidationError as e:
        logger.warning(f"Validation échouée: {e} - Utilisateur: {current_user.username}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e} - Utilisateur: {current_user.username}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

# Route pour scorer plusieurs transactions
@app.post("/predict/batch")
@limiter.limit("10/minute")
async def predict_batch(
    request: Request,
    batch_request: SecureBatchRequest,
    current_user: UserInDB = Depends(require_analyst_or_admin)
):
    """
    Prédit plusieurs transactions en une fois (analyst+ seulement)
    
    Args:
        batch_request: Requête de batch avec transactions
        current_user: Utilisateur authentifié
        
    Returns:
        Liste des prédictions
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    transactions = batch_request.transactions
    batch_id = batch_request.batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Utiliser le processeur de batch si disponible
        if batch_processor and len(transactions) > 10:
            # Traitement optimisé par batch
            try:
                # Préparer les données
                batch_data = []
                for transaction in transactions:
                    validated_features = transaction.validate_all_features()
                    batch_data.append(validated_features)
                
                # Convertir en DataFrame
                df_batch = pd.DataFrame(batch_data)
                df_batch = df_batch[feature_columns]
                
                # Prédiction par batch
                predictions, probabilities = batch_processor.process_batch(df_batch.values)
                
                # Formatter les résultats
                results = []
                for i, (prediction, probability) in enumerate(zip(predictions, probabilities[:, 1])):
                    # Déterminer le niveau de risque
                    if probability < 0.3:
                        risk_level = "LOW"
                        confidence = "HIGH"
                    elif probability < 0.7:
                        risk_level = "MEDIUM" 
                        confidence = "MEDIUM"
                    else:
                        risk_level = "HIGH"
                        confidence = "HIGH"
                    
                    if prediction == 1 and probability < 0.5:
                        confidence = "LOW"
                    
                    results.append({
                        "transaction_id": i,
                        "prediction": {
                            "is_fraud": bool(prediction),
                            "fraud_probability": float(probability),
                            "risk_level": risk_level,
                            "confidence": confidence
                        }
                    })
                
            except Exception as e:
                logger.error(f"Erreur batch processing: {e}")
                # Fallback vers traitement individuel
                results = []
                for i, transaction in enumerate(transactions):
                    try:
                        validated_features = transaction.validate_all_features()
                        df = pd.DataFrame([validated_features])
                        df = df[feature_columns]
                        
                        prediction = model.predict(df)[0]
                        probability = model.predict_proba(df)[0, 1]
                        
                        # Déterminer le niveau de risque
                        if probability < 0.3:
                            risk_level = "LOW"
                            confidence = "HIGH"
                        elif probability < 0.7:
                            risk_level = "MEDIUM" 
                            confidence = "MEDIUM"
                        else:
                            risk_level = "HIGH"
                            confidence = "HIGH"
                        
                        if prediction == 1 and probability < 0.5:
                            confidence = "LOW"
                        
                        results.append({
                            "transaction_id": i,
                            "prediction": {
                                "is_fraud": bool(prediction),
                                "fraud_probability": float(probability),
                                "risk_level": risk_level,
                                "confidence": confidence
                            }
                        })
                    except Exception as e:
                        results.append({
                            "transaction_id": i,
                            "error": str(e)
                        })
        else:
            # Traitement individuel standard
            results = []
            
            for i, transaction in enumerate(transactions):
                try:
                    # Utiliser la fonction de prédiction interne
                    validated_features = transaction.validate_all_features()
                    df = pd.DataFrame([validated_features])
                    df = df[feature_columns]
                    
                    if fast_predictor:
                        prediction = fast_predictor.predict(df.values)[0]
                        probability = fast_predictor.predict_proba(df.values)[0, 1]
                    else:
                        prediction = model.predict(df)[0]
                        probability = model.predict_proba(df)[0, 1]
                
                # Déterminer le niveau de risque
                if probability < 0.3:
                    risk_level = "LOW"
                    confidence = "HIGH"
                elif probability < 0.7:
                    risk_level = "MEDIUM" 
                    confidence = "MEDIUM"
                else:
                    risk_level = "HIGH"
                    confidence = "HIGH"
                
                if prediction == 1 and probability < 0.5:
                    confidence = "LOW"
                
                results.append({
                    "transaction_id": i,
                    "prediction": {
                        "is_fraud": bool(prediction),
                        "fraud_probability": float(probability),
                        "risk_level": risk_level,
                        "confidence": confidence
                    }
                })
            except Exception as e:
                results.append({
                    "transaction_id": i,
                    "error": str(e)
                })
        
        fraud_count = sum(1 for r in results if "prediction" in r and r["prediction"]["is_fraud"])
        
        # Logging sécurisé
        log_access(current_user, "/predict/batch", "POST")
        logger.info(f"Batch scoring: {batch_id} - Utilisateur: {current_user.username} - Transactions: {len(transactions)} - Fraudes: {fraud_count}")
        
        return {
            "batch_id": batch_id,
            "total_transactions": len(transactions),
            "fraud_detected": fraud_count,
            "fraud_rate": fraud_count / len(transactions),
            "processed_at": datetime.now().isoformat(),
            "processed_by": current_user.username,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du batch scoring: {e} - Utilisateur: {current_user.username}")
        raise HTTPException(status_code=500, detail=f"Erreur de batch scoring: {str(e)}")

# Route pour recharger le modèle
@app.post("/model/reload")
@limiter.limit("5/minute")
async def reload_model(request: Request, current_user: UserInDB = Depends(require_admin)):
    """Recharge le modèle depuis le disque (admin seulement)"""
    try:
        load_model()
        log_access(current_user, "/model/reload", "POST")
        logger.info(f"Modèle rechargé par: {current_user.username}")
        
        return {
            "message": "Modèle rechargé avec succès",
            "model_loaded": model is not None,
            "reloaded_by": current_user.username,
            "reloaded_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Erreur rechargement modèle: {e} - Utilisateur: {current_user.username}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du rechargement: {str(e)}")

# Route d'exemple pour tester
@app.get("/example")
@limiter.limit("20/minute")
async def get_example_transaction(request: Request, current_user: UserInDB = Depends(get_current_user)):
    """Retourne un exemple de transaction pour tester l'API (utilisateur authentifié)"""
    log_access(current_user, "/example", "GET")
    
    return {
        "example_normal": {
            "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
            "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
            "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
            "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
            "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
            "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
            "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
            "Amount": 149.62
        },
        "usage": "POST /predict avec les données ci-dessus",
        "auth_header": "Authorization: Bearer <your_token>",
        "user_role": current_user.role
    }

# Route d'administration
@app.get("/admin/users")
@limiter.limit("10/minute")
async def list_users(request: Request, current_user: UserInDB = Depends(require_admin)):
    """Liste tous les utilisateurs (admin seulement)"""
    log_access(current_user, "/admin/users", "GET")
    
    users = []
    for username, user_data in fake_users_db.items():
        users.append({
            "username": user_data["username"],
            "email": user_data["email"],
            "role": user_data["role"],
            "is_active": user_data["is_active"],
            "created_at": user_data["created_at"].isoformat()
        })
    
    return {
        "users": users,
        "total_users": len(users)
    }

# Route de métriques complètes
@app.get("/metrics")
@limiter.limit("30/minute")
async def get_metrics(request: Request, current_user: UserInDB = Depends(require_analyst_or_admin)):
    """Métriques complètes de l'API (analyst+ seulement)"""
    log_access(current_user, "/metrics", "GET")
    
    # Exporter les métriques si nécessaire
    metrics_export_middleware.maybe_export_metrics()
    
    return {
        "api_version": "2.0.0",
        "model_loaded": model is not None,
        "environment": ENVIRONMENT,
        "metrics": metrics_collector.get_metrics(),
        "endpoint_stats": metrics_collector.get_endpoint_stats(),
        "hourly_stats": metrics_collector.get_hourly_stats(6),  # 6 dernières heures
        "alerts": alert_manager.get_alerts(20)  # 20 dernières alertes
    }

@app.get("/metrics/detailed")
@limiter.limit("10/minute")
async def get_detailed_metrics(request: Request, current_user: UserInDB = Depends(require_admin)):
    """Métriques détaillées (admin seulement)"""
    log_access(current_user, "/metrics/detailed", "GET")
    
    return {
        "metrics": metrics_collector.get_metrics(),
        "endpoint_stats": metrics_collector.get_endpoint_stats(),
        "hourly_stats": metrics_collector.get_hourly_stats(24),
        "daily_stats": metrics_collector.get_daily_stats(7),
        "alerts": alert_manager.get_alerts(100)
    }

# Routes d'administration des métriques
@app.post("/admin/metrics/reset")
@limiter.limit("5/minute")
async def reset_metrics(request: Request, current_user: UserInDB = Depends(require_admin)):
    """Remet à zéro les métriques (admin seulement)"""
    log_access(current_user, "/admin/metrics/reset", "POST")
    
    metrics_collector.reset_metrics()
    logger.info(f"Métriques remises à zéro par: {current_user.username}")
    
    return {
        "message": "Métriques remises à zéro",
        "reset_by": current_user.username,
        "reset_at": datetime.now().isoformat()
    }

@app.post("/admin/metrics/export")
@limiter.limit("5/minute")
async def export_metrics(request: Request, current_user: UserInDB = Depends(require_admin)):
    """Exporte les métriques (admin seulement)"""
    log_access(current_user, "/admin/metrics/export", "POST")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/metrics_export_{timestamp}.json"
    
    metrics_collector.export_metrics(filename)
    logger.info(f"Métriques exportées par: {current_user.username} vers {filename}")
    
    return {
        "message": "Métriques exportées",
        "filename": filename,
        "exported_by": current_user.username,
        "exported_at": datetime.now().isoformat()
    }

# Routes d'optimisation et de performance
@app.get("/admin/performance")
@limiter.limit("10/minute")
async def get_performance_stats(request: Request, current_user: UserInDB = Depends(require_admin)):
    """Statistiques de performance (admin seulement)"""
    log_access(current_user, "/admin/performance", "GET")
    
    # Statistiques de performance
    perf_stats = {}
    if performance_monitor:
        perf_stats = performance_monitor.get_average_metrics()
    
    # Statistiques mémoire
    memory_stats = MemoryOptimizer.get_memory_usage()
    
    return {
        "performance_monitoring": perf_stats,
        "memory_usage": memory_stats,
        "optimizations": {
            "fast_predictor_enabled": fast_predictor is not None,
            "batch_processor_enabled": batch_processor is not None,
            "cache_enabled": get_cache() is not None
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/admin/performance/optimize")
@limiter.limit("5/minute")
async def optimize_memory(request: Request, current_user: UserInDB = Depends(require_admin)):
    """Optimise la mémoire (admin seulement)"""
    log_access(current_user, "/admin/performance/optimize", "POST")
    
    # Nettoyer la mémoire
    memory_before = MemoryOptimizer.get_memory_usage()
    cleanup_result = MemoryOptimizer.cleanup_memory()
    
    # Nettoyer le cache des prédicteurs
    if fast_predictor:
        fast_predictor.clear_cache()
    
    logger.info(f"Optimisation mémoire effectuée par: {current_user.username}")
    
    return {
        "message": "Optimisation mémoire effectuée",
        "memory_before": memory_before,
        "memory_after": cleanup_result,
        "optimized_by": current_user.username,
        "optimized_at": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    # Configuration sécurisée
    ssl_keyfile = os.getenv("SSL_KEYFILE")
    ssl_certfile = os.getenv("SSL_CERTFILE")
    
    if ENVIRONMENT == "production" and ssl_keyfile and ssl_certfile:
        # HTTPS en production
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8001,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            access_log=True
        )
    else:
        # HTTP en développement
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8001,
            access_log=True,
            reload=ENVIRONMENT == "development"
        )
