"""
Module de limitation de débit (rate limiting) pour l'API
Protection contre les attaques DDoS et l'utilisation excessive
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis
import logging
import os

logger = logging.getLogger(__name__)

# Configuration Redis pour le rate limiting distribué
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
USE_REDIS = os.getenv("USE_REDIS", "false").lower() == "true"

class InMemoryStorage:
    """Stockage en mémoire pour le rate limiting (développement)"""
    
    def __init__(self):
        self.storage: Dict[str, Dict[str, any]] = {}
    
    def get(self, key: str) -> Optional[int]:
        """Récupère le nombre de requêtes pour une clé"""
        if key in self.storage:
            data = self.storage[key]
            if datetime.now() < data['reset_time']:
                return data['count']
            else:
                del self.storage[key]
        return None
    
    def set(self, key: str, count: int, expire_seconds: int):
        """Définit le nombre de requêtes avec expiration"""
        self.storage[key] = {
            'count': count,
            'reset_time': datetime.now() + timedelta(seconds=expire_seconds)
        }
    
    def increment(self, key: str, expire_seconds: int = 3600) -> int:
        """Incrémente le compteur pour une clé"""
        current = self.get(key)
        if current is None:
            self.set(key, 1, expire_seconds)
            return 1
        else:
            new_count = current + 1
            self.set(key, new_count, expire_seconds)
            return new_count

class RedisStorage:
    """Stockage Redis pour le rate limiting (production)"""
    
    def __init__(self, redis_url: str):
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("✅ Redis connecté pour le rate limiting")
        except Exception as e:
            logger.error(f"❌ Erreur Redis: {e}")
            raise
    
    def get(self, key: str) -> Optional[int]:
        """Récupère le nombre de requêtes pour une clé"""
        try:
            value = self.redis_client.get(key)
            return int(value) if value else None
        except Exception as e:
            logger.error(f"Erreur Redis get: {e}")
            return None
    
    def set(self, key: str, count: int, expire_seconds: int):
        """Définit le nombre de requêtes avec expiration"""
        try:
            self.redis_client.setex(key, expire_seconds, count)
        except Exception as e:
            logger.error(f"Erreur Redis set: {e}")
    
    def increment(self, key: str, expire_seconds: int = 3600) -> int:
        """Incrémente le compteur pour une clé"""
        try:
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, expire_seconds)
            results = pipe.execute()
            return results[0]
        except Exception as e:
            logger.error(f"Erreur Redis increment: {e}")
            return 1

class RateLimiter:
    """Gestionnaire de limitation de débit personnalisé"""
    
    def __init__(self):
        if USE_REDIS:
            try:
                self.storage = RedisStorage(REDIS_URL)
                self.backend = "redis"
            except Exception:
                logger.warning("Redis non disponible, utilisation du stockage mémoire")
                self.storage = InMemoryStorage()
                self.backend = "memory"
        else:
            self.storage = InMemoryStorage()
            self.backend = "memory"
        
        logger.info(f"Rate limiter initialisé avec backend: {self.backend}")
    
    def is_allowed(self, identifier: str, max_requests: int, window_seconds: int) -> tuple[bool, dict]:
        """
        Vérifie si une requête est autorisée
        
        Args:
            identifier: Identifiant unique (IP, user_id, etc.)
            max_requests: Nombre maximum de requêtes
            window_seconds: Fenêtre de temps en secondes
            
        Returns:
            (is_allowed, info) où info contient les détails
        """
        key = f"rate_limit:{identifier}:{window_seconds}"
        
        try:
            current_count = self.storage.increment(key, window_seconds)
            
            remaining = max(0, max_requests - current_count)
            reset_time = datetime.now() + timedelta(seconds=window_seconds)
            
            info = {
                "limit": max_requests,
                "remaining": remaining,
                "reset_time": reset_time.isoformat(),
                "current_count": current_count
            }
            
            is_allowed = current_count <= max_requests
            
            if not is_allowed:
                logger.warning(f"Rate limit exceeded for {identifier}: {current_count}/{max_requests}")
            
            return is_allowed, info
            
        except Exception as e:
            logger.error(f"Erreur rate limiting: {e}")
            # En cas d'erreur, on autorise la requête
            return True, {"error": str(e)}

# Instance globale du rate limiter
rate_limiter = RateLimiter()

# Configuration des limites par défaut
DEFAULT_LIMITS = {
    "general": {"requests": 100, "window": 3600},  # 100 req/heure
    "predict": {"requests": 50, "window": 300},    # 50 req/5min
    "batch": {"requests": 10, "window": 300},      # 10 req/5min
    "auth": {"requests": 5, "window": 300},        # 5 req/5min
}

def get_client_identifier(request: Request) -> str:
    """Génère un identifiant unique pour le client"""
    # Priorité : utilisateur authentifié > IP
    user_id = getattr(request.state, 'user_id', None)
    if user_id:
        return f"user:{user_id}"
    
    # Utiliser l'IP comme fallback
    ip = get_remote_address(request)
    return f"ip:{ip}"

def rate_limit_check(limit_type: str = "general"):
    """
    Décorateur pour vérifier les limites de débit
    
    Args:
        limit_type: Type de limite à appliquer
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Récupérer la requête depuis les arguments
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                # Si pas de requête trouvée, passer sans limitation
                return await func(*args, **kwargs)
            
            # Configuration de la limite
            limit_config = DEFAULT_LIMITS.get(limit_type, DEFAULT_LIMITS["general"])
            max_requests = limit_config["requests"]
            window_seconds = limit_config["window"]
            
            # Identifier le client
            client_id = get_client_identifier(request)
            
            # Vérifier la limite
            is_allowed, info = rate_limiter.is_allowed(client_id, max_requests, window_seconds)
            
            if not is_allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "message": "Trop de requêtes",
                        "limit": info["limit"],
                        "remaining": info["remaining"],
                        "reset_time": info["reset_time"]
                    },
                    headers={
                        "X-RateLimit-Limit": str(info["limit"]),
                        "X-RateLimit-Remaining": str(info["remaining"]),
                        "X-RateLimit-Reset": info["reset_time"],
                        "Retry-After": str(window_seconds)
                    }
                )
            
            # Ajouter les headers de rate limiting
            response = await func(*args, **kwargs)
            
            if hasattr(response, 'headers'):
                response.headers["X-RateLimit-Limit"] = str(info["limit"])
                response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
                response.headers["X-RateLimit-Reset"] = info["reset_time"]
            
            return response
        
        return wrapper
    return decorator

# Limiter slowapi pour intégration simple
def get_limiter_key(request: Request) -> str:
    """Fonction pour slowapi limiter"""
    return get_client_identifier(request)

# Instance slowapi pour les routes simples
limiter = Limiter(key_func=get_limiter_key)

# Gestionnaire d'erreur personnalisé
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Gestionnaire d'erreur personnalisé pour les limites dépassées"""
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail={
            "message": "Trop de requêtes",
            "detail": str(exc.detail)
        }
    )

# Middleware pour logging des requêtes
async def rate_limit_middleware(request: Request, call_next):
    """Middleware pour logger les requêtes et ajouter les headers"""
    start_time = datetime.now()
    
    response = await call_next(request)
    
    # Logger les requêtes
    duration = (datetime.now() - start_time).total_seconds()
    client_id = get_client_identifier(request)
    
    logger.info(
        f"Request: {request.method} {request.url.path} - "
        f"Client: {client_id} - Duration: {duration:.3f}s - "
        f"Status: {response.status_code}"
    )
    
    return response