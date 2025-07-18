"""
Routes spécifiques au cache pour l'administration
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List
import logging

from ..api.auth import UserInDB, require_admin, log_access
from .caching_system import get_cache, invalidate_cache_by_tags, clear_cache

logger = logging.getLogger(__name__)

# Router pour les routes de cache
cache_router = APIRouter(prefix="/cache", tags=["cache"])

@cache_router.get("/stats")
async def get_cache_stats(current_user: UserInDB = Depends(require_admin)) -> Dict[str, Any]:
    """Retourne les statistiques du cache (admin seulement)"""
    log_access(current_user, "/cache/stats", "GET")
    
    cache = get_cache()
    stats = cache.get_stats()
    
    return {
        "cache_stats": stats,
        "cache_type": "hybrid",
        "timestamp": "now"
    }

@cache_router.get("/keys")
async def get_cache_keys(current_user: UserInDB = Depends(require_admin)) -> Dict[str, Any]:
    """Retourne les clés du cache (admin seulement)"""
    log_access(current_user, "/cache/keys", "GET")
    
    cache = get_cache()
    keys = cache.keys()
    
    return {
        "keys": keys,
        "count": len(keys)
    }

@cache_router.delete("/clear")
async def clear_all_cache(current_user: UserInDB = Depends(require_admin)) -> Dict[str, str]:
    """Vide tout le cache (admin seulement)"""
    log_access(current_user, "/cache/clear", "DELETE")
    
    clear_cache()
    logger.info(f"Cache vidé par: {current_user.username}")
    
    return {"message": "Cache vidé avec succès"}

@cache_router.delete("/invalidate")
async def invalidate_by_tags(
    tags: List[str],
    current_user: UserInDB = Depends(require_admin)
) -> Dict[str, str]:
    """Invalide le cache par tags (admin seulement)"""
    log_access(current_user, "/cache/invalidate", "DELETE")
    
    invalidate_cache_by_tags(tags)
    logger.info(f"Cache invalidé par tags {tags} par: {current_user.username}")
    
    return {"message": f"Cache invalidé pour les tags: {tags}"}

@cache_router.get("/health")
async def cache_health(current_user: UserInDB = Depends(require_admin)) -> Dict[str, Any]:
    """Vérification de santé du cache (admin seulement)"""
    log_access(current_user, "/cache/health", "GET")
    
    cache = get_cache()
    
    # Test du cache
    test_key = "health_check"
    test_value = "test_data"
    
    try:
        cache.set(test_key, test_value, ttl=10)
        retrieved_value = cache.get(test_key)
        cache.delete(test_key)
        
        is_healthy = retrieved_value == test_value
        
        return {
            "healthy": is_healthy,
            "test_passed": is_healthy,
            "cache_available": True
        }
    
    except Exception as e:
        logger.error(f"Erreur test santé cache: {e}")
        return {
            "healthy": False,
            "test_passed": False,
            "cache_available": False,
            "error": str(e)
        }