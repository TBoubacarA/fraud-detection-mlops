"""
Système de cache avancé pour l'API de détection de fraude
Support Redis et cache mémoire avec fallback
"""

import json
import hashlib
import time
import pickle
from typing import Any, Optional, Dict, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from functools import wraps
import threading
from collections import OrderedDict
import os

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Entrée de cache"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: List[str] = None

class InMemoryCache:
    """Cache en mémoire avec LRU et TTL"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Vérifie si une entrée est expirée"""
        if entry.expires_at is None:
            return False
        return datetime.now() > entry.expires_at
    
    def _evict_expired(self):
        """Évince les entrées expirées"""
        with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if self._is_expired(entry)
            ]
            for key in expired_keys:
                del self.cache[key]
                self._stats['evictions'] += 1
    
    def _evict_lru(self):
        """Évince les entrées LRU si nécessaire"""
        with self.lock:
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self._stats['evictions'] += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Vérifier l'expiration
                if self._is_expired(entry):
                    del self.cache[key]
                    self._stats['misses'] += 1
                    return None
                
                # Mettre à jour les statistiques d'accès
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                
                # Déplacer vers la fin (LRU)
                self.cache.move_to_end(key)
                
                self._stats['hits'] += 1
                return entry.value
            else:
                self._stats['misses'] += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: List[str] = None):
        """Stocke une valeur dans le cache"""
        with self.lock:
            # Nettoyer les entrées expirées
            self._evict_expired()
            
            # Éviction LRU si nécessaire
            self._evict_lru()
            
            # Calculer l'expiration
            expires_at = None
            if ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            elif self.default_ttl > 0:
                expires_at = datetime.now() + timedelta(seconds=self.default_ttl)
            
            # Créer l'entrée
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                tags=tags or []
            )
            
            self.cache[key] = entry
            self._stats['sets'] += 1
    
    def delete(self, key: str) -> bool:
        """Supprime une entrée du cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self._stats['deletes'] += 1
                return True
            return False
    
    def clear(self):
        """Vide le cache"""
        with self.lock:
            self.cache.clear()
            self._stats = {k: 0 for k in self._stats}
    
    def keys(self) -> List[str]:
        """Retourne les clés du cache"""
        with self.lock:
            return list(self.cache.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache"""
        with self.lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                **self._stats,
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'memory_usage': sum(len(pickle.dumps(entry.value)) for entry in self.cache.values())
            }
    
    def invalidate_by_tags(self, tags: List[str]):
        """Invalide les entrées par tags"""
        with self.lock:
            keys_to_delete = []
            for key, entry in self.cache.items():
                if entry.tags and any(tag in entry.tags for tag in tags):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self.cache[key]
                self._stats['deletes'] += 1

class RedisCache:
    """Cache Redis avec sérialisation JSON/Pickle"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 default_ttl: int = 3600, key_prefix: str = "fraud_api"):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.client = None
        self.is_connected = False
        
        self._connect()
    
    def _connect(self):
        """Connecte à Redis"""
        try:
            self.client = redis.from_url(self.redis_url, decode_responses=False)
            self.client.ping()
            self.is_connected = True
            logger.info("✅ Redis connecté")
        except Exception as e:
            logger.error(f"❌ Erreur connexion Redis: {e}")
            self.is_connected = False
    
    def _get_key(self, key: str) -> str:
        """Génère la clé Redis avec préfixe"""
        return f"{self.key_prefix}:{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Sérialise une valeur"""
        try:
            # Essayer JSON d'abord (plus rapide)
            return json.dumps(value).encode('utf-8')
        except (TypeError, ValueError):
            # Fallback vers pickle
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Désérialise une valeur"""
        try:
            # Essayer JSON d'abord
            return json.loads(data.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            # Fallback vers pickle
            return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache"""
        if not self.is_connected:
            return None
        
        try:
            redis_key = self._get_key(key)
            data = self.client.get(redis_key)
            
            if data is None:
                return None
            
            return self._deserialize(data)
        
        except Exception as e:
            logger.error(f"Erreur Redis get: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: List[str] = None):
        """Stocke une valeur dans le cache"""
        if not self.is_connected:
            return
        
        try:
            redis_key = self._get_key(key)
            serialized_value = self._serialize(value)
            
            # TTL
            expiry = ttl if ttl is not None else self.default_ttl
            
            self.client.setex(redis_key, expiry, serialized_value)
            
            # Gérer les tags
            if tags:
                for tag in tags:
                    tag_key = f"{self.key_prefix}:tag:{tag}"
                    self.client.sadd(tag_key, key)
                    self.client.expire(tag_key, expiry)
        
        except Exception as e:
            logger.error(f"Erreur Redis set: {e}")
    
    def delete(self, key: str) -> bool:
        """Supprime une entrée du cache"""
        if not self.is_connected:
            return False
        
        try:
            redis_key = self._get_key(key)
            result = self.client.delete(redis_key)
            return result > 0
        
        except Exception as e:
            logger.error(f"Erreur Redis delete: {e}")
            return False
    
    def clear(self):
        """Vide le cache (par préfixe)"""
        if not self.is_connected:
            return
        
        try:
            pattern = f"{self.key_prefix}:*"
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
        
        except Exception as e:
            logger.error(f"Erreur Redis clear: {e}")
    
    def keys(self) -> List[str]:
        """Retourne les clés du cache"""
        if not self.is_connected:
            return []
        
        try:
            pattern = f"{self.key_prefix}:*"
            redis_keys = self.client.keys(pattern)
            return [key.decode('utf-8').replace(f"{self.key_prefix}:", "") for key in redis_keys]
        
        except Exception as e:
            logger.error(f"Erreur Redis keys: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques Redis"""
        if not self.is_connected:
            return {}
        
        try:
            info = self.client.info()
            return {
                'connected': True,
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'total_connections_received': info.get('total_connections_received', 0),
                'total_commands_processed': info.get('total_commands_processed', 0)
            }
        
        except Exception as e:
            logger.error(f"Erreur Redis stats: {e}")
            return {'connected': False, 'error': str(e)}
    
    def invalidate_by_tags(self, tags: List[str]):
        """Invalide les entrées par tags"""
        if not self.is_connected:
            return
        
        try:
            keys_to_delete = []
            
            for tag in tags:
                tag_key = f"{self.key_prefix}:tag:{tag}"
                keys = self.client.smembers(tag_key)
                keys_to_delete.extend([self._get_key(key.decode('utf-8')) for key in keys])
                
                # Supprimer le tag
                self.client.delete(tag_key)
            
            # Supprimer les clés
            if keys_to_delete:
                self.client.delete(*keys_to_delete)
        
        except Exception as e:
            logger.error(f"Erreur Redis invalidate_by_tags: {e}")

class HybridCache:
    """Cache hybride avec Redis et fallback mémoire"""
    
    def __init__(self, redis_url: str = None, memory_cache_size: int = 1000, 
                 default_ttl: int = 3600):
        self.default_ttl = default_ttl
        
        # Cache mémoire (toujours disponible)
        self.memory_cache = InMemoryCache(memory_cache_size, default_ttl)
        
        # Cache Redis (optionnel)
        self.redis_cache = None
        if redis_url and REDIS_AVAILABLE:
            self.redis_cache = RedisCache(redis_url, default_ttl)
    
    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur (Redis puis mémoire)"""
        # Essayer Redis d'abord
        if self.redis_cache and self.redis_cache.is_connected:
            value = self.redis_cache.get(key)
            if value is not None:
                # Mettre en cache mémoire pour accès rapide
                self.memory_cache.set(key, value, ttl=300)  # 5 min en mémoire
                return value
        
        # Fallback vers cache mémoire
        return self.memory_cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: List[str] = None):
        """Stocke une valeur dans les deux caches"""
        # Cache mémoire
        self.memory_cache.set(key, value, ttl, tags)
        
        # Cache Redis
        if self.redis_cache and self.redis_cache.is_connected:
            self.redis_cache.set(key, value, ttl, tags)
    
    def delete(self, key: str) -> bool:
        """Supprime une entrée des deux caches"""
        memory_deleted = self.memory_cache.delete(key)
        redis_deleted = False
        
        if self.redis_cache and self.redis_cache.is_connected:
            redis_deleted = self.redis_cache.delete(key)
        
        return memory_deleted or redis_deleted
    
    def clear(self):
        """Vide les deux caches"""
        self.memory_cache.clear()
        if self.redis_cache and self.redis_cache.is_connected:
            self.redis_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques des deux caches"""
        stats = {
            'memory_cache': self.memory_cache.get_stats(),
            'redis_cache': None
        }
        
        if self.redis_cache:
            stats['redis_cache'] = self.redis_cache.get_stats()
        
        return stats
    
    def invalidate_by_tags(self, tags: List[str]):
        """Invalide les entrées par tags dans les deux caches"""
        self.memory_cache.invalidate_by_tags(tags)
        if self.redis_cache and self.redis_cache.is_connected:
            self.redis_cache.invalidate_by_tags(tags)

# Instance globale du cache
_cache_instance = None

def get_cache() -> HybridCache:
    """Retourne l'instance globale du cache"""
    global _cache_instance
    if _cache_instance is None:
        redis_url = os.getenv("REDIS_URL")
        _cache_instance = HybridCache(redis_url)
    return _cache_instance

def generate_cache_key(*args, **kwargs) -> str:
    """Génère une clé de cache à partir des arguments"""
    # Créer une chaîne unique
    key_parts = []
    
    # Ajouter les arguments positionnels
    for arg in args:
        if hasattr(arg, '__dict__'):
            # Objet complexe
            key_parts.append(str(hash(str(sorted(arg.__dict__.items())))))
        else:
            key_parts.append(str(arg))
    
    # Ajouter les arguments nommés
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
    
    # Créer le hash
    key_string = ":".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

def cache_result(ttl: int = 3600, tags: List[str] = None, key_prefix: str = ""):
    """Décorateur pour mettre en cache les résultats de fonction"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Générer la clé de cache
            cache_key = f"{key_prefix}:{func.__name__}:{generate_cache_key(*args, **kwargs)}"
            
            # Essayer de récupérer depuis le cache
            cache = get_cache()
            cached_result = cache.get(cache_key)
            
            if cached_result is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_result
            
            # Calculer le résultat
            logger.debug(f"Cache miss: {cache_key}")
            result = func(*args, **kwargs)
            
            # Mettre en cache
            cache.set(cache_key, result, ttl, tags)
            
            return result
        
        return wrapper
    return decorator

def invalidate_cache_by_tags(tags: List[str]):
    """Invalide le cache par tags"""
    cache = get_cache()
    cache.invalidate_by_tags(tags)

def clear_cache():
    """Vide tout le cache"""
    cache = get_cache()
    cache.clear()

# Décorateurs spécialisés
def cache_prediction(ttl: int = 1800):  # 30 minutes
    """Décorateur pour mettre en cache les prédictions"""
    return cache_result(ttl=ttl, tags=["predictions"], key_prefix="pred")

def cache_model_info(ttl: int = 7200):  # 2 heures
    """Décorateur pour mettre en cache les informations de modèle"""
    return cache_result(ttl=ttl, tags=["model_info"], key_prefix="model")

def cache_metrics(ttl: int = 300):  # 5 minutes
    """Décorateur pour mettre en cache les métriques"""
    return cache_result(ttl=ttl, tags=["metrics"], key_prefix="metrics")

# Exemple d'utilisation
if __name__ == "__main__":
    # Test du cache
    cache = get_cache()
    
    # Test basique
    cache.set("test_key", {"data": "test_value"}, ttl=60)
    result = cache.get("test_key")
    print(f"Cache test: {result}")
    
    # Test avec tags
    cache.set("tagged_key", "tagged_value", ttl=60, tags=["test_tag"])
    
    # Test décorateur
    @cache_result(ttl=60, tags=["function_cache"])
    def expensive_function(x, y):
        """Fonction coûteuse simulée"""
        time.sleep(0.1)  # Simulation
        return x + y
    
    # Premier appel (cache miss)
    start = time.time()
    result1 = expensive_function(1, 2)
    time1 = time.time() - start
    
    # Deuxième appel (cache hit)
    start = time.time()
    result2 = expensive_function(1, 2)
    time2 = time.time() - start
    
    print(f"Premier appel: {result1} en {time1:.3f}s")
    print(f"Deuxième appel: {result2} en {time2:.3f}s")
    
    # Statistiques
    stats = cache.get_stats()
    print(f"Statistiques cache: {stats}")
    
    # Nettoyage
    cache.clear()
    print("Cache vidé")