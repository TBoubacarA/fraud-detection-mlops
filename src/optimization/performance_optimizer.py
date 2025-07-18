"""
Module d'optimisation des performances
Optimisations pour la prédiction ML et l'API
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from dataclasses import dataclass
import psutil
import gc
from pathlib import Path
import joblib
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Métriques de performance"""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    timestamp: float

class ModelOptimizer:
    """Optimiseur de modèles ML"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.optimized_model = None
        self.feature_importance = None
        self.feature_selector = None
        
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle"""
        try:
            if Path(self.model_path).exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"Modèle chargé: {self.model_path}")
            else:
                logger.error(f"Modèle non trouvé: {self.model_path}")
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
    
    def optimize_feature_selection(self, X: np.ndarray, y: np.ndarray, 
                                  feature_threshold: float = 0.01) -> Tuple[np.ndarray, List[int]]:
        """Optimise la sélection des features"""
        if self.model is None:
            raise ValueError("Modèle non chargé")
        
        # Récupérer l'importance des features
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            logger.warning("Modèle sans feature_importances_, utilisation de toutes les features")
            return X, list(range(X.shape[1]))
        
        # Sélectionner les features importantes
        important_features = np.where(importances > feature_threshold)[0]
        
        logger.info(f"Sélection de {len(important_features)}/{X.shape[1]} features importantes")
        
        # Stocker les indices des features sélectionnées
        self.feature_selector = important_features
        
        return X[:, important_features], important_features.tolist()
    
    def optimize_model_memory(self):
        """Optimise l'utilisation mémoire du modèle"""
        if self.model is None:
            return
        
        # Pour les modèles sklearn, on peut optimiser certains paramètres
        if hasattr(self.model, 'n_jobs'):
            # Utiliser tous les CPU disponibles
            self.model.n_jobs = -1
        
        # Optimisations spécifiques aux Random Forest
        if hasattr(self.model, 'max_samples'):
            # Limiter les échantillons par arbre pour économiser la mémoire
            if self.model.max_samples is None:
                self.model.max_samples = 0.8
        
        logger.info("Optimisations mémoire appliquées")
    
    def create_fast_predictor(self, X_sample: np.ndarray) -> 'FastPredictor':
        """Crée un prédicteur optimisé pour la vitesse"""
        if self.model is None:
            raise ValueError("Modèle non chargé")
        
        # Optimiser la sélection des features si pas déjà fait
        if self.feature_selector is None:
            _, _ = self.optimize_feature_selection(X_sample, np.zeros(X_sample.shape[0]))
        
        return FastPredictor(self.model, self.feature_selector)
    
    def benchmark_prediction(self, X: np.ndarray, n_iterations: int = 100) -> Dict[str, float]:
        """Benchmark la performance de prédiction"""
        if self.model is None:
            raise ValueError("Modèle non chargé")
        
        # Mesurer les performances
        times = []
        memory_usage = []
        
        for i in range(n_iterations):
            # Mesurer la mémoire avant
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Mesurer le temps
            start_time = time.time()
            predictions = self.model.predict(X)
            end_time = time.time()
            
            # Mesurer la mémoire après
            mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            times.append(end_time - start_time)
            memory_usage.append(mem_after - mem_before)
        
        return {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'avg_memory': np.mean(memory_usage),
            'throughput': len(X) / np.mean(times)  # predictions/sec
        }

class FastPredictor:
    """Prédicteur optimisé pour la vitesse"""
    
    def __init__(self, model: BaseEstimator, feature_selector: Optional[List[int]] = None):
        self.model = model
        self.feature_selector = feature_selector
        self.prediction_cache = {}
        self.cache_size = 1000
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédiction optimisée"""
        # Appliquer la sélection des features
        if self.feature_selector is not None:
            X_selected = X[:, self.feature_selector]
        else:
            X_selected = X
        
        # Vérifier le cache pour les prédictions uniques
        if len(X_selected) == 1:
            cache_key = hash(X_selected.tobytes())
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]
        
        # Prédiction
        predictions = self.model.predict(X_selected)
        
        # Mettre en cache pour les prédictions uniques
        if len(X_selected) == 1 and len(self.prediction_cache) < self.cache_size:
            cache_key = hash(X_selected.tobytes())
            self.prediction_cache[cache_key] = predictions
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Prédiction de probabilités optimisée"""
        # Appliquer la sélection des features
        if self.feature_selector is not None:
            X_selected = X[:, self.feature_selector]
        else:
            X_selected = X
        
        return self.model.predict_proba(X_selected)
    
    def clear_cache(self):
        """Vide le cache de prédictions"""
        self.prediction_cache.clear()

class BatchProcessor:
    """Processeur de requêtes par batch"""
    
    def __init__(self, predictor: FastPredictor, batch_size: int = 32, 
                 max_workers: int = 4):
        self.predictor = predictor
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def process_batch(self, X_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Traite un batch de prédictions"""
        predictions = self.predictor.predict(X_batch)
        probabilities = self.predictor.predict_proba(X_batch)
        return predictions, probabilities
    
    def process_multiple_batches(self, X_list: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Traite plusieurs batches en parallèle"""
        futures = []
        
        for X_batch in X_list:
            future = self.executor.submit(self.process_batch, X_batch)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Erreur traitement batch: {e}")
                results.append((None, None))
        
        return results
    
    def process_large_dataset(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Traite un grand dataset par batches"""
        n_samples = X.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        all_predictions = []
        all_probabilities = []
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            
            X_batch = X[start_idx:end_idx]
            predictions, probabilities = self.process_batch(X_batch)
            
            all_predictions.append(predictions)
            all_probabilities.append(probabilities)
        
        return np.concatenate(all_predictions), np.concatenate(all_probabilities)
    
    def __del__(self):
        """Nettoyage des ressources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

class PerformanceMonitor:
    """Moniteur de performances"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = []
        self.current_metrics = None
    
    def start_monitoring(self, operation_name: str):
        """Démarre le monitoring d'une opération"""
        self.current_metrics = {
            'operation': operation_name,
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / 1024 / 1024,
            'start_cpu': psutil.cpu_percent()
        }
    
    def end_monitoring(self, throughput_items: int = 1):
        """Termine le monitoring et calcule les métriques"""
        if self.current_metrics is None:
            return
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        execution_time = end_time - self.current_metrics['start_time']
        memory_usage = end_memory - self.current_metrics['start_memory']
        cpu_usage = end_cpu - self.current_metrics['start_cpu']
        throughput = throughput_items / execution_time if execution_time > 0 else 0
        
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            throughput=throughput,
            timestamp=end_time
        )
        
        self.metrics_history.append(metrics)
        
        # Garder seulement les dernières métriques
        if len(self.metrics_history) > self.window_size:
            self.metrics_history = self.metrics_history[-self.window_size:]
        
        self.current_metrics = None
        return metrics
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Retourne les métriques moyennes"""
        if not self.metrics_history:
            return {}
        
        return {
            'avg_execution_time': np.mean([m.execution_time for m in self.metrics_history]),
            'avg_memory_usage': np.mean([m.memory_usage for m in self.metrics_history]),
            'avg_cpu_usage': np.mean([m.cpu_usage for m in self.metrics_history]),
            'avg_throughput': np.mean([m.throughput for m in self.metrics_history]),
            'total_operations': len(self.metrics_history)
        }

def performance_monitor(operation_name: str = None):
    """Décorateur pour monitorer les performances"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = PerformanceMonitor()
            name = operation_name or func.__name__
            
            monitor.start_monitoring(name)
            try:
                result = func(*args, **kwargs)
                # Essayer de déterminer le nombre d'items traités
                throughput_items = 1
                if hasattr(result, '__len__'):
                    throughput_items = len(result)
                elif isinstance(result, (list, tuple)) and len(result) > 0:
                    first_item = result[0]
                    if hasattr(first_item, '__len__'):
                        throughput_items = len(first_item)
                
                metrics = monitor.end_monitoring(throughput_items)
                logger.info(f"Performance {name}: {metrics.execution_time:.3f}s, "
                           f"throughput: {metrics.throughput:.1f} items/s")
                
                return result
            
            except Exception as e:
                monitor.end_monitoring()
                raise e
        
        return wrapper
    return decorator

class MemoryOptimizer:
    """Optimiseur de mémoire"""
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimise l'utilisation mémoire d'un DataFrame"""
        df_optimized = df.copy()
        
        for col in df_optimized.columns:
            col_type = df_optimized[col].dtype
            
            if col_type != 'object':
                c_min = df_optimized[col].min()
                c_max = df_optimized[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df_optimized[col] = df_optimized[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df_optimized[col] = df_optimized[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df_optimized[col] = df_optimized[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df_optimized[col] = df_optimized[col].astype(np.int64)
                
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df_optimized[col] = df_optimized[col].astype(np.float32)
                    else:
                        df_optimized[col] = df_optimized[col].astype(np.float64)
        
        return df_optimized
    
    @staticmethod
    def cleanup_memory():
        """Nettoie la mémoire"""
        gc.collect()
        
        # Statistiques mémoire
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Retourne l'utilisation mémoire actuelle"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }

class AsyncOptimizer:
    """Optimiseur pour les opérations asynchrones"""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    async def process_async_batch(self, items: List[Any], 
                                 processor_func: callable) -> List[Any]:
        """Traite un batch d'éléments de manière asynchrone"""
        async def process_item(item):
            async with self.semaphore:
                return await processor_func(item)
        
        tasks = [process_item(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrer les exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Erreur traitement item {i}: {result}")
            else:
                valid_results.append(result)
        
        return valid_results

# Fonctions utilitaires
def optimize_numpy_array(arr: np.ndarray) -> np.ndarray:
    """Optimise un array NumPy"""
    # Changer le type si possible
    if arr.dtype == np.float64:
        if np.all(arr == arr.astype(np.float32)):
            return arr.astype(np.float32)
    
    return arr

def profile_function(func: callable, *args, **kwargs) -> Dict[str, Any]:
    """Profile une fonction"""
    import cProfile
    import pstats
    import io
    
    pr = cProfile.Profile()
    pr.enable()
    
    result = func(*args, **kwargs)
    
    pr.disable()
    
    # Analyser les résultats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    return {
        'result': result,
        'profile_output': s.getvalue(),
        'total_calls': ps.total_calls,
        'total_time': ps.total_tt
    }

# Exemple d'utilisation
if __name__ == "__main__":
    # Test d'optimisation
    print("Module d'optimisation prêt")
    
    # Test du moniteur de performance
    @performance_monitor("test_operation")
    def test_function():
        time.sleep(0.1)
        return list(range(1000))
    
    result = test_function()
    print(f"Résultat test: {len(result)} items")
    
    # Test de l'optimiseur mémoire
    memory_info = MemoryOptimizer.get_memory_usage()
    print(f"Utilisation mémoire: {memory_info}")
    
    # Nettoyage mémoire
    cleanup_result = MemoryOptimizer.cleanup_memory()
    print(f"Après nettoyage: {cleanup_result}")