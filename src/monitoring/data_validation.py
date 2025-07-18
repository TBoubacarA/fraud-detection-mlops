"""
Module de validation des données pour le monitoring
Validation de la qualité des données d'entrée
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Niveaux de sévérité des validations"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Résultat d'une validation"""
    check_name: str
    status: str  # "passed", "failed", "warning"
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime

class DataQualityValidator:
    """Validateur de qualité des données"""
    
    def __init__(self, reference_schema: Dict[str, Any]):
        self.reference_schema = reference_schema
        self.validation_results = []
    
    def validate_schema(self, data: pd.DataFrame) -> ValidationResult:
        """Valide le schéma des données"""
        expected_columns = set(self.reference_schema.get('columns', []))
        actual_columns = set(data.columns)
        
        missing_columns = expected_columns - actual_columns
        extra_columns = actual_columns - expected_columns
        
        if missing_columns or extra_columns:
            status = "failed"
            severity = ValidationSeverity.ERROR
            message = f"Schéma invalide: {len(missing_columns)} colonnes manquantes, {len(extra_columns)} colonnes supplémentaires"
            details = {
                'missing_columns': list(missing_columns),
                'extra_columns': list(extra_columns),
                'expected_columns': list(expected_columns),
                'actual_columns': list(actual_columns)
            }
        else:
            status = "passed"
            severity = ValidationSeverity.INFO
            message = "Schéma valide"
            details = {'columns_count': len(actual_columns)}
        
        return ValidationResult(
            check_name="schema_validation",
            status=status,
            severity=severity,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
    
    def validate_data_types(self, data: pd.DataFrame) -> ValidationResult:
        """Valide les types de données"""
        type_errors = []
        expected_types = self.reference_schema.get('dtypes', {})
        
        for column, expected_type in expected_types.items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                if actual_type != expected_type:
                    type_errors.append({
                        'column': column,
                        'expected_type': expected_type,
                        'actual_type': actual_type
                    })
        
        if type_errors:
            status = "failed"
            severity = ValidationSeverity.ERROR
            message = f"Types de données invalides: {len(type_errors)} colonnes"
            details = {'type_errors': type_errors}
        else:
            status = "passed"
            severity = ValidationSeverity.INFO
            message = "Types de données valides"
            details = {'validated_columns': len(expected_types)}
        
        return ValidationResult(
            check_name="data_types_validation",
            status=status,
            severity=severity,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
    
    def validate_missing_values(self, data: pd.DataFrame, 
                               max_missing_rate: float = 0.1) -> ValidationResult:
        """Valide les valeurs manquantes"""
        missing_stats = {}
        issues = []
        
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            missing_rate = missing_count / len(data)
            
            missing_stats[column] = {
                'count': int(missing_count),
                'rate': float(missing_rate)
            }
            
            if missing_rate > max_missing_rate:
                issues.append({
                    'column': column,
                    'missing_rate': missing_rate,
                    'missing_count': missing_count
                })
        
        if issues:
            status = "warning"
            severity = ValidationSeverity.WARNING
            message = f"Taux de valeurs manquantes élevé: {len(issues)} colonnes"
            details = {'issues': issues, 'missing_stats': missing_stats}
        else:
            status = "passed"
            severity = ValidationSeverity.INFO
            message = "Valeurs manquantes acceptables"
            details = {'missing_stats': missing_stats}
        
        return ValidationResult(
            check_name="missing_values_validation",
            status=status,
            severity=severity,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
    
    def validate_data_ranges(self, data: pd.DataFrame) -> ValidationResult:
        """Valide les plages de valeurs"""
        range_config = self.reference_schema.get('ranges', {})
        out_of_range_issues = []
        
        for column, ranges in range_config.items():
            if column in data.columns:
                min_val = ranges.get('min')
                max_val = ranges.get('max')
                
                if min_val is not None:
                    below_min = (data[column] < min_val).sum()
                    if below_min > 0:
                        out_of_range_issues.append({
                            'column': column,
                            'issue': 'below_minimum',
                            'count': int(below_min),
                            'min_value': min_val
                        })
                
                if max_val is not None:
                    above_max = (data[column] > max_val).sum()
                    if above_max > 0:
                        out_of_range_issues.append({
                            'column': column,
                            'issue': 'above_maximum',
                            'count': int(above_max),
                            'max_value': max_val
                        })
        
        if out_of_range_issues:
            status = "warning"
            severity = ValidationSeverity.WARNING
            message = f"Valeurs hors plage: {len(out_of_range_issues)} issues"
            details = {'out_of_range_issues': out_of_range_issues}
        else:
            status = "passed"
            severity = ValidationSeverity.INFO
            message = "Plages de valeurs valides"
            details = {'validated_columns': len(range_config)}
        
        return ValidationResult(
            check_name="data_ranges_validation",
            status=status,
            severity=severity,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
    
    def validate_duplicates(self, data: pd.DataFrame, 
                           max_duplicate_rate: float = 0.05) -> ValidationResult:
        """Valide les doublons"""
        duplicate_count = data.duplicated().sum()
        duplicate_rate = duplicate_count / len(data)
        
        if duplicate_rate > max_duplicate_rate:
            status = "warning"
            severity = ValidationSeverity.WARNING
            message = f"Taux de doublons élevé: {duplicate_rate:.2%}"
            details = {
                'duplicate_count': int(duplicate_count),
                'duplicate_rate': float(duplicate_rate),
                'threshold': max_duplicate_rate
            }
        else:
            status = "passed"
            severity = ValidationSeverity.INFO
            message = f"Taux de doublons acceptable: {duplicate_rate:.2%}"
            details = {
                'duplicate_count': int(duplicate_count),
                'duplicate_rate': float(duplicate_rate)
            }
        
        return ValidationResult(
            check_name="duplicates_validation",
            status=status,
            severity=severity,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
    
    def validate_statistical_properties(self, data: pd.DataFrame) -> ValidationResult:
        """Valide les propriétés statistiques"""
        stats_config = self.reference_schema.get('statistics', {})
        anomalies = []
        
        for column, expected_stats in stats_config.items():
            if column in data.columns and data[column].dtype in ['int64', 'float64']:
                actual_stats = {
                    'mean': float(data[column].mean()),
                    'std': float(data[column].std()),
                    'min': float(data[column].min()),
                    'max': float(data[column].max())
                }
                
                # Vérifier les écarts significatifs
                for stat, expected_value in expected_stats.items():
                    if stat in actual_stats:
                        actual_value = actual_stats[stat]
                        if expected_value != 0:
                            relative_diff = abs(actual_value - expected_value) / abs(expected_value)
                            if relative_diff > 0.5:  # Écart > 50%
                                anomalies.append({
                                    'column': column,
                                    'statistic': stat,
                                    'expected': expected_value,
                                    'actual': actual_value,
                                    'relative_diff': relative_diff
                                })
        
        if anomalies:
            status = "warning"
            severity = ValidationSeverity.WARNING
            message = f"Propriétés statistiques anormales: {len(anomalies)} anomalies"
            details = {'anomalies': anomalies}
        else:
            status = "passed"
            severity = ValidationSeverity.INFO
            message = "Propriétés statistiques normales"
            details = {'validated_columns': len(stats_config)}
        
        return ValidationResult(
            check_name="statistical_properties_validation",
            status=status,
            severity=severity,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
    
    def validate_business_rules(self, data: pd.DataFrame) -> ValidationResult:
        """Valide les règles métier"""
        business_rules = self.reference_schema.get('business_rules', [])
        rule_violations = []
        
        for rule in business_rules:
            rule_name = rule.get('name', 'unknown')
            rule_condition = rule.get('condition', '')
            
            try:
                # Évaluer la condition (attention: seulement pour des conditions sûres)
                if rule_condition:
                    violations = data.query(f"not ({rule_condition})")
                    if len(violations) > 0:
                        rule_violations.append({
                            'rule_name': rule_name,
                            'condition': rule_condition,
                            'violations_count': len(violations)
                        })
            except Exception as e:
                logger.error(f"Erreur évaluation règle {rule_name}: {e}")
        
        if rule_violations:
            status = "warning"
            severity = ValidationSeverity.WARNING
            message = f"Violations de règles métier: {len(rule_violations)} règles"
            details = {'rule_violations': rule_violations}
        else:
            status = "passed"
            severity = ValidationSeverity.INFO
            message = "Règles métier respectées"
            details = {'evaluated_rules': len(business_rules)}
        
        return ValidationResult(
            check_name="business_rules_validation",
            status=status,
            severity=severity,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
    
    def run_all_validations(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Exécute toutes les validations"""
        validations = [
            self.validate_schema(data),
            self.validate_data_types(data),
            self.validate_missing_values(data),
            self.validate_data_ranges(data),
            self.validate_duplicates(data),
            self.validate_statistical_properties(data),
            self.validate_business_rules(data)
        ]
        
        self.validation_results.extend(validations)
        return validations
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des validations"""
        if not self.validation_results:
            return {}
        
        status_counts = {}
        severity_counts = {}
        
        for result in self.validation_results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
            severity_counts[result.severity.value] = severity_counts.get(result.severity.value, 0) + 1
        
        return {
            'total_validations': len(self.validation_results),
            'status_counts': status_counts,
            'severity_counts': severity_counts,
            'last_validation': self.validation_results[-1].timestamp if self.validation_results else None
        }

class DataProfiler:
    """Profileur de données pour générer des schémas de référence"""
    
    def __init__(self):
        pass
    
    def profile_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Génère un profil complet des données"""
        profile = {
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'shape': data.shape,
            'statistics': {},
            'ranges': {},
            'missing_values': {},
            'duplicates': {
                'count': int(data.duplicated().sum()),
                'rate': float(data.duplicated().sum() / len(data))
            },
            'created_at': datetime.now().isoformat()
        }
        
        # Statistiques par colonne
        for column in data.columns:
            col_data = data[column]
            
            # Valeurs manquantes
            missing_count = col_data.isnull().sum()
            profile['missing_values'][column] = {
                'count': int(missing_count),
                'rate': float(missing_count / len(data))
            }
            
            # Statistiques pour les colonnes numériques
            if col_data.dtype in ['int64', 'float64']:
                profile['statistics'][column] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'q25': float(col_data.quantile(0.25)),
                    'q50': float(col_data.quantile(0.50)),
                    'q75': float(col_data.quantile(0.75))
                }
                
                profile['ranges'][column] = {
                    'min': float(col_data.min()),
                    'max': float(col_data.max())
                }
        
        return profile
    
    def generate_validation_schema(self, data: pd.DataFrame, 
                                  strict: bool = False) -> Dict[str, Any]:
        """Génère un schéma de validation à partir des données"""
        profile = self.profile_data(data)
        
        # Schéma de base
        schema = {
            'columns': profile['columns'],
            'dtypes': profile['dtypes'],
            'ranges': profile['ranges'],
            'statistics': profile['statistics']
        }
        
        # Règles métier basiques
        business_rules = []
        
        # Exemple de règles pour les données de fraude
        if 'Amount' in data.columns:
            business_rules.append({
                'name': 'amount_positive',
                'condition': 'Amount >= 0',
                'description': 'Le montant doit être positif'
            })
        
        if 'Class' in data.columns:
            business_rules.append({
                'name': 'class_binary',
                'condition': 'Class in [0, 1]',
                'description': 'La classe doit être 0 ou 1'
            })
        
        schema['business_rules'] = business_rules
        
        # Seuils stricts ou permissifs
        if strict:
            schema['max_missing_rate'] = 0.01  # 1% max
            schema['max_duplicate_rate'] = 0.01  # 1% max
        else:
            schema['max_missing_rate'] = 0.1  # 10% max
            schema['max_duplicate_rate'] = 0.05  # 5% max
        
        return schema
    
    def save_schema(self, schema: Dict[str, Any], filepath: str):
        """Sauvegarde le schéma dans un fichier"""
        try:
            with open(filepath, 'w') as f:
                json.dump(schema, f, indent=2, default=str)
            logger.info(f"Schéma sauvegardé: {filepath}")
        except Exception as e:
            logger.error(f"Erreur sauvegarde schéma: {e}")
    
    def load_schema(self, filepath: str) -> Dict[str, Any]:
        """Charge un schéma depuis un fichier"""
        try:
            with open(filepath, 'r') as f:
                schema = json.load(f)
            logger.info(f"Schéma chargé: {filepath}")
            return schema
        except Exception as e:
            logger.error(f"Erreur chargement schéma: {e}")
            return {}

# Fonction utilitaire pour valider des données
def validate_data(data: pd.DataFrame, schema_path: str) -> List[ValidationResult]:
    """Valide des données avec un schéma"""
    try:
        # Charger le schéma
        profiler = DataProfiler()
        schema = profiler.load_schema(schema_path)
        
        # Créer le validateur
        validator = DataQualityValidator(schema)
        
        # Exécuter les validations
        results = validator.run_all_validations(data)
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur validation données: {e}")
        return []

# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple de création de schéma de référence
    profiler = DataProfiler()
    
    # Charger des données d'exemple
    # data = pd.read_csv("data/processed/train.csv")
    
    # Générer un schéma
    # schema = profiler.generate_validation_schema(data)
    # profiler.save_schema(schema, "configs/data_validation_schema.json")
    
    # Valider de nouvelles données
    # new_data = pd.read_csv("data/processed/test.csv")
    # results = validate_data(new_data, "configs/data_validation_schema.json")
    
    # Afficher les résultats
    # for result in results:
    #     print(f"{result.check_name}: {result.status} - {result.message}")
    
    print("Module de validation des données prêt")