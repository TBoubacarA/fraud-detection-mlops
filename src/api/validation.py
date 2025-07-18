"""
Module de validation et sanitisation des données d'entrée
Protection contre les injections et validation métier
"""

import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, validator, Field
from fastapi import HTTPException, status
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Exception personnalisée pour les erreurs de validation"""
    pass

class SecurityValidator:
    """Validateur de sécurité pour les données d'entrée"""
    
    # Patterns de détection d'injections
    SQL_INJECTION_PATTERNS = [
        r"(\bunion\b.*\bselect\b)",
        r"(\bselect\b.*\bfrom\b)",
        r"(\binsert\b.*\binto\b)",
        r"(\bupdate\b.*\bset\b)",
        r"(\bdelete\b.*\bfrom\b)",
        r"(\bdrop\b.*\btable\b)",
        r"(\bexec\b.*\b)",
        r"(\bexecute\b.*\b)",
        r"(--|\#|\/\*)",
        r"(\bor\b.*\b=\b.*\bor\b)",
        r"(\band\b.*\b=\b.*\band\b)",
        r"(\bor\b.*\b1\b.*\b=\b.*\b1\b)",
        r"(\bunion\b.*\ball\b.*\bselect\b)"
    ]
    
    XSS_PATTERNS = [
        r"<script.*?>.*?</script>",
        r"javascript:",
        r"vbscript:",
        r"onload\s*=",
        r"onerror\s*=",
        r"onclick\s*=",
        r"onmouseover\s*=",
        r"<iframe.*?>",
        r"<object.*?>",
        r"<embed.*?>",
        r"<link.*?>",
        r"<meta.*?>"
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r"(\bcat\b|\bls\b|\bpwd\b|\bcd\b)",
        r"(\brm\b|\bmv\b|\bcp\b|\bchmod\b)",
        r"(\bwget\b|\bcurl\b|\bping\b)",
        r"(\bsh\b|\bbash\b|\bcsh\b|\bzsh\b)",
        r"(\bpython\b|\bperl\b|\bruby\b|\bphp\b)",
        r"(\beval\b|\bexec\b|\bsystem\b)",
        r"(&&|\|\||;|\$\(|\`)"
    ]
    
    @staticmethod
    def validate_string(value: str, field_name: str) -> str:
        """Valide et sanitise une chaîne de caractères"""
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} doit être une chaîne de caractères")
        
        # Vérifier la longueur
        if len(value) > 1000:
            raise ValidationError(f"{field_name} trop long (max 1000 caractères)")
        
        # Détection d'injections SQL
        for pattern in SecurityValidator.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Tentative d'injection SQL détectée: {field_name}")
                raise ValidationError(f"Contenu suspect détecté dans {field_name}")
        
        # Détection d'injections XSS
        for pattern in SecurityValidator.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Tentative d'injection XSS détectée: {field_name}")
                raise ValidationError(f"Contenu suspect détecté dans {field_name}")
        
        # Détection d'injections de commandes
        for pattern in SecurityValidator.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Tentative d'injection de commande détectée: {field_name}")
                raise ValidationError(f"Contenu suspect détecté dans {field_name}")
        
        return value.strip()
    
    @staticmethod
    def validate_email(email: str) -> str:
        """Valide un email"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            raise ValidationError("Format d'email invalide")
        return email.lower().strip()
    
    @staticmethod
    def validate_username(username: str) -> str:
        """Valide un nom d'utilisateur"""
        if not re.match(r'^[a-zA-Z0-9_-]{3,20}$', username):
            raise ValidationError("Nom d'utilisateur invalide (3-20 caractères, lettres, chiffres, _ et - seulement)")
        return username.lower().strip()

class TransactionValidator:
    """Validateur spécifique aux transactions financières"""
    
    @staticmethod
    def validate_transaction_amount(amount: float) -> float:
        """Valide le montant d'une transaction"""
        if not isinstance(amount, (int, float)):
            raise ValidationError("Le montant doit être un nombre")
        
        if amount < 0:
            raise ValidationError("Le montant ne peut pas être négatif")
        
        if amount > 100000:  # Limite arbitraire
            raise ValidationError("Montant trop élevé (max 100,000)")
        
        if np.isnan(amount) or np.isinf(amount):
            raise ValidationError("Montant invalide (NaN ou Inf)")
        
        return float(amount)
    
    @staticmethod
    def validate_transaction_features(features: Dict[str, float]) -> Dict[str, float]:
        """Valide les features d'une transaction"""
        validated_features = {}
        
        # Vérifier les features V1-V28
        for i in range(1, 29):
            feature_name = f"V{i}"
            if feature_name not in features:
                raise ValidationError(f"Feature manquante: {feature_name}")
            
            value = features[feature_name]
            
            # Valider le type
            if not isinstance(value, (int, float)):
                raise ValidationError(f"{feature_name} doit être un nombre")
            
            # Valider les valeurs
            if np.isnan(value) or np.isinf(value):
                raise ValidationError(f"{feature_name} contient une valeur invalide")
            
            # Valider la plage (les features PCA sont généralement entre -5 et 5)
            if abs(value) > 10:
                logger.warning(f"Feature {feature_name} hors plage normale: {value}")
            
            validated_features[feature_name] = float(value)
        
        # Valider le montant
        if "Amount" not in features:
            raise ValidationError("Feature manquante: Amount")
        
        validated_features["Amount"] = TransactionValidator.validate_transaction_amount(features["Amount"])
        
        return validated_features

# Modèles Pydantic avec validation renforcée
class SecureTransactionRequest(BaseModel):
    """Modèle sécurisé pour les requêtes de transaction"""
    
    # Features V1-V28 avec validation
    V1: float = Field(..., ge=-10, le=10, description="Feature V1 (PCA)")
    V2: float = Field(..., ge=-10, le=10, description="Feature V2 (PCA)")
    V3: float = Field(..., ge=-10, le=10, description="Feature V3 (PCA)")
    V4: float = Field(..., ge=-10, le=10, description="Feature V4 (PCA)")
    V5: float = Field(..., ge=-10, le=10, description="Feature V5 (PCA)")
    V6: float = Field(..., ge=-10, le=10, description="Feature V6 (PCA)")
    V7: float = Field(..., ge=-10, le=10, description="Feature V7 (PCA)")
    V8: float = Field(..., ge=-10, le=10, description="Feature V8 (PCA)")
    V9: float = Field(..., ge=-10, le=10, description="Feature V9 (PCA)")
    V10: float = Field(..., ge=-10, le=10, description="Feature V10 (PCA)")
    V11: float = Field(..., ge=-10, le=10, description="Feature V11 (PCA)")
    V12: float = Field(..., ge=-10, le=10, description="Feature V12 (PCA)")
    V13: float = Field(..., ge=-10, le=10, description="Feature V13 (PCA)")
    V14: float = Field(..., ge=-10, le=10, description="Feature V14 (PCA)")
    V15: float = Field(..., ge=-10, le=10, description="Feature V15 (PCA)")
    V16: float = Field(..., ge=-10, le=10, description="Feature V16 (PCA)")
    V17: float = Field(..., ge=-10, le=10, description="Feature V17 (PCA)")
    V18: float = Field(..., ge=-10, le=10, description="Feature V18 (PCA)")
    V19: float = Field(..., ge=-10, le=10, description="Feature V19 (PCA)")
    V20: float = Field(..., ge=-10, le=10, description="Feature V20 (PCA)")
    V21: float = Field(..., ge=-10, le=10, description="Feature V21 (PCA)")
    V22: float = Field(..., ge=-10, le=10, description="Feature V22 (PCA)")
    V23: float = Field(..., ge=-10, le=10, description="Feature V23 (PCA)")
    V24: float = Field(..., ge=-10, le=10, description="Feature V24 (PCA)")
    V25: float = Field(..., ge=-10, le=10, description="Feature V25 (PCA)")
    V26: float = Field(..., ge=-10, le=10, description="Feature V26 (PCA)")
    V27: float = Field(..., ge=-10, le=10, description="Feature V27 (PCA)")
    V28: float = Field(..., ge=-10, le=10, description="Feature V28 (PCA)")
    
    # Montant avec validation stricte
    Amount: float = Field(..., ge=0, le=100000, description="Montant de la transaction")
    
    # Metadata optionnelle
    transaction_id: Optional[str] = Field(None, max_length=100, description="ID de transaction")
    timestamp: Optional[str] = Field(None, max_length=50, description="Timestamp de la transaction")
    
    @validator('transaction_id')
    def validate_transaction_id(cls, v):
        """Valide l'ID de transaction"""
        if v is not None:
            return SecurityValidator.validate_string(v, "transaction_id")
        return v
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Valide le timestamp"""
        if v is not None:
            return SecurityValidator.validate_string(v, "timestamp")
        return v
    
    def validate_all_features(self) -> Dict[str, float]:
        """Valide toutes les features de la transaction"""
        features = {}
        
        # Extraire toutes les features
        for field_name, field_value in self.dict().items():
            if field_name.startswith('V') or field_name == 'Amount':
                features[field_name] = field_value
        
        # Validation approfondie
        return TransactionValidator.validate_transaction_features(features)

class SecureBatchRequest(BaseModel):
    """Modèle sécurisé pour les requêtes de batch"""
    
    transactions: List[SecureTransactionRequest] = Field(..., min_items=1, max_items=100)
    batch_id: Optional[str] = Field(None, max_length=100, description="ID du batch")
    
    @validator('batch_id')
    def validate_batch_id(cls, v):
        """Valide l'ID du batch"""
        if v is not None:
            return SecurityValidator.validate_string(v, "batch_id")
        return v
    
    @validator('transactions')
    def validate_transactions_list(cls, v):
        """Valide la liste des transactions"""
        if len(v) > 100:
            raise ValidationError("Trop de transactions (max 100)")
        return v

class SecureUserCreate(BaseModel):
    """Modèle sécurisé pour la création d'utilisateurs"""
    
    username: str = Field(..., min_length=3, max_length=20)
    email: str = Field(..., max_length=100)
    password: str = Field(..., min_length=8, max_length=128)
    role: str = Field("viewer", description="Rôle de l'utilisateur")
    
    @validator('username')
    def validate_username(cls, v):
        """Valide le nom d'utilisateur"""
        return SecurityValidator.validate_username(v)
    
    @validator('email')
    def validate_email(cls, v):
        """Valide l'email"""
        return SecurityValidator.validate_email(v)
    
    @validator('password')
    def validate_password(cls, v):
        """Valide le mot de passe"""
        # Vérifier la complexité
        if len(v) < 8:
            raise ValidationError("Mot de passe trop court (min 8 caractères)")
        
        if not re.search(r'[A-Z]', v):
            raise ValidationError("Mot de passe doit contenir au moins une majuscule")
        
        if not re.search(r'[a-z]', v):
            raise ValidationError("Mot de passe doit contenir au moins une minuscule")
        
        if not re.search(r'[0-9]', v):
            raise ValidationError("Mot de passe doit contenir au moins un chiffre")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValidationError("Mot de passe doit contenir au moins un caractère spécial")
        
        return v
    
    @validator('role')
    def validate_role(cls, v):
        """Valide le rôle"""
        allowed_roles = ['admin', 'analyst', 'viewer']
        if v not in allowed_roles:
            raise ValidationError(f"Rôle invalide. Autorisés: {allowed_roles}")
        return v

# Fonction utilitaire pour la validation
def validate_request_data(data: Dict[str, Any], model_class: BaseModel) -> BaseModel:
    """
    Valide les données d'une requête avec un modèle Pydantic
    
    Args:
        data: Données à valider
        model_class: Classe du modèle Pydantic
        
    Returns:
        Instance validée du modèle
        
    Raises:
        HTTPException: Si la validation échoue
    """
    try:
        return model_class(**data)
    except ValidationError as e:
        logger.warning(f"Erreur de validation: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Données invalides: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Erreur de validation inattendue: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Erreur de validation des données"
        )