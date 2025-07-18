"""
Tests pour le module de validation
"""

import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.api.validation import (
    SecurityValidator, TransactionValidator, SecureTransactionRequest,
    SecureBatchRequest, SecureUserCreate, ValidationError
)
from pydantic import ValidationError as PydanticValidationError

class TestSecurityValidator:
    """Tests pour SecurityValidator"""
    
    def test_validate_string_normal(self):
        """Test de validation d'une chaîne normale"""
        result = SecurityValidator.validate_string("normal_string", "test_field")
        assert result == "normal_string"
    
    def test_validate_string_with_whitespace(self):
        """Test de validation avec espaces"""
        result = SecurityValidator.validate_string("  normal_string  ", "test_field")
        assert result == "normal_string"
    
    def test_validate_string_too_long(self):
        """Test de validation d'une chaîne trop longue"""
        long_string = "a" * 1001
        with pytest.raises(ValidationError):
            SecurityValidator.validate_string(long_string, "test_field")
    
    def test_validate_string_sql_injection(self):
        """Test de détection d'injection SQL"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM users",
            "INSERT INTO users VALUES",
            "DELETE FROM users WHERE"
        ]
        
        for malicious_input in malicious_inputs:
            with pytest.raises(ValidationError):
                SecurityValidator.validate_string(malicious_input, "test_field")
    
    def test_validate_string_xss_injection(self):
        """Test de détection d'injection XSS"""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<iframe src='malicious.com'></iframe>",
            "onload=alert('xss')",
            "<object data='malicious.swf'></object>"
        ]
        
        for malicious_input in malicious_inputs:
            with pytest.raises(ValidationError):
                SecurityValidator.validate_string(malicious_input, "test_field")
    
    def test_validate_string_command_injection(self):
        """Test de détection d'injection de commande"""
        malicious_inputs = [
            "test; rm -rf /",
            "test && cat /etc/passwd",
            "test | ls -la",
            "test $(whoami)",
            "test `id`"
        ]
        
        for malicious_input in malicious_inputs:
            with pytest.raises(ValidationError):
                SecurityValidator.validate_string(malicious_input, "test_field")
    
    def test_validate_email_valid(self):
        """Test de validation d'email valide"""
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "user+tag@example.org",
            "123@example.com"
        ]
        
        for email in valid_emails:
            result = SecurityValidator.validate_email(email)
            assert result == email.lower()
    
    def test_validate_email_invalid(self):
        """Test de validation d'email invalide"""
        invalid_emails = [
            "invalid_email",
            "@example.com",
            "test@",
            "test@.com",
            "test..test@example.com",
            "test@example",
            ""
        ]
        
        for email in invalid_emails:
            with pytest.raises(ValidationError):
                SecurityValidator.validate_email(email)
    
    def test_validate_username_valid(self):
        """Test de validation de nom d'utilisateur valide"""
        valid_usernames = [
            "user123",
            "test_user",
            "admin-user",
            "user_123",
            "a1b2c3"
        ]
        
        for username in valid_usernames:
            result = SecurityValidator.validate_username(username)
            assert result == username.lower()
    
    def test_validate_username_invalid(self):
        """Test de validation de nom d'utilisateur invalide"""
        invalid_usernames = [
            "ab",  # Trop court
            "a" * 21,  # Trop long
            "user@domain",  # Caractères interdits
            "user space",  # Espaces
            "user.name",  # Points
            "123",  # Trop court
            ""  # Vide
        ]
        
        for username in invalid_usernames:
            with pytest.raises(ValidationError):
                SecurityValidator.validate_username(username)

class TestTransactionValidator:
    """Tests pour TransactionValidator"""
    
    def test_validate_transaction_amount_valid(self):
        """Test de validation de montant valide"""
        valid_amounts = [0, 10.50, 1000, 99999.99]
        
        for amount in valid_amounts:
            result = TransactionValidator.validate_transaction_amount(amount)
            assert result == float(amount)
    
    def test_validate_transaction_amount_invalid(self):
        """Test de validation de montant invalide"""
        invalid_amounts = [-10, 100001, float('nan'), float('inf')]
        
        for amount in invalid_amounts:
            with pytest.raises(ValidationError):
                TransactionValidator.validate_transaction_amount(amount)
    
    def test_validate_transaction_features_valid(self):
        """Test de validation de features valides"""
        valid_features = {}
        for i in range(1, 29):
            valid_features[f"V{i}"] = float(i * 0.1)
        valid_features["Amount"] = 100.0
        
        result = TransactionValidator.validate_transaction_features(valid_features)
        assert len(result) == 29
        assert all(isinstance(v, float) for v in result.values())
    
    def test_validate_transaction_features_missing(self):
        """Test de validation avec features manquantes"""
        incomplete_features = {"V1": 1.0, "Amount": 100.0}
        
        with pytest.raises(ValidationError):
            TransactionValidator.validate_transaction_features(incomplete_features)
    
    def test_validate_transaction_features_invalid_values(self):
        """Test de validation avec valeurs invalides"""
        invalid_features = {}
        for i in range(1, 29):
            invalid_features[f"V{i}"] = float(i * 0.1)
        
        # Tester avec NaN
        invalid_features["Amount"] = float('nan')
        with pytest.raises(ValidationError):
            TransactionValidator.validate_transaction_features(invalid_features)
        
        # Tester avec Inf
        invalid_features["Amount"] = float('inf')
        with pytest.raises(ValidationError):
            TransactionValidator.validate_transaction_features(invalid_features)

class TestSecureTransactionRequest:
    """Tests pour SecureTransactionRequest"""
    
    def test_valid_transaction_request(self):
        """Test de création d'une requête de transaction valide"""
        valid_data = {}
        for i in range(1, 29):
            valid_data[f"V{i}"] = float(i * 0.1)
        valid_data["Amount"] = 100.0
        
        request = SecureTransactionRequest(**valid_data)
        assert request.Amount == 100.0
        assert request.V1 == 0.1
    
    def test_invalid_transaction_request_out_of_range(self):
        """Test avec valeurs hors limites"""
        invalid_data = {}
        for i in range(1, 29):
            invalid_data[f"V{i}"] = float(i * 0.1)
        invalid_data["Amount"] = 100.0
        invalid_data["V1"] = 15.0  # Hors limite
        
        with pytest.raises(PydanticValidationError):
            SecureTransactionRequest(**invalid_data)
    
    def test_invalid_transaction_request_negative_amount(self):
        """Test avec montant négatif"""
        invalid_data = {}
        for i in range(1, 29):
            invalid_data[f"V{i}"] = float(i * 0.1)
        invalid_data["Amount"] = -100.0
        
        with pytest.raises(PydanticValidationError):
            SecureTransactionRequest(**invalid_data)
    
    def test_transaction_request_with_metadata(self):
        """Test avec métadonnées optionnelles"""
        valid_data = {}
        for i in range(1, 29):
            valid_data[f"V{i}"] = float(i * 0.1)
        valid_data["Amount"] = 100.0
        valid_data["transaction_id"] = "TXN_123456"
        valid_data["timestamp"] = "2024-01-01T12:00:00Z"
        
        request = SecureTransactionRequest(**valid_data)
        assert request.transaction_id == "TXN_123456"
        assert request.timestamp == "2024-01-01T12:00:00Z"
    
    def test_transaction_request_malicious_metadata(self):
        """Test avec métadonnées malicieuses"""
        valid_data = {}
        for i in range(1, 29):
            valid_data[f"V{i}"] = float(i * 0.1)
        valid_data["Amount"] = 100.0
        valid_data["transaction_id"] = "<script>alert('xss')</script>"
        
        with pytest.raises(PydanticValidationError):
            SecureTransactionRequest(**valid_data)
    
    def test_validate_all_features(self):
        """Test de validation complète des features"""
        valid_data = {}
        for i in range(1, 29):
            valid_data[f"V{i}"] = float(i * 0.1)
        valid_data["Amount"] = 100.0
        
        request = SecureTransactionRequest(**valid_data)
        features = request.validate_all_features()
        
        assert len(features) == 29
        assert features["Amount"] == 100.0
        assert features["V1"] == 0.1

class TestSecureBatchRequest:
    """Tests pour SecureBatchRequest"""
    
    def test_valid_batch_request(self):
        """Test de création d'une requête de batch valide"""
        transaction_data = {}
        for i in range(1, 29):
            transaction_data[f"V{i}"] = float(i * 0.1)
        transaction_data["Amount"] = 100.0
        
        transactions = [SecureTransactionRequest(**transaction_data) for _ in range(3)]
        
        batch_request = SecureBatchRequest(transactions=transactions)
        assert len(batch_request.transactions) == 3
    
    def test_batch_request_too_many_transactions(self):
        """Test avec trop de transactions"""
        transaction_data = {}
        for i in range(1, 29):
            transaction_data[f"V{i}"] = float(i * 0.1)
        transaction_data["Amount"] = 100.0
        
        transactions = [SecureTransactionRequest(**transaction_data) for _ in range(101)]
        
        with pytest.raises(PydanticValidationError):
            SecureBatchRequest(transactions=transactions)
    
    def test_batch_request_empty_transactions(self):
        """Test avec liste vide"""
        with pytest.raises(PydanticValidationError):
            SecureBatchRequest(transactions=[])
    
    def test_batch_request_with_batch_id(self):
        """Test avec ID de batch"""
        transaction_data = {}
        for i in range(1, 29):
            transaction_data[f"V{i}"] = float(i * 0.1)
        transaction_data["Amount"] = 100.0
        
        transactions = [SecureTransactionRequest(**transaction_data)]
        
        batch_request = SecureBatchRequest(
            transactions=transactions,
            batch_id="BATCH_123"
        )
        
        assert batch_request.batch_id == "BATCH_123"

class TestSecureUserCreate:
    """Tests pour SecureUserCreate"""
    
    def test_valid_user_create(self):
        """Test de création d'utilisateur valide"""
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "TestPassword123!",
            "role": "analyst"
        }
        
        user = SecureUserCreate(**user_data)
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == "analyst"
    
    def test_invalid_username(self):
        """Test avec nom d'utilisateur invalide"""
        user_data = {
            "username": "ab",  # Trop court
            "email": "test@example.com",
            "password": "TestPassword123!",
            "role": "analyst"
        }
        
        with pytest.raises(PydanticValidationError):
            SecureUserCreate(**user_data)
    
    def test_invalid_email(self):
        """Test avec email invalide"""
        user_data = {
            "username": "testuser",
            "email": "invalid_email",
            "password": "TestPassword123!",
            "role": "analyst"
        }
        
        with pytest.raises(PydanticValidationError):
            SecureUserCreate(**user_data)
    
    def test_weak_password(self):
        """Test avec mot de passe faible"""
        weak_passwords = [
            "12345678",  # Pas de lettres
            "password",  # Pas de chiffres/majuscules/symboles
            "Pass123",   # Pas de symboles
            "PASSWORD123!",  # Pas de minuscules
            "Password!",  # Pas de chiffres
            "Pass1!"      # Trop court
        ]
        
        for password in weak_passwords:
            user_data = {
                "username": "testuser",
                "email": "test@example.com",
                "password": password,
                "role": "analyst"
            }
            
            with pytest.raises(PydanticValidationError):
                SecureUserCreate(**user_data)
    
    def test_invalid_role(self):
        """Test avec rôle invalide"""
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "TestPassword123!",
            "role": "invalid_role"
        }
        
        with pytest.raises(PydanticValidationError):
            SecureUserCreate(**user_data)