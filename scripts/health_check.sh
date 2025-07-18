#!/bin/bash

# Script de vérification de santé pour l'API de détection de fraude
# Usage: ./scripts/health_check.sh [environment]

set -e

ENVIRONMENT="${1:-production}"
API_URL="${API_URL:-http://localhost:8001}"
TIMEOUT=30

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    return 1
}

# Test de base
test_health_endpoint() {
    log "Test de l'endpoint de santé..."
    
    if curl -f -s --max-time $TIMEOUT "$API_URL/health" > /dev/null; then
        log "✓ Endpoint de santé accessible"
        return 0
    else
        error "✗ Endpoint de santé non accessible"
        return 1
    fi
}

# Test des métriques (nécessite auth)
test_metrics_endpoint() {
    log "Test de l'endpoint des métriques..."
    
    # Pour tester les métriques, on a besoin d'un token
    # Ici on teste juste l'accessibilité sans auth
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT "$API_URL/metrics")
    
    if [[ "$STATUS" == "403" ]]; then
        log "✓ Endpoint des métriques accessible (auth requise)"
        return 0
    elif [[ "$STATUS" == "200" ]]; then
        log "✓ Endpoint des métriques accessible"
        return 0
    else
        error "✗ Endpoint des métriques non accessible (status: $STATUS)"
        return 1
    fi
}

# Test de performance
test_performance() {
    log "Test de performance..."
    
    # Mesurer le temps de réponse
    START_TIME=$(date +%s%3N)
    
    if curl -f -s --max-time $TIMEOUT "$API_URL/health" > /dev/null; then
        END_TIME=$(date +%s%3N)
        RESPONSE_TIME=$((END_TIME - START_TIME))
        
        if [[ $RESPONSE_TIME -lt 1000 ]]; then
            log "✓ Temps de réponse OK: ${RESPONSE_TIME}ms"
            return 0
        else
            warn "⚠ Temps de réponse élevé: ${RESPONSE_TIME}ms"
            return 1
        fi
    else
        error "✗ Test de performance échoué"
        return 1
    fi
}

# Test de la base de données
test_database() {
    log "Test de la base de données..."
    
    if docker ps | grep -q "fraud-postgres-prod"; then
        if docker exec fraud-postgres-prod pg_isready -U "${POSTGRES_USER:-frauduser}" -d "${POSTGRES_DB:-frauddb}" &> /dev/null; then
            log "✓ Base de données accessible"
            return 0
        else
            error "✗ Base de données non accessible"
            return 1
        fi
    else
        warn "⚠ Conteneur PostgreSQL non trouvé"
        return 1
    fi
}

# Test de Redis
test_redis() {
    log "Test de Redis..."
    
    if docker ps | grep -q "fraud-redis-prod"; then
        if docker exec fraud-redis-prod redis-cli ping &> /dev/null; then
            log "✓ Redis accessible"
            return 0
        else
            error "✗ Redis non accessible"
            return 1
        fi
    else
        warn "⚠ Conteneur Redis non trouvé"
        return 1
    fi
}

# Test MLflow
test_mlflow() {
    log "Test de MLflow..."
    
    if curl -f -s --max-time $TIMEOUT "http://localhost:5000/health" > /dev/null; then
        log "✓ MLflow accessible"
        return 0
    else
        error "✗ MLflow non accessible"
        return 1
    fi
}

# Test complet
run_all_tests() {
    log "Exécution des tests de santé pour l'environnement: $ENVIRONMENT"
    
    TESTS=(
        "test_health_endpoint"
        "test_metrics_endpoint"
        "test_performance"
        "test_database"
        "test_redis"
        "test_mlflow"
    )
    
    PASSED=0
    FAILED=0
    
    for test in "${TESTS[@]}"; do
        if $test; then
            PASSED=$((PASSED + 1))
        else
            FAILED=$((FAILED + 1))
        fi
        echo
    done
    
    log "Résultats des tests:"
    log "  ✓ Tests réussis: $PASSED"
    if [[ $FAILED -gt 0 ]]; then
        error "  ✗ Tests échoués: $FAILED"
        return 1
    else
        log "  ✗ Tests échoués: $FAILED"
        log "Tous les tests ont réussi!"
        return 0
    fi
}

# Point d'entrée
main() {
    run_all_tests
}

main "$@"