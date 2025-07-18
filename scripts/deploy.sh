#!/bin/bash

# Script de déploiement automatisé pour l'API de détection de fraude
# Usage: ./scripts/deploy.sh [environment] [version]

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENVIRONMENT="${1:-staging}"
VERSION="${2:-latest}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io}"
IMAGE_NAME="${IMAGE_NAME:-fraud-detection-api}"
COMPOSE_FILE="docker-compose.prod.yml"

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction de logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Vérification des prérequis
check_prerequisites() {
    log "Vérification des prérequis..."
    
    # Vérifier Docker
    if ! command -v docker &> /dev/null; then
        error "Docker n'est pas installé"
    fi
    
    # Vérifier Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose n'est pas installé"
    fi
    
    # Vérifier les permissions
    if ! docker ps &> /dev/null; then
        error "Permissions Docker insuffisantes"
    fi
    
    # Vérifier les fichiers de configuration
    if [[ ! -f "$PROJECT_DIR/$COMPOSE_FILE" ]]; then
        error "Fichier docker-compose non trouvé: $COMPOSE_FILE"
    fi
    
    log "Prérequis vérifiés avec succès"
}

# Validation de l'environnement
validate_environment() {
    log "Validation de l'environnement: $ENVIRONMENT"
    
    case $ENVIRONMENT in
        staging|production)
            log "Environnement valide: $ENVIRONMENT"
            ;;
        *)
            error "Environnement non valide: $ENVIRONMENT. Utilisez 'staging' ou 'production'"
            ;;
    esac
}

# Chargement des variables d'environnement
load_environment_variables() {
    log "Chargement des variables d'environnement..."
    
    # Fichier d'environnement principal
    if [[ -f "$PROJECT_DIR/.env" ]]; then
        source "$PROJECT_DIR/.env"
        log "Variables d'environnement chargées depuis .env"
    fi
    
    # Fichier d'environnement spécifique
    ENV_FILE="$PROJECT_DIR/.env.$ENVIRONMENT"
    if [[ -f "$ENV_FILE" ]]; then
        source "$ENV_FILE"
        log "Variables d'environnement chargées depuis .env.$ENVIRONMENT"
    else
        warn "Fichier d'environnement non trouvé: $ENV_FILE"
    fi
    
    # Vérification des variables obligatoires
    REQUIRED_VARS=(
        "POSTGRES_PASSWORD"
        "REDIS_PASSWORD"
        "JWT_SECRET_KEY"
    )
    
    for var in "${REQUIRED_VARS[@]}"; do
        if [[ -z "${!var}" ]]; then
            error "Variable d'environnement manquante: $var"
        fi
    done
    
    log "Variables d'environnement validées"
}

# Backup des données
backup_data() {
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "Création d'un backup des données..."
        
        BACKUP_DIR="$PROJECT_DIR/backups/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$BACKUP_DIR"
        
        # Backup de la base de données
        if docker ps | grep -q "fraud-postgres-prod"; then
            docker exec fraud-postgres-prod pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" | gzip > "$BACKUP_DIR/database.sql.gz"
            log "Backup de la base de données créé"
        fi
        
        # Backup des modèles
        if [[ -d "$PROJECT_DIR/models" ]]; then
            tar -czf "$BACKUP_DIR/models.tar.gz" -C "$PROJECT_DIR" models/
            log "Backup des modèles créé"
        fi
        
        # Backup des métriques
        if [[ -d "$PROJECT_DIR/logs" ]]; then
            tar -czf "$BACKUP_DIR/logs.tar.gz" -C "$PROJECT_DIR" logs/
            log "Backup des logs créé"
        fi
        
        log "Backup terminé: $BACKUP_DIR"
    fi
}

# Arrêt des services existants
stop_services() {
    log "Arrêt des services existants..."
    
    cd "$PROJECT_DIR"
    
    # Arrêt gracieux avec timeout
    docker-compose -f "$COMPOSE_FILE" stop --timeout 30 || true
    
    # Attendre que les conteneurs s'arrêtent
    sleep 5
    
    # Forcer l'arrêt si nécessaire
    docker-compose -f "$COMPOSE_FILE" down --timeout 10 || true
    
    log "Services arrêtés"
}

# Téléchargement des nouvelles images
pull_images() {
    log "Téléchargement des nouvelles images..."
    
    cd "$PROJECT_DIR"
    
    # Télécharger les images
    docker-compose -f "$COMPOSE_FILE" pull
    
    log "Images téléchargées"
}

# Démarrage des services
start_services() {
    log "Démarrage des services..."
    
    cd "$PROJECT_DIR"
    
    # Démarrer les services
    docker-compose -f "$COMPOSE_FILE" up -d
    
    log "Services démarrés"
}

# Vérification de la santé des services
health_check() {
    log "Vérification de la santé des services..."
    
    # Attendre que les services démarrent
    sleep 10
    
    # Vérifier la base de données
    if ! docker exec fraud-postgres-prod pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" &> /dev/null; then
        error "Base de données non accessible"
    fi
    log "✓ Base de données OK"
    
    # Vérifier Redis
    if ! docker exec fraud-redis-prod redis-cli ping &> /dev/null; then
        error "Redis non accessible"
    fi
    log "✓ Redis OK"
    
    # Vérifier MLflow
    sleep 5
    if ! curl -f http://localhost:5000/health &> /dev/null; then
        error "MLflow non accessible"
    fi
    log "✓ MLflow OK"
    
    # Vérifier l'API
    sleep 5
    MAX_RETRIES=10
    RETRY_COUNT=0
    
    while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
        if curl -f http://localhost:8001/health &> /dev/null; then
            log "✓ API OK"
            break
        fi
        
        RETRY_COUNT=$((RETRY_COUNT + 1))
        warn "Tentative $RETRY_COUNT/$MAX_RETRIES - API non accessible, nouvelle tentative dans 10s..."
        sleep 10
    done
    
    if [[ $RETRY_COUNT -eq $MAX_RETRIES ]]; then
        error "API non accessible après $MAX_RETRIES tentatives"
    fi
    
    log "Tous les services sont opérationnels"
}

# Tests de fumée
smoke_tests() {
    log "Exécution des tests de fumée..."
    
    # Test de l'endpoint de santé
    HEALTH_RESPONSE=$(curl -s http://localhost:8001/health)
    if [[ "$HEALTH_RESPONSE" == *"healthy"* ]]; then
        log "✓ Test de santé réussi"
    else
        error "Test de santé échoué"
    fi
    
    # Test de l'endpoint des métriques (nécessite auth)
    # Note: Adapté selon votre configuration d'auth
    
    log "Tests de fumée terminés avec succès"
}

# Nettoyage des anciennes images
cleanup() {
    log "Nettoyage des anciennes images..."
    
    # Supprimer les images non utilisées
    docker image prune -f
    
    # Supprimer les volumes non utilisés
    docker volume prune -f
    
    log "Nettoyage terminé"
}

# Rollback en cas d'échec
rollback() {
    error "Échec du déploiement, rollback en cours..."
    
    # Arrêter les services actuels
    docker-compose -f "$COMPOSE_FILE" down --timeout 10 || true
    
    # Restaurer depuis le backup si disponible
    if [[ -d "$PROJECT_DIR/backups" ]]; then
        LATEST_BACKUP=$(ls -t "$PROJECT_DIR/backups" | head -1)
        if [[ -n "$LATEST_BACKUP" ]]; then
            warn "Restauration depuis le backup: $LATEST_BACKUP"
            # Logique de restauration à implémenter selon vos besoins
        fi
    fi
    
    error "Rollback terminé"
}

# Notification du déploiement
notify_deployment() {
    log "Notification du déploiement..."
    
    # Slack notification (si configuré)
    if [[ -n "$SLACK_WEBHOOK_URL" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Déploiement réussi en $ENVIRONMENT - Version: $VERSION\"}" \
            "$SLACK_WEBHOOK_URL" &> /dev/null || true
    fi
    
    # Email notification (si configuré)
    if [[ -n "$EMAIL_RECIPIENT" ]]; then
        echo "Déploiement réussi en $ENVIRONMENT - Version: $VERSION" | \
            mail -s "Déploiement Fraud Detection API" "$EMAIL_RECIPIENT" &> /dev/null || true
    fi
    
    log "Notifications envoyées"
}

# Fonction principale
main() {
    log "Démarrage du déploiement..."
    log "Environnement: $ENVIRONMENT"
    log "Version: $VERSION"
    
    # Piège pour gérer les erreurs
    trap rollback ERR
    
    # Étapes du déploiement
    check_prerequisites
    validate_environment
    load_environment_variables
    backup_data
    stop_services
    pull_images
    start_services
    health_check
    smoke_tests
    cleanup
    notify_deployment
    
    log "Déploiement terminé avec succès!"
    log "Services disponibles:"
    log "  - API: http://localhost:8001"
    log "  - MLflow: http://localhost:5000"
    log "  - Grafana: http://localhost:3000"
    log "  - Prometheus: http://localhost:9090"
}

# Gestion des signaux
trap 'error "Déploiement interrompu par l'\''utilisateur"' SIGINT SIGTERM

# Exécution
main "$@"