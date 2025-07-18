#!/bin/bash

# Script de sauvegarde automatisé pour l'API de détection de fraude
# Usage: ./scripts/backup.sh [type] [destination]

set -e

BACKUP_TYPE="${1:-full}"
DESTINATION="${2:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$DESTINATION/backup_$TIMESTAMP"

# Configuration
POSTGRES_CONTAINER="fraud-postgres-prod"
REDIS_CONTAINER="fraud-redis-prod"
S3_BUCKET="${S3_BACKUP_BUCKET:-}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Créer le répertoire de sauvegarde
create_backup_directory() {
    log "Création du répertoire de sauvegarde: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    
    # Créer un fichier de métadonnées
    cat > "$BACKUP_DIR/metadata.json" <<EOF
{
    "timestamp": "$TIMESTAMP",
    "backup_type": "$BACKUP_TYPE",
    "environment": "${ENVIRONMENT:-production}",
    "version": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "created_by": "$(whoami)",
    "hostname": "$(hostname)"
}
EOF
    
    log "Répertoire de sauvegarde créé"
}

# Sauvegarde de la base de données PostgreSQL
backup_postgres() {
    log "Sauvegarde de la base de données PostgreSQL..."
    
    if docker ps | grep -q "$POSTGRES_CONTAINER"; then
        # Dump de la base de données
        docker exec "$POSTGRES_CONTAINER" pg_dump \
            -U "${POSTGRES_USER:-frauduser}" \
            -d "${POSTGRES_DB:-frauddb}" \
            --no-password \
            | gzip > "$BACKUP_DIR/database.sql.gz"
        
        # Vérifier la taille du backup
        DB_SIZE=$(stat -c%s "$BACKUP_DIR/database.sql.gz" 2>/dev/null || stat -f%z "$BACKUP_DIR/database.sql.gz" 2>/dev/null || echo 0)
        if [[ $DB_SIZE -gt 1024 ]]; then
            log "✓ Sauvegarde PostgreSQL réussie ($(numfmt --to=iec $DB_SIZE))"
        else
            error "Sauvegarde PostgreSQL échouée ou vide"
        fi
    else
        warn "Conteneur PostgreSQL non trouvé, ignoré"
    fi
}

# Sauvegarde de Redis
backup_redis() {
    log "Sauvegarde de Redis..."
    
    if docker ps | grep -q "$REDIS_CONTAINER"; then
        # Déclencher une sauvegarde Redis
        docker exec "$REDIS_CONTAINER" redis-cli BGSAVE
        
        # Attendre que la sauvegarde soit terminée
        while docker exec "$REDIS_CONTAINER" redis-cli LASTSAVE | grep -q "$(docker exec "$REDIS_CONTAINER" redis-cli LASTSAVE)"; do
            sleep 1
        done
        
        # Copier le dump Redis
        docker cp "$REDIS_CONTAINER:/data/dump.rdb" "$BACKUP_DIR/redis.rdb"
        
        if [[ -f "$BACKUP_DIR/redis.rdb" ]]; then
            log "✓ Sauvegarde Redis réussie"
        else
            warn "Sauvegarde Redis échouée"
        fi
    else
        warn "Conteneur Redis non trouvé, ignoré"
    fi
}

# Sauvegarde des modèles ML
backup_models() {
    log "Sauvegarde des modèles ML..."
    
    if [[ -d "./models" ]]; then
        tar -czf "$BACKUP_DIR/models.tar.gz" models/
        
        if [[ -f "$BACKUP_DIR/models.tar.gz" ]]; then
            MODEL_SIZE=$(stat -c%s "$BACKUP_DIR/models.tar.gz" 2>/dev/null || stat -f%z "$BACKUP_DIR/models.tar.gz" 2>/dev/null || echo 0)
            log "✓ Sauvegarde des modèles réussie ($(numfmt --to=iec $MODEL_SIZE))"
        else
            warn "Sauvegarde des modèles échouée"
        fi
    else
        warn "Répertoire des modèles non trouvé, ignoré"
    fi
}

# Sauvegarde des données
backup_data() {
    log "Sauvegarde des données..."
    
    if [[ -d "./data" ]]; then
        tar -czf "$BACKUP_DIR/data.tar.gz" data/
        
        if [[ -f "$BACKUP_DIR/data.tar.gz" ]]; then
            DATA_SIZE=$(stat -c%s "$BACKUP_DIR/data.tar.gz" 2>/dev/null || stat -f%z "$BACKUP_DIR/data.tar.gz" 2>/dev/null || echo 0)
            log "✓ Sauvegarde des données réussie ($(numfmt --to=iec $DATA_SIZE))"
        else
            warn "Sauvegarde des données échouée"
        fi
    else
        warn "Répertoire des données non trouvé, ignoré"
    fi
}

# Sauvegarde des logs
backup_logs() {
    log "Sauvegarde des logs..."
    
    if [[ -d "./logs" ]]; then
        tar -czf "$BACKUP_DIR/logs.tar.gz" logs/
        
        if [[ -f "$BACKUP_DIR/logs.tar.gz" ]]; then
            LOG_SIZE=$(stat -c%s "$BACKUP_DIR/logs.tar.gz" 2>/dev/null || stat -f%z "$BACKUP_DIR/logs.tar.gz" 2>/dev/null || echo 0)
            log "✓ Sauvegarde des logs réussie ($(numfmt --to=iec $LOG_SIZE))"
        else
            warn "Sauvegarde des logs échouée"
        fi
    else
        warn "Répertoire des logs non trouvé, ignoré"
    fi
}

# Sauvegarde des configurations
backup_configs() {
    log "Sauvegarde des configurations..."
    
    # Créer une archive des fichiers de configuration
    tar -czf "$BACKUP_DIR/configs.tar.gz" \
        docker-compose.yml \
        docker-compose.prod.yml \
        .env.* \
        configs/ \
        docker/ \
        2>/dev/null || true
    
    if [[ -f "$BACKUP_DIR/configs.tar.gz" ]]; then
        CONFIG_SIZE=$(stat -c%s "$BACKUP_DIR/configs.tar.gz" 2>/dev/null || stat -f%z "$BACKUP_DIR/configs.tar.gz" 2>/dev/null || echo 0)
        log "✓ Sauvegarde des configurations réussie ($(numfmt --to=iec $CONFIG_SIZE))"
    else
        warn "Sauvegarde des configurations échouée"
    fi
}

# Upload vers S3 (optionnel)
upload_to_s3() {
    if [[ -n "$S3_BUCKET" ]]; then
        log "Upload vers S3: $S3_BUCKET"
        
        if command -v aws &> /dev/null; then
            tar -czf "$BACKUP_DIR.tar.gz" -C "$DESTINATION" "backup_$TIMESTAMP"
            
            aws s3 cp "$BACKUP_DIR.tar.gz" "s3://$S3_BUCKET/fraud-detection-backups/"
            
            if [[ $? -eq 0 ]]; then
                log "✓ Upload S3 réussi"
                rm "$BACKUP_DIR.tar.gz"
            else
                error "Upload S3 échoué"
            fi
        else
            warn "AWS CLI non disponible, upload S3 ignoré"
        fi
    else
        info "S3_BUCKET non configuré, upload ignoré"
    fi
}

# Nettoyage des anciens backups
cleanup_old_backups() {
    log "Nettoyage des anciens backups (> $RETENTION_DAYS jours)..."
    
    if [[ -d "$DESTINATION" ]]; then
        find "$DESTINATION" -name "backup_*" -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \; 2>/dev/null || true
        
        REMAINING_BACKUPS=$(find "$DESTINATION" -name "backup_*" -type d | wc -l)
        log "Backups restants: $REMAINING_BACKUPS"
    fi
}

# Vérification de l'intégrité
verify_backup() {
    log "Vérification de l'intégrité du backup..."
    
    BACKUP_SIZE=0
    FILE_COUNT=0
    
    if [[ -d "$BACKUP_DIR" ]]; then
        for file in "$BACKUP_DIR"/*; do
            if [[ -f "$file" ]]; then
                FILE_COUNT=$((FILE_COUNT + 1))
                FILE_SIZE=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
                BACKUP_SIZE=$((BACKUP_SIZE + FILE_SIZE))
            fi
        done
    fi
    
    log "Backup créé avec $FILE_COUNT fichiers ($(numfmt --to=iec $BACKUP_SIZE))"
    
    # Générer un checksum
    if command -v sha256sum &> /dev/null; then
        find "$BACKUP_DIR" -type f -exec sha256sum {} \; > "$BACKUP_DIR/checksums.sha256"
        log "✓ Checksums générés"
    fi
}

# Sauvegarde complète
full_backup() {
    log "Démarrage de la sauvegarde complète..."
    
    create_backup_directory
    backup_postgres
    backup_redis
    backup_models
    backup_data
    backup_logs
    backup_configs
    verify_backup
    upload_to_s3
    cleanup_old_backups
    
    log "Sauvegarde complète terminée: $BACKUP_DIR"
}

# Sauvegarde des données uniquement
data_backup() {
    log "Démarrage de la sauvegarde des données..."
    
    create_backup_directory
    backup_postgres
    backup_redis
    backup_models
    backup_data
    verify_backup
    upload_to_s3
    
    log "Sauvegarde des données terminée: $BACKUP_DIR"
}

# Sauvegarde des configurations uniquement
config_backup() {
    log "Démarrage de la sauvegarde des configurations..."
    
    create_backup_directory
    backup_configs
    verify_backup
    upload_to_s3
    
    log "Sauvegarde des configurations terminée: $BACKUP_DIR"
}

# Fonction principale
main() {
    log "Démarrage du backup de type: $BACKUP_TYPE"
    
    case $BACKUP_TYPE in
        "full")
            full_backup
            ;;
        "data")
            data_backup
            ;;
        "config")
            config_backup
            ;;
        *)
            error "Type de backup non valide: $BACKUP_TYPE. Utilisez 'full', 'data' ou 'config'"
            ;;
    esac
    
    log "Backup terminé avec succès!"
}

# Gestion des signaux
trap 'error "Backup interrompu par l'\''utilisateur"' SIGINT SIGTERM

# Exécution
main "$@"