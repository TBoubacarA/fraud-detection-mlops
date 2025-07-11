#!/bin/bash
# Lancer docker compose
docker compose --env-file .env up -d --build
echo ""
echo "[✓] Lancement terminé !"
echo "🌐 MLflow UI      → http://localhost:5001"
echo "🌐 pgAdmin        → http://localhost:8081"
