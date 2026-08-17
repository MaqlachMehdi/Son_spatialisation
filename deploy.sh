#!/usr/bin/env bash
# Déploie la dernière version de main sur le serveur de prod : pull, rebuild
# des conteneurs modifiés, migrations, puis vérifie que le site répond.
# Usage : ./deploy.sh (depuis Git Bash / WSL, avec la clé SSH de déploiement
# déjà générée — voir la clé utilisée pendant la session de mise en prod).
set -euo pipefail

SERVER="ubuntu@141.95.29.41"
SSH_KEY="${DEPLOY_SSH_KEY:-$HOME/.ssh/beintheflow_deploy}"
REMOTE_DIR="~/BeInTheFlow/Son_spatialisation"
HEALTH_URL="https://beintheflow.site/api/hrtfs"

ssh_run() {
  ssh -i "$SSH_KEY" -o BatchMode=yes "$SERVER" "$@"
}

echo "==> Vérification qu'il n'y a rien d'uncommitted en local"
if [ -n "$(git status --porcelain)" ]; then
  echo "ERREUR : des changements locaux ne sont pas commités. Commit + push d'abord." >&2
  git status --short
  exit 1
fi

echo "==> Push de la branche courante vers GitHub"
git push origin main

echo "==> Synchronisation du serveur sur origin/main (fetch + reset --hard : le serveur est un miroir de déploiement, jamais une source de vérité)"
ssh_run "cd $REMOTE_DIR && git fetch origin && git reset --hard origin/main"

echo "==> Rebuild + redémarrage des conteneurs modifiés"
ssh_run "cd $REMOTE_DIR && docker compose up -d --build"

echo "==> Migrations de base de données (idempotent, sans effet si rien de neuf)"
ssh_run "cd $REMOTE_DIR && source .venv/bin/activate && alembic upgrade head"

echo "==> Vérification de l'état des conteneurs"
ssh_run "cd $REMOTE_DIR && docker compose ps"

echo "==> Healthcheck public"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL")
if [ "$STATUS" != "200" ]; then
  echo "ATTENTION : $HEALTH_URL a répondu $STATUS (attendu 200)." >&2
  exit 1
fi
echo "OK : $HEALTH_URL répond 200. Déploiement terminé."
