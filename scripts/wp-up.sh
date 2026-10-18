#!/usr/bin/env bash
# Start the local WordPress and write its connection settings into .env.
# Usage: scripts/wp-up.sh   (WP_PORT=8088 scripts/wp-up.sh if 8080 is taken)
set -euo pipefail
cd "$(dirname "$0")/.."

port="${WP_PORT:-8080}"
user="${WP_ADMIN_USER:-admin}"

docker compose up -d --wait db wordpress
password="$(docker compose run --rm -T setup | tail -n 1 | tr -d '\r')"
if [ -z "$password" ]; then
  echo "setup did not return an application password" >&2
  exit 1
fi

[ -f .env ] || cp .env.example .env
# Replace only the WordPress connection lines; everything else in .env is kept.
grep -v -E '^(WP_BASE_URL|WP_USERNAME|WP_APP_PASSWORD)=' .env > .env.tmp || true
{
  echo "WP_BASE_URL=http://localhost:${port}"
  echo "WP_USERNAME=${user}"
  echo "WP_APP_PASSWORD=${password}"
} >> .env.tmp
mv .env.tmp .env

echo "WordPress is up at http://localhost:${port}/wp-admin (user ${user}); .env updated."
