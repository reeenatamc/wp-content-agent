#!/usr/bin/env bash
# One command from a fresh clone to a WordPress draft:
#   ANTHROPIC_API_KEY=sk-ant-... scripts/demo.sh
# Starts the local WordPress, writes .env and runs the example brief. Extra arguments
# go to `wpagent run`, for example --yes to approve without asking.
set -euo pipefail
cd "$(dirname "$0")/.."

for tool in docker uv; do
  command -v "$tool" >/dev/null || { echo "$tool is required" >&2; exit 1; }
done

[ -f .env ] || cp .env.example .env
chmod 600 .env
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  # Keep the key in .env so later wpagent commands find it.
  grep -v -E '^ANTHROPIC_API_KEY=' .env > .env.tmp || true
  echo "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}" >> .env.tmp
  chmod 600 .env.tmp
  mv .env.tmp .env
elif ! grep -q -E '^(ANTHROPIC|OPENAI)_API_KEY=.+' .env; then
  echo "set ANTHROPIC_API_KEY (or put it in .env) and run this again" >&2
  exit 1
fi

uv sync --quiet
scripts/wp-up.sh
uv run wpagent run --brief "${BRIEF:-examples/briefs/django-caching.yaml}" "$@"
