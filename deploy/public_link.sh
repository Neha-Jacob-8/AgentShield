#!/usr/bin/env bash
# Share the demo console from this laptop with a public https link. No account needed.
#
#   deploy/public_link.sh          # starts the console and prints the link; Ctrl+C stops sharing
#
# The link is a Cloudflare quick tunnel (brew install cloudflared) when the network allows it, otherwise
# localhost.run over SSH (campus networks often block Cloudflare's port 7844). It works while this script runs
# and changes on every start.
# The console runs with --public: visitors get their own sessions, cannot read local files and do not see the
# samples built from the local datasets. No audit log is written, so visitors' documents never reach the disk.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=${PORT:-8000}

# the first python with the console's packages: $PYTHON, then python3 on the PATH, then conda's base environment
# (another python3 can come first on the PATH when conda is not activated)
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  CONDA_BASE=$(conda info --base 2>/dev/null || true)
  for c in python3 ${CONDA_BASE:+"$CONDA_BASE/bin/python3"} /opt/anaconda3/bin/python3 /opt/miniconda3/bin/python3 \
           "$HOME/anaconda3/bin/python3" "$HOME/miniconda3/bin/python3"; do
    if command -v "$c" >/dev/null && "$c" -c "import fastapi, uvicorn, torch, transformers" 2>/dev/null; then
      PY=$c
      break
    fi
  done
  if [ -z "$PY" ]; then
    echo "no python3 with the console's packages found: pip install -r requirements.txt, or run PYTHON=/path/to/python3 $0" >&2
    exit 1
  fi
fi
echo "using $(command -v "$PY")"

if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
  echo "port $PORT is in use: stop the running console first, or run PORT=8010 $0" >&2
  exit 1
fi

SERVER="" TUNNEL="" LOG=$(mktemp)
trap 'kill $SERVER $TUNNEL 2>/dev/null; rm -f "$LOG"' EXIT

"$PY" -m src.defense.api --port "$PORT" --public --audit_log "" &
SERVER=$!
echo "loading the models ..."
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
  kill -0 "$SERVER" 2>/dev/null || { echo "the console did not start (with $PY): see the error above" >&2; exit 1; }
  sleep 1
done

if command -v cloudflared >/dev/null && nc -z -G 5 region1.v2.argotunnel.com 7844 2>/dev/null; then
  cloudflared tunnel --no-autoupdate --protocol http2 --url "http://127.0.0.1:$PORT" >"$LOG" 2>&1 &
  PATTERN='https://[a-z0-9-]*\.trycloudflare\.com' READY="Registered tunnel connection"
else
  ssh -T -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
      -R 80:127.0.0.1:"$PORT" nokey@localhost.run >"$LOG" 2>&1 &
  PATTERN='https://[a-z0-9-]*\.lhr\.life' READY="tunneled with tls termination"
fi
TUNNEL=$!
URL=""
until [ -n "$URL" ] && grep -q "$READY" "$LOG"; do          # the link works once the tunnel is connected
  kill -0 "$TUNNEL" 2>/dev/null || { cat "$LOG" >&2; exit 1; }
  sleep 1
  URL=$(grep -o "$PATTERN" "$LOG" | head -1 || true)
done

echo
echo "  Public link:  $URL"
echo "  Ctrl+C stops sharing."
echo
wait "$TUNNEL"
