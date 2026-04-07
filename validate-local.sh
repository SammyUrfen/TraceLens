#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-.}"

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  echo "FAIL: repo directory not found: ${1:-.}"
  exit 1
fi

cd "$REPO_DIR"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "Running local validation..."

echo "Step 1/6: Docker build"
docker build . >/dev/null

echo "Step 2/6: OpenEnv validate"
openenv validate >/dev/null

echo "Step 3/6: Ensure local server is running"
if curl -fsS "http://127.0.0.1:7860/health" >/dev/null 2>&1; then
  echo "Server already running on :7860"
else
  python -m uvicorn server.app:app --host 127.0.0.1 --port 7860 >/tmp/validate-local-server.log 2>&1 &
  SERVER_PID=$!
  for _ in {1..40}; do
    if curl -fsS "http://127.0.0.1:7860/health" >/dev/null 2>&1; then
      break
    fi
    sleep 0.25
  done
  curl -fsS "http://127.0.0.1:7860/health" >/dev/null
fi

echo "Step 4/6: Endpoint checks"
curl -fsS -X POST "http://127.0.0.1:7860/reset" -H "Content-Type: application/json" -d '{}' >/dev/null
curl -fsS "http://127.0.0.1:7860/state" >/dev/null
curl -fsS "http://127.0.0.1:7860/tasks" >/dev/null

echo "Step 5/6: Oracle inference run"
python inference.py --mode oracle --episodes 1 --base-url "http://127.0.0.1:7860" >/tmp/validate-local-inference.log

echo "Step 6/6: Summary"
echo "PASS: local validation succeeded"
