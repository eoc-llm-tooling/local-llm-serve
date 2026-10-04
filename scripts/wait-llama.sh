#!/usr/bin/env bash
# Block until llama-server answers /health. A dead container ends the wait.
set -euo pipefail
container=${1:?}
bind=${2:?}
port=${3:?}
host=$bind
if [[ $host == 0.0.0.0 ]]; then
  host=127.0.0.1
fi
url="http://${host}:${port}/health"
echo "waiting for ${container}"
until curl -sf -o /dev/null "$url"; do
  running=$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)
  if [[ $running != true ]]; then
    docker logs --tail 40 "$container" || true
    echo "llama-server exited" >&2
    exit 1
  fi
  sleep 2
done
echo "ready ${url}"
