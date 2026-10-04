#!/usr/bin/env bash
# Block until every model in $OVMS_DIR/config.json answers an embedding request.
# OVMS answers /v3/models before a graph has compiled, so only a real request counts.
set -euo pipefail
cd "$(dirname "$0")/.."
dir=${OVMS_DIR:?}
port=${OVMS_PORT:-11551}
bind=${BIND:-127.0.0.1}
host=$bind
if [[ $host == 0.0.0.0 ]]; then
  host=127.0.0.1
fi
url="http://${host}:${port}/v3/embeddings"
config="$dir/config.json"
if [[ ! -f $config ]]; then
  echo "no $config" >&2
  exit 1
fi
# --add_to_config writes model_config_list; a hand-written file may use mediapipe_config_list.
models=$(python3 -c '
import json, sys
cfg = json.load(open(sys.argv[1]))
names = [m["config"]["name"] for m in cfg.get("model_config_list", [])]
names += [m["name"] for m in cfg.get("mediapipe_config_list", [])]
print(*names)
' "$config")
if [[ -z ${models// /} ]]; then
  echo "config.json names no models" >&2
  exit 1
fi
for model in $models; do
  echo "waiting for ${model}"
  until curl -sf -o /dev/null "$url" -H 'Content-Type: application/json' \
      -d "{\"model\":\"${model}\",\"input\":[\"ready\"]}"; do
    sleep 2
  done
  echo "ready ${model}"
done
