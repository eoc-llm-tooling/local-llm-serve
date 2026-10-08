#!/usr/bin/env bash
# Start one llama.cpp server named in models.toml.
#   scripts/llama-up.sh [NAME] [--if-present] [--recreate]
# --if-present exits 0 without starting when the GGUF is not in LLAMA_DIR.
set -euo pipefail
cd "$(dirname "$0")/.."

name=
if_present=0
recreate=0
for arg in "$@"; do
  case "$arg" in
    --if-present) if_present=1 ;;
    --recreate) recreate=1 ;;
    *)
      if [[ -n $name ]]; then
        echo "unexpected argument: $arg" >&2
        exit 2
      fi
      name=$arg
      ;;
  esac
done
if [[ -z $name ]]; then
  name=$(python3 scripts/catalog.py llama-default)
fi
if [[ ! $name =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "bad name: $name" >&2
  exit 2
fi

if [[ $if_present -eq 1 ]] && ! python3 scripts/catalog.py llama-present "$name"; then
  echo "$name: no weights in ${LLAMA_DIR:-$HOME/.local-llm-serve/llama} — left stopped"
  exit 0
fi

python3 scripts/catalog.py llama-files "$name"
env_file=".run/${name}.env"
set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

args=(up -d)
if [[ $recreate -eq 1 ]]; then
  args+=(--force-recreate)
fi
docker compose -p "$LLAMA_PROJECT" -f compose.llama.yaml -f ".run/${name}.yml" "${args[@]}"
bash scripts/wait-llama.sh "$LLAMA_CONTAINER" "$BIND" "$LLAMA_PORT"
