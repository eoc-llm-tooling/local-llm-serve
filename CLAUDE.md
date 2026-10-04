# local-llm-serve

- Model identity and per-model flags live in `models.toml`. Add a model there. A served model is not a new Makefile variable.
- Ports, `BIND`, image tags and the model directories come from `.env`. The Makefile and the compose files hold the same defaults; change both.
- The OpenVINO pin in `scripts/convert.py` stays on the same release as `OVMS_IMAGE`.
- `make models` leaves a model directory that already matches the catalog. Rebuilding one OVMS model is `make ovms-pull NAME=… REPLACE=1`.
- Weights live under `MODELS_DIR` (default `~/.local-llm-serve`). Keep weights, `.env` and a machine's copy notes out of this repository.
- `make up` starts OVMS, and the default llama.cpp server only when that GGUF is already on disk. It does not start Ollama.

## Before committing

- Run `make lint` and commit only when it passes.
- Never commit paths from your own machine, IP or MAC addresses, internal host names, tokens or
  keys; use the documented defaults or placeholders.
- Work on a branch and open a pull request; never push to `main`.
