# Architecture

How the parts of the repository fit together. What each server does and how to run it is in the
[README](README.md); the catalog and the flags each model is served with are in
[docs/models.md](docs/models.md).

## Layout

| Path | Role |
|---|---|
| `models.toml` | the catalog: every model a server loads, with its server, device and per-model flags |
| `models.enabled.toml` | which catalog models are enabled; a git-ignored `models.enabled.local.toml` is read instead when present |
| `Makefile` | the entry point; every operation is a target, and its defaults are overridden by `.env` |
| `.env.example` | the settings a machine overrides: ports, `BIND`, image tags, model directories |
| `compose.yaml` | the OVMS and Ollama services |
| `compose.npu.yaml` | replaces the OVMS device list with GPU and NPU; included only when `/dev/accel/accel0` exists |
| `compose.llama.yaml` | one llama.cpp server, instantiated per catalog row |
| `Dockerfile.ollama` | the Ollama image with Mesa's Intel Vulkan driver added |
| `scripts/catalog.py` | compares the model directories with the catalog; pulls or converts what is missing, writes the OVMS `config.json`, reports GGUFs |
| `scripts/convert.py` | turns a Hugging Face ONNX export into an OVMS embeddings model, pinned to the OpenVINO release of the OVMS image |
| `scripts/llama-up.sh` | starts the llama.cpp server for one catalog row |
| `scripts/wait-ready.sh` | blocks until every OVMS model answers an embedding request |
| `scripts/wait-llama.sh` | blocks until a llama.cpp server answers `/health` |
| `tests/` | unit tests for the catalog logic |

## Flow

`make models` runs `scripts/catalog.py apply`. It reads `models.toml` and the enabled models,
inspects the directories under `MODELS_DIR`, and acts only where an enabled row's directory does
not already match it: an OVMS
model is pulled or converted, an Ollama model is pulled, a GGUF is reported. It then writes the
OVMS `config.json`, leaving out NPU rows when the device is absent.

`make up` starts OVMS from that `config.json`, starts the default llama.cpp server when its GGUF
is on disk, and waits until every OVMS model answers. Ollama and further llama.cpp servers are
started by their own targets.

## Configuration

A setting has one default, held in the `Makefile` and repeated in the compose files, and `.env`
overrides it. Model identity and per-model flags belong to `models.toml`, not to the `Makefile`.
Flags that OVMS reads from `graph.pbtxt` are fixed when a model is pulled or converted, so a
change to them takes effect only when the model directory is rebuilt.
