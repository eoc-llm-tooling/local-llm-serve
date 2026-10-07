# local-llm-serve

Starts OVMS, Ollama (Intel Vulkan) and llama.cpp, and prepares the models those servers load.

## Servers

| Service | Image | Published address | Weights |
|---|---|---|---|
| OVMS | `openvino/model_server:2026.4.0-gpu` | `127.0.0.1:11551` | `~/.local-llm-serve/ovms` |
| Ollama | `ollama/ollama:0.34.3` plus Mesa's Intel Vulkan driver | `127.0.0.1:11439` | `~/.local-llm-serve/ollama` |
| llama.cpp | `ghcr.io/ggml-org/llama.cpp:server-intel` (SYCL), or `server-vulkan` | one port per model; Qwen3-4B Instruct is `11441` | `~/.local-llm-serve/llama` |

`make up` starts OVMS and, when that GGUF is already in the llama directory, the default llama.cpp server: Qwen3-4B-Instruct-2507 at Q4_K_M on SYCL. That checkpoint is the smallest instruct model in the catalog, and it has no reasoning pass. Ollama starts with `make ollama-up`.

`BIND` (default `127.0.0.1`) is the published address. `BIND=0.0.0.0` publishes on every interface. Each process binds all interfaces inside its container, which is the address the docker proxy connects to.

One stack runs at a time: two OVMS processes on one GPU share a compile cache.

## Models

| Name | Server | Device | Role | Model |
|---|---|---|---|---|
| `bge-gpu`, `bge-npu` | OVMS | GPU, NPU | embedding | bge-base-en-v1.5, OpenVINO fp16 |
| `bge-small-gpu`, `bge-small-npu` | OVMS | GPU, NPU | embedding | bge-small-en-v1.5, converted from ONNX |
| `qwen3-gpu`, `qwen3-npu` | OVMS | GPU, NPU | embedding | Qwen3-Embedding-0.6B, OpenVINO int8 |
| `gte-gpu`, `gte-npu` | OVMS | GPU, NPU | embedding | gte-modernbert-base, converted from ONNX |
| `qwen3-embedding:0.6b` | Ollama | GPU (Vulkan) | embedding | Qwen3-Embedding-0.6B |
| `qwen3-4b` | llama.cpp | GPU (SYCL) | completion, started by `make up` | Qwen3-4B-Instruct-2507, Q4_K_M |
| `qwen3-embed` | llama.cpp | GPU (SYCL) | embedding | Qwen3-Embedding-0.6B, Q8_0 |

On demand with `make llama-up NAME=…`, all llama.cpp completion models: `qwen3-8b`, `qwen2.5-7b`,
`gemma3-4b`, `phi4-mini`, `coder-1.5b`, `coder-3b`, `coder-7b`.

The catalog, the flags each model is prepared with and the weight layouts are in
[docs/models.md](docs/models.md).

## Where weights live

`MODELS_DIR` defaults to `~/.local-llm-serve`, with `ovms/`, `ollama/` and `llama/` under it. `OVMS_DIR`, `OLLAMA_DIR` or `LLAMA_DIR` replaces that one service's directory. A path in `.env` is absolute (`~` is not expanded by Make). The OVMS directory is the tree the server mounts: `gpu/`, `npu/`, `cache/` and `config.json`.

## Run

```bash
cp .env.example .env          # optional; the Makefile defaults bind loopback and ~/.local-llm-serve
make devices
make models                   # leaves a directory that already holds the model
make up                       # OVMS, plus Qwen3-4B Instruct when its GGUF is present
make show
make ollama-up                # when that server is wanted
make llama-up NAME=qwen3-embed
make down
```

`make ready` waits until each name in the OVMS config answers an embedding request. `/v3/models` returns before a graph has compiled. The first start of a llama.cpp model that is already on disk still loads it before `/health` succeeds.

`make lint` runs ruff, shellcheck and the unit tests.
