# Models

How `models.toml` describes the models, how `make models` prepares them, and the flags each one
is served with. The list of models is in the [README](../README.md#models).

## Embedding models

General facts from the model cards.

| Model | Dimension | Longest input (tokens) | Pooling |
|---|---|---|---|
| bge-base-en-v1.5 | 768 | 512 | CLS |
| bge-small-en-v1.5 | 384 | 512 | CLS |
| Qwen3-Embedding-0.6B | 1024 | 32768 | last token |
| gte-modernbert-base | 768 | 8192 | CLS |

The served limit can be lower than the model's: an NPU graph has a static length (`max_length`
below), and a llama.cpp embedding server refuses an input longer than its physical batch.

## Preparation

`make models` compares the tree with `models.toml`.

- An OVMS pull or conversion runs when `graph.pbtxt` and a weight `.bin` are absent.
- A graph whose device, pooling, truncate or `max_length` disagrees with the catalog is reported and left on disk. `make ovms-pull NAME=qwen3-npu REPLACE=1` deletes that one directory and builds it again.
- An Ollama model whose manifest and blobs are already stored is left in place.
- A llama.cpp GGUF is reported. `make llama-up` is what downloads one.

NPU rows stay out of `config.json` when `/dev/accel/accel0` is absent, including when the files are already on disk. A config that names an NPU graph on a machine without the device keeps OVMS from starting. The files remain in `npu/`.

## Catalog

`models.toml` lists the models. An OVMS row is a published OpenVINO repository (`prepare` defaults to `pull`) or an ONNX conversion (`prepare = "convert"`). gte-modernbert has no OpenVINO build; optimum-intel no longer exports ModernBERT, and the repository publishes an ONNX file. bge-small-en-v1.5 has no published OpenVINO build either; pdf-mcp takes its embeddings only from that model.

Flags written into `graph.pbtxt` at pull or convert time:

| Field | What it sets | When a change takes effect |
|---|---|---|
| `device` | GPU and NPU are different directories | pull or convert, then an OVMS restart |
| `pooling` | Qwen3-Embedding is `LAST`. gte-modernbert and bge-small are `CLS`, read from the repository when the field is unset | same |
| `truncate` | The bge-gpu pull and both bge-small conversions set it. OVMS 2026.4.0 ignores it: it passes `max_length` to the tokenizer without turning truncation on, so an over-long input is HTTP 400 on every graph, and `make models` warns. The caller keeps each input within the model's length | same |
| `max_length` | NPU static length. Qwen3 is 2048. gte on the NPU is 256: OVMS 2026.4 compiles a non-Qwen embedding model there only below 1024, and every input is then padded to that length, which suits queries | `REPLACE=1` when the graph is already on disk |
| `onnx`, `revision` | Which file of a Hugging Face repository becomes the IR | convert |
| OpenVINO pin in `scripts/convert.py` | The same release as `OVMS_IMAGE`. A conversion is refused when they differ | bump the pin and the image together |

Ollama's pull parameter is the model name (`qwen3-embedding:0.6b`). `OLLAMA_NUM_PARALLEL` and `OLLAMA_CONTEXT_LENGTH` belong to the container; embeddings truncate at the context length with no error. A change to either is a new Ollama container. `OLLAMA_IGPU_ENABLE=1` is set because discovery drops an integrated GPU otherwise. The image exposes Mesa's Intel driver and hides llvmpipe.

llama.cpp serves one GGUF per container. `make llama-up NAME=qwen3-4b` recreates that container so a changed flag is the one that runs. `BACKEND=vulkan` selects `server-vulkan` and, when that is not the row's own backend, a second container name, so the SYCL server can stay up. `LLAMA_HF`, `LLAMA_PORT` and `LLAMA_ROLE` override the row for that run. A name absent from the catalog needs all three.

`-c` is shared across the slots. An embedding row sets `-ub` to at least the longest input, because llama.cpp refuses a longer one. A completion row sets `-n` as the cap on a generation. `extra` is the flags that belong to one checkpoint; Qwen3-8B carries `--reasoning-budget 0`, and the 4B 2507 instruct preset has none.

The llama directory holds either layout, including both at once:

- Hugging Face hub: `models--<owner>--<repo>/snapshots/<rev>/*.gguf`
- legacy llama.cpp cache: `manifest=<owner>_<repo>=<quant>.json` beside a `.gguf` whose name contains the repository and the quant

`LLAMA_CACHE` and `HF_HUB_CACHE` point at that directory, and the same directory is mounted at the legacy `/root/.cache/llama.cpp` path.

## Enabled models

`models.enabled.toml` sets each name in `models.toml` to `true` or `false`. `make enabled-models-config` copies it to `models.enabled.local.toml`, which is git-ignored and read instead when present; a name the local file lacks keeps the default's value, and a name `models.toml` does not have is an error.

A disabled model is not prepared by `make models`, published to the OVMS config or reported missing. `make ovms-pull NAME=` still builds one, and `make llama-up NAME=` still starts one. Disabled by default: `qwen3-gpu`, `qwen3-npu`, `bge-small-gpu`, `bge-small-npu`, and the llama.cpp presets Qwen3-8B, Qwen2.5-7B, Gemma3-4B, Phi-4-mini and the Qwen2.5-Coder 1.5B / 3B / 7B GGUFs.
