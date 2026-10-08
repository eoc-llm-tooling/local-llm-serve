.DEFAULT_GOAL := help
.PHONY: help dirs devices show list models ovms-pull ollama-models \
        up ovms-up ollama-up ready down ovms-down ollama-down \
        llama-up llama-start llama-down llama-list lint

-include .env
export

HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)
RENDER_NODE := $(firstword $(wildcard /dev/dri/renderD*))
RENDER_GID := $(if $(RENDER_NODE),$(shell stat -c %g $(RENDER_NODE)),$(shell getent group render | cut -d: -f3))
ACCEL_GID := $(if $(wildcard /dev/accel/accel0),$(shell stat -c %g /dev/accel/accel0),$(RENDER_GID))
# compose rejects a repeated group_add entry; the user's own group is a harmless stand-in.
ACCEL_GID := $(if $(filter $(ACCEL_GID),$(RENDER_GID)),$(HOST_GID),$(ACCEL_GID))
export HOST_UID HOST_GID RENDER_GID ACCEL_GID

BIND ?= 127.0.0.1
MODELS_DIR ?= $(HOME)/.local-llm-serve
OVMS_DIR ?= $(MODELS_DIR)/ovms
OLLAMA_DIR ?= $(MODELS_DIR)/ollama
LLAMA_DIR ?= $(MODELS_DIR)/llama
OVMS_DIR := $(abspath $(OVMS_DIR))
OLLAMA_DIR := $(abspath $(OLLAMA_DIR))
LLAMA_DIR := $(abspath $(LLAMA_DIR))
export BIND MODELS_DIR OVMS_DIR OLLAMA_DIR LLAMA_DIR

OVMS_IMAGE ?= openvino/model_server:2026.4.0-gpu
OVMS_PORT ?= 11551
OLLAMA_IMAGE ?= ollama/ollama:0.34.3
OLLAMA_PORT ?= 11439
OLLAMA_NUM_PARALLEL ?= 1
OLLAMA_CONTEXT_LENGTH ?= 8192
LLAMA_IMAGE_SYCL ?= ghcr.io/ggml-org/llama.cpp:server-intel
LLAMA_IMAGE_VULKAN ?= ghcr.io/ggml-org/llama.cpp:server-vulkan
export OVMS_IMAGE OVMS_PORT OLLAMA_IMAGE OLLAMA_PORT OLLAMA_NUM_PARALLEL OLLAMA_CONTEXT_LENGTH
export LLAMA_IMAGE_SYCL LLAMA_IMAGE_VULKAN

COMPOSE := -f compose.yaml
ifneq ($(wildcard /dev/accel/accel0),)
COMPOSE += -f compose.npu.yaml
endif

help:
	@echo "Models live in $(MODELS_DIR) (ovms, ollama, llama). make show prints what is on disk."
	@echo "  dirs           create the three model directories"
	@echo "  devices        GPU and NPU nodes the containers need"
	@echo "  show           bind, ports, directories, and whether each enabled model is on disk"
	@echo "  list           every catalog entry"
	@echo "  models         pull or convert what is missing; leave a complete directory in place"
	@echo "  ovms-pull      one OVMS model: NAME=…  REPLACE=1 deletes that directory first"
	@echo "  ollama-models  pull enabled Ollama models that are not already stored"
	@echo "  up             OVMS, and the default llama.cpp server when its GGUF is on disk"
	@echo "  ovms-up        OVMS on $(BIND):$(OVMS_PORT)"
	@echo "  ollama-up      Ollama Vulkan on $(BIND):$(OLLAMA_PORT)"
	@echo "  ready          wait until every OVMS model answers an embedding request"
	@echo "  down           stop OVMS, Ollama and every llama.cpp container"
	@echo "  llama-up       llama.cpp: NAME= (default qwen3-4b) BACKEND=vulkan LLAMA_HF= LLAMA_PORT="
	@echo "  llama-down     stop that container"
	@echo "  llama-list     presets in models.toml"
	@echo "  lint           ruff, shellcheck, unit tests"

dirs:
	mkdir -p "$(OVMS_DIR)/gpu" "$(OVMS_DIR)/npu" "$(OVMS_DIR)/cache" "$(OLLAMA_DIR)" "$(LLAMA_DIR)" .run

devices:
	@ls -l /dev/dri/renderD* /dev/accel/accel0 2>&1 || true
	@for d in $(RENDER_NODE) /dev/accel/accel0; do \
	  if [ ! -e $$d ]; then echo "MISSING $$d"; \
	  elif [ "$$(stat -c %a $$d | cut -c2)" -lt 6 ]; then echo "NO GROUP rw on $$d (mode $$(stat -c %a $$d))"; \
	  else echo "ok $$d (gid $$(stat -c %g $$d))"; fi; \
	done

show: dirs
	python3 scripts/catalog.py show

list:
	python3 scripts/catalog.py list

models: dirs
	python3 scripts/catalog.py apply

ovms-pull: dirs
	@test -n "$(NAME)" || { echo "NAME is required"; exit 1; }
	python3 scripts/catalog.py apply-ovms --name "$(NAME)" $(if $(REPLACE),--replace)

ollama-models: dirs
	python3 scripts/catalog.py apply-ollama

ovms-up: dirs
	@test -f "$(OVMS_DIR)/config.json" || { echo "no $(OVMS_DIR)/config.json — make models"; exit 1; }
	docker compose $(COMPOSE) up -d ovms

ollama-up: dirs
	docker compose $(COMPOSE) up -d --build ollama

llama-up:
	bash scripts/llama-up.sh $(NAME) --recreate

llama-start:
	bash scripts/llama-up.sh $(NAME) --if-present

llama-down:
	@set -euo pipefail; \
	name="$(NAME)"; \
	if [ -z "$$name" ]; then name=$$(python3 scripts/catalog.py llama-default); fi; \
	python3 scripts/catalog.py llama-files "$$name" >/dev/null; \
	set -a; . .run/$$name.env; set +a; \
	docker compose -p "$$LLAMA_PROJECT" -f compose.llama.yaml down

llama-list: list

up: ovms-up llama-start ready

ready:
	bash scripts/wait-ready.sh

ovms-down:
	docker compose $(COMPOSE) stop ovms

ollama-down:
	docker compose $(COMPOSE) stop ollama

down:
	docker compose $(COMPOSE) down
	@set -euo pipefail; \
	docker ps -aq --filter name=local-llm-serve-llama- | while read -r id; do \
	  proj=$$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' "$$id"); \
	  echo "$$proj"; \
	done | sort -u | while read -r proj; do \
	  [ -n "$$proj" ] && docker compose -p "$$proj" down; \
	done

lint:
	uv run ruff check scripts tests
	uv run ruff format --check scripts tests
	shellcheck scripts/*.sh
	python3 -m unittest discover -s tests -v
