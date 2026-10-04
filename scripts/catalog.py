"""Read models.toml and decide which model directories already hold a served model.

`apply` pulls, converts or rewrites config only for an OVMS entry whose directory does not
already match the catalog. A llama GGUF is reported; `make llama-up` is what downloads one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "models.toml"
CONVERT_PATH = ROOT / "scripts" / "convert.py"
OVMS_CONTAINER = "local-llm-serve-ovms"

LLAMA_IMAGES = {
    "sycl": "ghcr.io/ggml-org/llama.cpp:server-intel",
    "vulkan": "ghcr.io/ggml-org/llama.cpp:server-vulkan",
}
NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def models_dir() -> Path:
    return Path(os.environ.get("MODELS_DIR", Path.home() / ".local-llm-serve")).expanduser()


def service_dir(kind: str) -> Path:
    override = os.environ.get(f"{kind.upper()}_DIR")
    path = Path(override).expanduser() if override else models_dir() / kind
    return path.resolve()


def accel_present() -> bool:
    return Path(os.environ.get("ACCEL_NODE", "/dev/accel/accel0")).exists()


def load_catalog(path: Path | None = None) -> dict:
    path = path or CATALOG_PATH
    with path.open("rb") as handle:
        catalog = tomllib.load(handle)
    problems = validate(catalog)
    if problems:
        raise SystemExit("models.toml:\n  " + "\n  ".join(problems))
    return catalog


def validate(catalog: dict) -> list[str]:
    problems: list[str] = []
    names: set[str] = set()
    defaults = 0
    for entry in catalog.get("ovms", []):
        name = str(entry.get("name", ""))
        if not name or not NAME_RE.match(name):
            problems.append(f"ovms name {name!r} is empty or not a single token")
        elif name in names:
            problems.append(f"duplicate name {name}")
        names.add(name)
        if not entry.get("source"):
            problems.append(f"{name or '?'} has no source")
        if entry.get("device") not in {"GPU", "NPU"}:
            problems.append(f"{name}: device is GPU or NPU")
        if entry.get("prepare", "pull") not in {"pull", "convert"}:
            problems.append(f"{name}: prepare is pull or convert")
    for entry in catalog.get("ollama", []):
        if not entry.get("name"):
            problems.append("an ollama entry has no name")
    for entry in catalog.get("llama", []):
        name = str(entry.get("name", ""))
        if not name or not NAME_RE.match(name):
            problems.append(f"llama name {name!r} is empty or not a single token")
        elif name in names:
            problems.append(f"duplicate name {name}")
        names.add(name)
        if entry.get("default"):
            defaults += 1
        if entry.get("role", "completion") not in {"completion", "embedding"}:
            problems.append(f"{name}: role is completion or embedding")
        if entry.get("backend", "sycl") not in LLAMA_IMAGES:
            problems.append(f"{name}: backend is sycl or vulkan")
        if not entry.get("hf"):
            problems.append(f"{name or '?'}: llama entry has no hf")
        port = entry.get("port")
        if port is not None and not isinstance(port, int):
            problems.append(f"{name}: port is an integer")
    if defaults != 1:
        problems.append(f"exactly one llama entry is default = true ({defaults} found)")
    return problems


def enabled(entry: dict) -> bool:
    return entry.get("enabled", True)


def ovms_rel(entry: dict) -> str:
    device = entry["device"].lower()
    source = entry["source"]
    if entry.get("prepare", "pull") == "convert":
        leaf = entry.get("out") or source.rsplit("/", 1)[-1]
        return f"{device}/converted/{leaf}"
    return f"{device}/{source}"


def graph_fields(text: str) -> dict[str, str]:
    # OVMS writes protobuf text (`pooling: LAST,`). The comma is a separator.
    found: dict[str, str] = {}
    for key in ("target_device", "pooling", "max_length", "truncate"):
        match = re.search(rf'^\s*{key}:\s*"?([^",\s]+)"?', text, re.M)
        if match:
            found[key] = match.group(1)
    return found


def has_weights(directory: Path) -> bool:
    return any(path.suffix == ".bin" and path.is_file() for path in directory.rglob("*.bin"))


def graph_diffs(entry: dict, text: str) -> list[str]:
    fields = graph_fields(text)
    diffs: list[str] = []
    if fields.get("target_device") != entry["device"]:
        diffs.append(f"device graph={fields.get('target_device')} catalog={entry['device']}")
    if entry.get("pooling") and fields.get("pooling") != entry["pooling"]:
        diffs.append(f"pooling graph={fields.get('pooling')} catalog={entry['pooling']}")
    if entry.get("max_length") and fields.get("max_length") != str(entry["max_length"]):
        diffs.append(f"max_length graph={fields.get('max_length')} catalog={entry['max_length']}")
    if entry.get("truncate") and fields.get("truncate") != "true":
        diffs.append(f"truncate graph={fields.get('truncate')} catalog=true")
    return diffs


def ovms_action(entry: dict, root: Path, accel: bool) -> dict:
    """One OVMS catalog row against the files currently in root."""
    rel = ovms_rel(entry)
    directory = root / rel
    graph = directory / "graph.pbtxt"
    base = {"name": entry["name"], "rel": rel, "prepare": entry.get("prepare", "pull")}
    npu = entry["device"] == "NPU"
    if npu and not accel:
        kind = "kept" if graph.is_file() else "skip"
        reason = "no /dev/accel/accel0"
        detail = f"{reason}; left on disk, omitted from config" if kind == "kept" else reason
        return {**base, "kind": kind, "detail": detail}
    if not graph.is_file():
        return {**base, "kind": "missing", "detail": str(directory)}
    if not has_weights(directory):
        return {**base, "kind": "incomplete", "detail": f"{directory} has graph.pbtxt and no .bin"}
    diffs = graph_diffs(entry, graph.read_text())
    if diffs:
        return {**base, "kind": "stale", "detail": "; ".join(diffs)}
    return {**base, "kind": "present", "detail": str(directory)}


def ovms_actions(catalog: dict, root: Path, accel: bool) -> list[dict]:
    return [ovms_action(entry, root, accel) for entry in catalog.get("ovms", []) if enabled(entry)]


def served_actions(actions: list[dict]) -> list[dict]:
    """Rows whose graph is loaded. A stale graph still serves; a missing NPU does not."""
    return [action for action in actions if action["kind"] in {"present", "stale", "incomplete"}]


def config_names(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    cfg = json.loads(path.read_text())
    names = {item["config"]["name"] for item in cfg.get("model_config_list", [])}
    names.update(item["name"] for item in cfg.get("mediapipe_config_list", []))
    return names


def split_hf(spec: str) -> tuple[str, str, str]:
    repo, sep, quant = spec.rpartition(":")
    if not sep or "/" not in repo:
        repo, quant = spec, ""
    owner, _, name = repo.partition("/")
    return owner, name, quant


def quant_in_filename(filename: str, quant: str) -> bool:
    if not quant:
        return True
    stem = filename.lower().removesuffix(".gguf")
    return quant.lower() in set(re.split(r"-+", stem))


def repo_needle(name: str) -> str:
    lowered = name.lower()
    return lowered.removesuffix("-gguf") if lowered.endswith("-gguf") else lowered


def _ggufs(root: Path):
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".gguf":
            continue
        if path.stat().st_size <= 0 or ".incomplete" in path.name or path.name.endswith(".partial"):
            continue
        yield path


def llama_present(cache: Path, hf: str) -> bool:
    """True when cache holds this GGUF, in hub layout or the legacy flat cache."""
    if not cache.is_dir():
        return False
    owner, name, quant = split_hf(hf)
    folder = cache / f"models--{owner}--{name.replace('/', '--')}"
    legacy = cache / f"manifest={owner}_{name}={quant}.json" if quant else None
    needle = repo_needle(name)
    matched_file = False
    for path in _ggufs(cache):
        under_repo = folder in path.parents
        stem = path.name.lower()
        quant_ok = quant_in_filename(path.name, quant)
        if under_repo and quant_ok:
            return True
        if quant_ok and needle in stem.removesuffix(".gguf"):
            matched_file = True
    if legacy is not None and legacy.is_file() and matched_file:
        return True
    return matched_file


def ollama_manifest(root: Path, name: str) -> Path:
    repo, sep, tag = name.rpartition(":")
    if not sep:
        repo, tag = name, "latest"
    if "/" not in repo:
        repo = f"library/{repo}"
    return root / "models" / "manifests" / "registry.ollama.ai" / repo / tag


def ollama_present(root: Path, name: str) -> bool:
    blobs = root / "models" / "blobs"
    return ollama_manifest(root, name).is_file() and blobs.is_dir() and any(blobs.iterdir())


def llama_default(catalog: dict) -> dict:
    return next(entry for entry in catalog["llama"] if entry.get("default"))


def llama_by_name(catalog: dict, name: str) -> dict | None:
    return next((entry for entry in catalog.get("llama", []) if entry["name"] == name), None)


def llama_image(backend: str) -> str:
    env = {"sycl": "LLAMA_IMAGE_SYCL", "vulkan": "LLAMA_IMAGE_VULKAN"}[backend]
    return os.environ.get(env) or LLAMA_IMAGES[backend]


def llama_resolved(catalog: dict, name: str) -> dict:
    """Catalog row plus LLAMA_HF, LLAMA_PORT, LLAMA_ROLE and BACKEND from the environment."""
    entry = llama_by_name(catalog, name)
    if entry is None:
        hf = os.environ.get("LLAMA_HF", "")
        role = os.environ.get("LLAMA_ROLE", "")
        port = os.environ.get("LLAMA_PORT", "")
        if not (hf and role and port):
            known = ", ".join(item["name"] for item in catalog.get("llama", []))
            raise SystemExit(
                f"unknown llama {name}; known: {known} (or set LLAMA_HF, LLAMA_ROLE, LLAMA_PORT)"
            )
        entry = {"name": name, "hf": hf, "role": role, "port": int(port), "backend": "sycl"}
    backend = os.environ.get("BACKEND") or entry.get("backend", "sycl")
    if backend not in LLAMA_IMAGES:
        raise SystemExit(f"BACKEND is sycl or vulkan, got {backend}")
    role = os.environ.get("LLAMA_ROLE") or entry.get("role", "completion")
    hf = os.environ.get("LLAMA_HF") or entry["hf"]
    port = int(os.environ.get("LLAMA_PORT") or entry.get("port") or 0)
    if port <= 0:
        raise SystemExit(f"{name}: set port in models.toml or LLAMA_PORT")
    suffix = ""
    if os.environ.get("BACKEND") and os.environ["BACKEND"] != entry.get("backend", "sycl"):
        suffix = f"-{backend}"
    container = f"local-llm-serve-llama-{name}{suffix}"
    return {
        "name": name,
        "hf": hf,
        "role": role,
        "backend": backend,
        "image": llama_image(backend),
        "port": port,
        "parallel": int(entry.get("parallel", 4)),
        "ctx": int(entry.get("ctx", 16384)),
        "ubatch": entry.get("ubatch"),
        "pooling": entry.get("pooling", "last"),
        "max_tokens": int(entry.get("max_tokens", 512)),
        "extra": list(entry.get("extra") or []),
        "gpu_layers": int(entry.get("gpu_layers", 99)),
        "container": container,
        "project": container,
    }


def llama_command(spec: dict) -> list[str]:
    args = ["-hf", spec["hf"], "-ngl", str(spec["gpu_layers"])]
    if spec["role"] == "embedding":
        args += ["--embedding", "--pooling", spec["pooling"]]
    else:
        args += ["--jinja", "-n", str(spec["max_tokens"])]
    args += ["-np", str(spec["parallel"]), "-c", str(spec["ctx"])]
    if spec.get("ubatch"):
        args += ["-b", str(spec["ubatch"]), "-ub", str(spec["ubatch"])]
    args += list(spec["extra"])
    # 8080 is the port inside the container. The catalog port is the published one.
    # The process binds every interface so the docker proxy on BIND can reach it.
    args += ["--host", "0.0.0.0", "--port", "8080"]
    return args


def openvino_pin() -> str:
    match = re.search(r"openvino==([0-9.]+)", CONVERT_PATH.read_text())
    if not match:
        raise SystemExit(f"no openvino pin in {CONVERT_PATH}")
    return match.group(1)


def image_release(image: str) -> str:
    match = re.search(r":(\d+\.\d+\.\d+)", image)
    return match.group(1) if match else ""


def check_openvino_pin() -> None:
    image = os.environ.get("OVMS_IMAGE", "openvino/model_server:2026.4.0-gpu")
    pin = openvino_pin()
    release = image_release(image)
    if release != pin:
        raise SystemExit(
            f"OVMS_IMAGE {image} is release {release or '?'}; "
            f"scripts/convert.py pins openvino=={pin}"
        )


def pull_flags(entry: dict) -> list[str]:
    flags = ["--target_device", entry["device"]]
    if entry.get("pooling"):
        flags += ["--pooling", entry["pooling"]]
    if entry.get("truncate"):
        flags += ["--truncate", "true"]
    if entry.get("max_length"):
        flags += ["--max_length", str(entry["max_length"])]
    return flags


def compose_base() -> list[str]:
    command = ["docker", "compose", "-f", str(ROOT / "compose.yaml")]
    if accel_present():
        command += ["-f", str(ROOT / "compose.npu.yaml")]
    return command


def ensure_runtime_env() -> None:
    os.environ.setdefault("HOST_UID", str(os.getuid()))
    os.environ.setdefault("HOST_GID", str(os.getgid()))
    os.environ.setdefault("BIND", "127.0.0.1")
    os.environ["OVMS_DIR"] = str(service_dir("ovms"))
    os.environ["OLLAMA_DIR"] = str(service_dir("ollama"))
    os.environ["LLAMA_DIR"] = str(service_dir("llama"))
    render = os.environ.get("RENDER_GID")
    if not render:
        node = next(Path("/dev/dri").glob("renderD*"), None) if Path("/dev/dri").is_dir() else None
        if node is not None:
            os.environ["RENDER_GID"] = str(node.stat().st_gid)
    if accel_present() and not os.environ.get("ACCEL_GID"):
        gid = Path("/dev/accel/accel0").stat().st_gid
        render_gid = os.environ.get("RENDER_GID")
        # compose rejects the same group twice; the user's group stands in when they match.
        os.environ["ACCEL_GID"] = os.environ["HOST_GID"] if str(gid) == render_gid else str(gid)


def run(command: list[str]) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def safe_rmtree(root: Path, target: Path) -> None:
    target = target.resolve()
    root = root.resolve()
    if root != target and root not in target.parents:
        raise SystemExit(f"refuse to delete {target}")
    shutil.rmtree(target)


def ovms_entry(catalog: dict, name: str) -> dict:
    entry = next((item for item in catalog.get("ovms", []) if item["name"] == name), None)
    if entry is None:
        known = ", ".join(item["name"] for item in catalog.get("ovms", []))
        raise SystemExit(f"unknown ovms model {name}; known: {known}")
    return entry


def apply_ovms(catalog: dict, only: str | None, replace: bool) -> None:
    ensure_runtime_env()
    root = service_dir("ovms")
    (root / "gpu").mkdir(parents=True, exist_ok=True)
    (root / "npu").mkdir(parents=True, exist_ok=True)
    (root / "cache").mkdir(parents=True, exist_ok=True)
    actions = ovms_actions(catalog, root, accel_present())
    if only:
        actions = [action for action in actions if action["name"] == only]
        if not actions:
            ovms_entry(catalog, only)
            raise SystemExit(f"{only} is disabled or not an ovms model")
    rebuild = {action["name"] for action in actions} if replace else set()
    if replace and not only:
        raise SystemExit("REPLACE=1 needs NAME=, so one model directory is deleted")
    needs_convert = False
    for action in actions:
        print(f"ovms {action['name']}: {action['kind']} ({action['detail']})")
        due = action["kind"] == "missing" or action["name"] in rebuild
        if due and action["prepare"] == "convert":
            needs_convert = True
    if needs_convert:
        check_openvino_pin()
    for action in actions:
        if action["kind"] not in {"missing", "stale", "present", "incomplete"}:
            continue
        if action["kind"] != "missing" and action["name"] not in rebuild:
            continue
        entry = ovms_entry(catalog, action["name"])
        directory = root / action["rel"]
        if action["name"] in rebuild and directory.exists():
            safe_rmtree(root, directory)
        if action["prepare"] == "convert":
            command = [
                "uv",
                "run",
                "--script",
                str(CONVERT_PATH),
                "--source",
                entry["source"],
                "--out",
                str(directory),
                "--device",
                entry["device"],
            ]
            if entry.get("onnx"):
                command += ["--onnx", entry["onnx"]]
            if entry.get("pooling"):
                command += ["--pooling", entry["pooling"]]
            if entry.get("revision"):
                command += ["--revision", entry["revision"]]
            if entry.get("max_length"):
                command += ["--max-length", str(entry["max_length"])]
            run(command)
        else:
            subdir = entry["device"].lower()
            run(
                [
                    *compose_base(),
                    "run",
                    "--rm",
                    "--no-deps",
                    "-T",
                    "ovms",
                    "--pull",
                    "--model_repository_path",
                    f"/b/{subdir}",
                    "--source_model",
                    entry["source"],
                    "--task",
                    "embeddings",
                    *pull_flags(entry),
                ]
            )
    # Republish every servable graph. A one-model pull would otherwise drop the others.
    publish = served_actions(ovms_actions(catalog, root, accel_present()))
    current = config_names(root / "config.json")
    expected = {action["name"] for action in publish}
    if current == expected and (root / "config.json").is_file():
        print("ovms config.json: already lists the servable models")
        return
    config = root / "config.json"
    if config.exists():
        config.unlink()
    for action in publish:
        run(
            [
                *compose_base(),
                "run",
                "--rm",
                "--no-deps",
                "-T",
                "ovms",
                "--add_to_config",
                "--config_path",
                "/b/config.json",
                "--model_name",
                action["name"],
                "--model_path",
                f"/b/{action['rel']}",
            ]
        )
    print("ovms config.json:", ", ".join(action["name"] for action in publish) or "(empty)")
    probe = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", OVMS_CONTAINER],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.stdout.strip() == "true":
        run([*compose_base(), "restart", "ovms"])


def apply_ollama(catalog: dict) -> None:
    ensure_runtime_env()
    root = service_dir("ollama")
    root.mkdir(parents=True, exist_ok=True)
    missing = []
    for entry in catalog.get("ollama", []):
        if not enabled(entry):
            continue
        if ollama_present(root, entry["name"]):
            print(f"ollama {entry['name']}: present")
        else:
            print(f"ollama {entry['name']}: missing")
            missing.append(entry["name"])
    for name in missing:
        run(
            [
                *compose_base(),
                "run",
                "--build",
                "--rm",
                "--no-deps",
                "ollama",
                "ollama",
                "pull",
                name,
            ]
        )


def report_llama(catalog: dict) -> None:
    cache = service_dir("llama")
    for entry in catalog.get("llama", []):
        if not enabled(entry):
            continue
        state = "present" if llama_present(cache, entry["hf"]) else "missing"
        mark = " (make up)" if entry.get("default") else ""
        print(f"llama {entry['name']}: {state}{mark}  {entry['hf']}")


def apply_all(catalog: dict) -> None:
    apply_ovms(catalog, only=None, replace=False)
    apply_ollama(catalog)
    report_llama(catalog)


def write_llama_files(catalog: dict, name: str) -> Path:
    if not NAME_RE.match(name):
        raise SystemExit(f"name {name!r} is not a single token")
    spec = llama_resolved(catalog, name)
    run_dir = ROOT / ".run"
    run_dir.mkdir(exist_ok=True)
    env_path = run_dir / f"{name}.env"
    yml_path = run_dir / f"{name}.yml"
    bind = os.environ.get("BIND", "127.0.0.1")
    llama_dir = str(service_dir("llama"))
    lines = [
        f"LLAMA_IMAGE={shlex.quote(spec['image'])}",
        f"LLAMA_PORT={spec['port']}",
        f"LLAMA_CONTAINER={shlex.quote(spec['container'])}",
        f"LLAMA_PROJECT={shlex.quote(spec['project'])}",
        f"BIND={shlex.quote(bind)}",
        f"LLAMA_DIR={shlex.quote(llama_dir)}",
    ]
    env_path.write_text("\n".join(lines) + "\n")
    body = ["services:", "  llama:", "    command:"]
    body += [f"      - {json.dumps(arg)}" for arg in llama_command(spec)]
    yml_path.write_text("\n".join(body) + "\n")
    print(env_path)
    return env_path


def cmd_show(catalog: dict) -> None:
    bind = os.environ.get("BIND", "127.0.0.1")
    host = "127.0.0.1" if bind == "0.0.0.0" else bind
    ovms_port = os.environ.get("OVMS_PORT", "11551")
    ollama_port = os.environ.get("OLLAMA_PORT", "11439")
    print(f"bind    {bind}")
    print(f"ovms    http://{host}:{ovms_port}/v3/embeddings")
    print(f"        {service_dir('ovms')}")
    print(f"ollama  http://{host}:{ollama_port}/v1/embeddings")
    print(f"        {service_dir('ollama')}")
    print(f"llama   {service_dir('llama')}")
    accel = accel_present()
    print(f"npu     {'/dev/accel/accel0' if accel else 'absent'}")
    for action in ovms_actions(catalog, service_dir("ovms"), accel):
        print(f"  ovms {action['name']}: {action['kind']}")
    for entry in catalog.get("ollama", []):
        if not enabled(entry):
            continue
        state = "present" if ollama_present(service_dir("ollama"), entry["name"]) else "missing"
        print(f"  ollama {entry['name']}: {state}")
    cache = service_dir("llama")
    for entry in catalog.get("llama", []):
        if not enabled(entry) and not entry.get("default"):
            continue
        state = "present" if llama_present(cache, entry["hf"]) else "missing"
        default = " default" if entry.get("default") else ""
        print(f"  llama {entry['name']}:{default} {state}  :{entry.get('port')}  {entry['hf']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command")
    parser.add_argument("name", nargs="?")
    parser.add_argument("--name")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()
    catalog = load_catalog(args.catalog)

    if args.command == "show":
        cmd_show(catalog)
    elif args.command == "list":
        for entry in catalog.get("ovms", []):
            print(f"ovms    {entry['name']}")
        for entry in catalog.get("ollama", []):
            print(f"ollama  {entry['name']}")
        for entry in catalog.get("llama", []):
            flag = " default" if entry.get("default") else ""
            off = "" if enabled(entry) else " disabled"
            print(f"llama   {entry['name']}{flag}{off}  {entry['hf']}")
    elif args.command == "llama-default":
        print(llama_default(catalog)["name"])
    elif args.command == "llama-present":
        if not args.name:
            raise SystemExit("llama-present NAME")
        entry = llama_by_name(catalog, args.name)
        hf = os.environ.get("LLAMA_HF") or (entry or {}).get("hf")
        if not hf:
            raise SystemExit(f"unknown llama {args.name}")
        sys.exit(0 if llama_present(service_dir("llama"), hf) else 1)
    elif args.command == "llama-files":
        if not args.name:
            raise SystemExit("llama-files NAME")
        ensure_runtime_env()
        write_llama_files(catalog, args.name)
    elif args.command == "apply":
        apply_all(catalog)
    elif args.command == "apply-ovms":
        apply_ovms(catalog, only=args.name, replace=args.replace)
    elif args.command == "apply-ollama":
        apply_ollama(catalog)
    else:
        raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
