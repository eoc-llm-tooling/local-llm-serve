import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import catalog


def load_convert() -> types.ModuleType:
    # convert.py runs under its own uv script environment; its imports are stubbed here.
    stubs = {
        "openvino": types.ModuleType("openvino"),
        "huggingface_hub": types.ModuleType("huggingface_hub"),
    }
    stubs["huggingface_hub"].snapshot_download = None
    with mock.patch.dict(sys.modules, stubs):
        sys.modules.pop("convert", None)
        return importlib.import_module("convert")


def write_model(root: Path, rel: str, graph: str, weights: bool = True) -> None:
    directory = root / rel
    directory.mkdir(parents=True)
    (directory / "graph.pbtxt").write_text(graph)
    if weights:
        (directory / "openvino_model.bin").write_bytes(b"x" * 16)


GPU_GRAPH = """
target_device: "GPU"
pooling: LAST
normalize_embeddings: true
"""


class OvmsActionTests(unittest.TestCase):
    def test_missing_without_graph(self) -> None:
        entry = {"name": "bge-gpu", "source": "OpenVINO/bge", "device": "GPU", "truncate": True}
        action = catalog.ovms_action(entry, Path("/no/such/ovms"), accel=True)
        self.assertEqual(action["kind"], "missing")
        self.assertEqual(action["rel"], "gpu/OpenVINO/bge")

    def test_present_when_graph_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model(root, "gpu/OpenVINO/bge", GPU_GRAPH + "truncate: true\n")
            entry = {
                "name": "bge-gpu",
                "source": "OpenVINO/bge",
                "device": "GPU",
                "truncate": True,
            }
            action = catalog.ovms_action(entry, root, accel=True)
            self.assertEqual(action["kind"], "present")

    def test_protobuf_trailing_commas_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model(
                root,
                "npu/OpenVINO/qwen",
                'target_device: "NPU",\npooling: LAST,\nmax_length: 2048,\n',
            )
            entry = {
                "name": "qwen3-npu",
                "source": "OpenVINO/qwen",
                "device": "NPU",
                "pooling": "LAST",
                "max_length": 2048,
            }
            action = catalog.ovms_action(entry, root, accel=True)
            self.assertEqual(action["kind"], "present")

    def test_stale_max_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model(
                root,
                "npu/OpenVINO/qwen",
                'target_device: "NPU"\npooling: LAST\nmax_length: 1024\n',
            )
            entry = {
                "name": "qwen3-npu",
                "source": "OpenVINO/qwen",
                "device": "NPU",
                "pooling": "LAST",
                "max_length": 2048,
            }
            action = catalog.ovms_action(entry, root, accel=True)
            self.assertEqual(action["kind"], "stale")
            self.assertIn("max_length", action["detail"])

    def test_incomplete_without_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_model(root, "gpu/OpenVINO/bge", GPU_GRAPH, weights=False)
            entry = {"name": "bge-gpu", "source": "OpenVINO/bge", "device": "GPU"}
            self.assertEqual(catalog.ovms_action(entry, root, accel=True)["kind"], "incomplete")

    def test_npu_omitted_without_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = {"name": "qwen3-npu", "source": "OpenVINO/qwen", "device": "NPU"}
            self.assertEqual(catalog.ovms_action(entry, root, accel=False)["kind"], "skip")
            write_model(root, "npu/OpenVINO/qwen", 'target_device: "NPU"\n')
            kept = catalog.ovms_action(entry, root, accel=False)
            self.assertEqual(kept["kind"], "kept")
            served = catalog.served_actions([kept])
            self.assertEqual(served, [])

    def test_convert_rel(self) -> None:
        entry = {
            "name": "gte-gpu",
            "prepare": "convert",
            "source": "Alibaba-NLP/gte-modernbert-base",
            "device": "GPU",
        }
        self.assertEqual(catalog.ovms_rel(entry), "gpu/converted/gte-modernbert-base")


BGE_SMALL_NPU = {
    "name": "bge-small-npu",
    "prepare": "convert",
    "source": "BAAI/bge-small-en-v1.5",
    "device": "NPU",
    "onnx": "onnx/model.onnx",
    "truncate": True,
    "max_length": 512,
}


class ConvertTests(unittest.TestCase):
    def test_command_truncate(self) -> None:
        command = catalog.convert_command(BGE_SMALL_NPU, Path("/b/npu"))
        self.assertIn("--truncate", command)
        self.assertEqual(command[command.index("--max-length") + 1], "512")
        plain = {key: value for key, value in BGE_SMALL_NPU.items() if key != "truncate"}
        self.assertNotIn("--truncate", catalog.convert_command(plain, Path("/b/npu")))
        self.assertNotIn(
            "--truncate", catalog.convert_command({**plain, "truncate": False}, Path("/b/npu"))
        )

    def test_graph_truncate(self) -> None:
        convert = load_convert()
        cut = catalog.graph_fields(convert.graph("CLS", "NPU", 512, truncate=True))
        self.assertEqual(cut.get("truncate"), "true")
        self.assertEqual(cut.get("max_length"), "512")
        plain = convert.graph("CLS", "GPU", None, truncate=False)
        self.assertNotIn("truncate", plain)
        self.assertNotIn("max_length", plain)

    def test_converted_graph_matches_catalog(self) -> None:
        convert = load_convert()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rel = catalog.ovms_rel(BGE_SMALL_NPU)
            write_model(root, rel, convert.graph("CLS", "NPU", 512, truncate=True))
            action = catalog.ovms_action(BGE_SMALL_NPU, root, accel=True)
            self.assertEqual(action["kind"], "present", action["detail"])
            write_model(root / "plain", rel, convert.graph("CLS", "NPU", 512, truncate=False))
            stale = catalog.ovms_action(BGE_SMALL_NPU, root / "plain", accel=True)
            self.assertEqual(stale["kind"], "stale")
            self.assertIn("truncate", stale["detail"])


class ConfigTests(unittest.TestCase):
    def test_both_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "model_config_list": [{"config": {"name": "bge-gpu"}}],
                        "mediapipe_config_list": [{"name": "qwen3-gpu"}],
                    }
                )
            )
            self.assertEqual(catalog.config_names(path), {"bge-gpu", "qwen3-gpu"})


class LlamaTests(unittest.TestCase):
    def setUp(self) -> None:
        keys = ("BACKEND", "LLAMA_HF", "LLAMA_PORT", "LLAMA_ROLE", "LLAMA_IMAGE_SYCL")
        self._env = {key: os.environ.get(key) for key in keys}
        for key in self._env:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_legacy_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest=unsloth_Qwen3-4B-Instruct-2507-GGUF=Q4_K_M.json").write_text("{}")
            (root / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf").write_bytes(b"gguf")
            spec = "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"
            self.assertTrue(catalog.llama_present(root, spec))
            other = "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q8_0"
            self.assertFalse(catalog.llama_present(root, other))

    def test_hf_hub_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = root / "models--unsloth--Qwen3-4B-Instruct-2507-GGUF" / "snapshots" / "abc"
            snap.mkdir(parents=True)
            (snap / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf").write_bytes(b"gguf")
            spec = "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"
            self.assertTrue(catalog.llama_present(root, spec))

    def test_ignores_empty_and_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf").write_bytes(b"")
            (root / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf.incomplete").write_bytes(b"partial-data")
            spec = "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"
            self.assertFalse(catalog.llama_present(root, spec))

    def test_command_shapes(self) -> None:
        catalog_data = {
            "llama": [
                {
                    "name": "embed",
                    "default": True,
                    "role": "embedding",
                    "hf": "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0",
                    "port": 11440,
                    "ubatch": 2048,
                    "pooling": "last",
                },
                {
                    "name": "chat",
                    "role": "completion",
                    "hf": "org/Chat:Q4_K_M",
                    "port": 11441,
                    "extra": ["--reasoning-budget", "0"],
                },
            ]
        }
        embed = catalog.llama_command(catalog.llama_resolved(catalog_data, "embed"))
        self.assertIn("--embedding", embed)
        self.assertEqual(embed[embed.index("--pooling") + 1], "last")
        self.assertIn("-ub", embed)
        self.assertEqual(embed[-1], "8080")
        chat = catalog.llama_command(catalog.llama_resolved(catalog_data, "chat"))
        self.assertIn("--jinja", chat)
        self.assertEqual(chat[chat.index("--reasoning-budget") + 1], "0")

    def test_backend_override(self) -> None:
        data = {
            "llama": [
                {
                    "name": "chat",
                    "default": True,
                    "hf": "org/Chat:Q4_K_M",
                    "port": 11441,
                    "backend": "sycl",
                }
            ]
        }
        os.environ["BACKEND"] = "vulkan"
        spec = catalog.llama_resolved(data, "chat")
        self.assertEqual(spec["image"], catalog.LLAMA_IMAGES["vulkan"])
        self.assertEqual(spec["container"], "local-llm-serve-llama-chat-vulkan")


class OllamaTests(unittest.TestCase):
    def test_manifest_and_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = catalog.ollama_manifest(root, "qwen3-embedding:0.6b")
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}")
            self.assertFalse(catalog.ollama_present(root, "qwen3-embedding:0.6b"))
            blobs = root / "models" / "blobs"
            blobs.mkdir()
            (blobs / "sha256-abc").write_bytes(b"weights")
            self.assertTrue(catalog.ollama_present(root, "qwen3-embedding:0.6b"))


TRUNCATE_ENTRIES = [
    {"name": "bge-gpu", "truncate": True},
    {"name": "gte-gpu"},
    {"name": "bge-small-npu", "truncate": True},
]


class TruncateWarningTests(unittest.TestCase):
    def test_checked_release(self) -> None:
        warning = catalog.truncate_warning(TRUNCATE_ENTRIES, "2026.4.0")
        self.assertIn("bge-gpu, bge-small-npu", warning)
        self.assertIn("OVMS 2026.4.0 ignores it", warning)
        self.assertNotIn("gte-gpu", warning)

    def test_unchecked_release(self) -> None:
        warning = catalog.truncate_warning(TRUNCATE_ENTRIES, "2026.5.0")
        self.assertIn("2026.5.0 is unchecked", warning)
        self.assertNotIn("HTTP 400", warning)

    def test_none_without_truncate(self) -> None:
        self.assertIsNone(catalog.truncate_warning([{"name": "gte-gpu"}], "2026.4.0"))


class CatalogFileTests(unittest.TestCase):
    def test_shipped_catalog(self) -> None:
        loaded = catalog.load_catalog()
        self.assertEqual(catalog.llama_default(loaded)["name"], "qwen3-4b")
        names = [entry["name"] for entry in loaded["ovms"]]
        self.assertIn("bge-gpu", names)
        self.assertIn("gte-npu", names)
        rows = {entry["name"]: entry for entry in loaded["ovms"]}
        for name in ("bge-small-gpu", "bge-small-npu"):
            self.assertEqual(rows[name]["prepare"], "convert")
            self.assertEqual(rows[name]["source"], "BAAI/bge-small-en-v1.5")
            self.assertTrue(rows[name]["truncate"])
        self.assertEqual(rows["bge-small-npu"]["max_length"], 512)

    def test_openvino_pin_matches_default_image(self) -> None:
        self.assertEqual(catalog.openvino_pin(), "2026.4.0")
        self.assertEqual(catalog.image_release("openvino/model_server:2026.4.0-gpu"), "2026.4.0")

    def test_duplicate_name_rejected(self) -> None:
        text = """\
[[ovms]]
name = "same"
source = "OpenVINO/a"
device = "GPU"

[[llama]]
name = "same"
default = true
hf = "org/M:Q4_K_M"
port = 1
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.toml"
            path.write_text(text)
            with self.assertRaises(SystemExit):
                catalog.load_catalog(path)


if __name__ == "__main__":
    unittest.main()
