# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openvino==2026.4.0",
#   "openvino-tokenizers[transformers]==2026.4.0.0",
#   "huggingface_hub",
# ]
# ///
"""Turn the ONNX export a Hugging Face embedding model ships into an OVMS embeddings model.

    uv run --script scripts/convert.py --source Alibaba-NLP/gte-modernbert-base \
        --out ~/.local-llm-serve/ovms/gpu/converted/gte-modernbert-base \
        [--device GPU] [--pooling CLS]

For models whose repositories publish an ONNX file (ModernBERT, where optimum-intel no
longer exports). Writes the layout OVMS serves:

  openvino_model.xml/.bin        the ONNX graph, weights stored in fp16
  openvino_tokenizer.xml/.bin    the Hugging Face tokenizer, converted
  config.json                    OVMS takes the input limit from max_position_embeddings;
                                 without it the limit is 1024
  tokenizer.json and friends     the Hugging Face tokenizer beside the IR
  graph.pbtxt                    device, pooling, normalisation

Pooling is read from the repository's sentence-transformers config unless --pooling is given.
The OpenVINO pin matches the release inside OVMS_IMAGE; a tokenizer converted by another
openvino-tokenizers release may not load there. Only model files are downloaded; nothing runs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import openvino as ov
from huggingface_hub import snapshot_download

# sentence-transformers' 1_Pooling/config.json flag -> OVMS pooling mode.
POOLING = {
    "pooling_mode_cls_token": "CLS",
    "pooling_mode_mean_tokens": "MEAN",
    "pooling_mode_lasttoken": "LAST",
}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")

# Truncation off: an over-long input is rejected rather than cut.
GRAPH = """\
input_stream: "REQUEST_PAYLOAD:input"
output_stream: "RESPONSE_PAYLOAD:output"
node {{
  name: "EmbeddingsExecutor"
  input_side_packet: "EMBEDDINGS_NODE_RESOURCES:embeddings_servable"
  calculator: "EmbeddingsCalculatorOV"
  input_stream: "REQUEST_PAYLOAD:input"
  output_stream: "RESPONSE_PAYLOAD:output"
  node_options: {{
    [type.googleapis.com / mediapipe.EmbeddingsCalculatorOVOptions]: {{
      models_path: "./",
      plugin_config: '{{"NUM_STREAMS": "1" }}',
      normalize_embeddings: true,
      pooling: {pooling},
      target_device: "{device}"{max_length}
    }}
  }}
}}
"""


def pooling_of(snap: Path) -> str:
    cfg = snap / "1_Pooling" / "config.json"
    if not cfg.exists():
        sys.exit(f"{cfg.relative_to(snap)} not in the repository: pass --pooling")
    flags = json.loads(cfg.read_text())
    modes = [mode for flag, mode in POOLING.items() if flags.get(flag)]
    if len(modes) != 1:
        sys.exit(f"pooling in {cfg.name} is not one of {sorted(POOLING)}: pass --pooling")
    return modes[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", required=True, help="Hugging Face repository id")
    ap.add_argument("--revision", help="pin a commit of the repository")
    ap.add_argument("--onnx", default="onnx/model.onnx", help="ONNX file inside the repository")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--pooling", choices=sorted(set(POOLING.values())))
    ap.add_argument("--max-length", type=int, help="static input length; NPU only, below 1024")
    args = ap.parse_args()
    # OVMS 2026.4 compiles a non-Qwen model for the NPU as a static graph only below 1024
    # tokens; at 1024 or more, or with no length, it takes the path built for Qwen3.
    if args.device == "NPU" and not (args.max_length and args.max_length < 1024):
        ap.error("--device NPU needs --max-length below 1024")

    snap = Path(
        snapshot_download(
            args.source,
            revision=args.revision,
            allow_patterns=[
                args.onnx,
                f"{args.onnx}_data",
                "config.json",
                "1_Pooling/*",
                *TOKENIZER_FILES,
            ],
        )
    )
    pooling = args.pooling or pooling_of(snap)

    model = ov.convert_model(snap / args.onnx)
    inputs = {name for port in model.inputs for name in port.get_names()}
    if missing := {"input_ids", "attention_mask"} - inputs:
        sys.exit(f"ONNX inputs {sorted(inputs)} lack {sorted(missing)}, which OVMS feeds")
    if not any(
        port.get_partial_shape().rank.is_static and port.get_partial_shape().rank.get_length() == 3
        for port in model.outputs
    ):
        sys.exit("no 3-D output (last_hidden_state) for OVMS to pool")

    args.out.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, args.out / "openvino_model.xml")

    # Imported late: it pulls in transformers, which the checks above do not need.
    from openvino_tokenizers import convert_tokenizer
    from transformers import AutoTokenizer

    ov.save_model(
        convert_tokenizer(AutoTokenizer.from_pretrained(snap)), args.out / "openvino_tokenizer.xml"
    )

    for name in ("config.json", *TOKENIZER_FILES):
        if (snap / name).exists():
            shutil.copyfile(snap / name, args.out / name)

    max_length = f",\n      max_length: {args.max_length}" if args.max_length else ""
    (args.out / "graph.pbtxt").write_text(
        GRAPH.format(pooling=pooling, device=args.device, max_length=max_length)
    )

    config = json.loads((snap / "config.json").read_text())
    print(f"{args.source} -> {args.out}")
    print(f"  inputs {sorted(inputs)}, outputs {[p.get_any_name() for p in model.outputs]}")
    print(
        f"  pooling {pooling}, device {args.device}, "
        f"max_position_embeddings {config.get('max_position_embeddings')}"
    )


if __name__ == "__main__":
    main()
