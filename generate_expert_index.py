#!/usr/bin/env python3
"""Generate expert_index.json for repack_experts.py.

Scans the model's safetensors files and records the exact byte offset and
stride for each expert component so repack_experts.py can pread them directly.

Each layer's experts are stored as stacked tensors:
  switch_mlp.gate_proj.weight  shape [256, 1024, 384]  dtype U32
  switch_mlp.gate_proj.scales  shape [256, 1024,  48]  dtype BF16
  switch_mlp.gate_proj.biases  shape [256, 1024,  48]  dtype BF16
  switch_mlp.up_proj.weight    shape [256, 1024, 384]  dtype U32
  switch_mlp.up_proj.scales    shape [256, 1024,  48]  dtype BF16
  switch_mlp.up_proj.biases    shape [256, 1024,  48]  dtype BF16
  switch_mlp.down_proj.weight  shape [256, 2048, 64]  dtype U32
  switch_mlp.down_proj.scales  shape [256, 2048,  8]  dtype BF16
  switch_mlp.down_proj.biases  shape [256, 2048,  8]  dtype BF16

expert_stride = total_tensor_bytes / num_experts
abs_offset    = file data_start + tensor data_offsets[0] + expert_idx * expert_stride

Usage:
    python generate_expert_index.py
    python generate_expert_index.py --model /path/to/model --output expert_index.json
"""

import argparse
import json
import os
import struct
import sys


# Maps component name (as used in repack_experts.py COMPONENTS) to safetensors key suffix
COMPONENT_MAP = {
    "gate_proj.weight": "mlp.switch_mlp.gate_proj.weight",
    "gate_proj.scales": "mlp.switch_mlp.gate_proj.scales",
    "gate_proj.biases": "mlp.switch_mlp.gate_proj.biases",
    "up_proj.weight":   "mlp.switch_mlp.up_proj.weight",
    "up_proj.scales":   "mlp.switch_mlp.up_proj.scales",
    "up_proj.biases":   "mlp.switch_mlp.up_proj.biases",
    "down_proj.weight": "mlp.switch_mlp.down_proj.weight",
    "down_proj.scales": "mlp.switch_mlp.down_proj.scales",
    "down_proj.biases": "mlp.switch_mlp.down_proj.biases",
}

PREFIX      = "language_model.model.layers"


def read_safetensors_header(path):
    """Return (header_dict, data_start_offset) for a safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def dtype_bytes(dtype):
    return {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "U8": 1, "I8": 1}[dtype]


def main():
    parser = argparse.ArgumentParser(description="Generate expert_index.json for repack_experts.py")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model directory containing safetensors files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for expert_index.json (default: <model>/expert_index.json)",
    )
    args = parser.parse_args()

    model_path = os.path.realpath(args.model)
    output_path = args.output or os.path.join(model_path, "expert_index.json")

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        print(f"ERROR: {index_file} not found", file=sys.stderr)
        sys.exit(1)

    with open(index_file) as f:
        st_index = json.load(f)
    weight_map = st_index["weight_map"]  # tensor_name -> filename

    # Auto-detect NUM_LAYERS and NUM_EXPERTS from config.json
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            model_cfg = json.load(f)
        text_config = model_cfg.get("text_config", {})
        NUM_LAYERS = text_config.get("num_hidden_layers")
        NUM_EXPERTS = text_config.get("num_experts")
    else:
        print(f"ERROR: {config_path} not found, exiting.", file=sys.stderr)
        sys.exit(1)

    if NUM_LAYERS is None or NUM_EXPERTS is None:
        print(f"ERROR: Could not detect NUM_LAYERS or NUM_EXPERTS from {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Model: {model_path}")
    print(f"Layers: {NUM_LAYERS}, Experts: {NUM_EXPERTS}")

    # Load headers for all required shard files (cache to avoid re-reading)
    header_cache = {}  # filename -> (header, data_start)

    def get_header(fname):
        if fname not in header_cache:
            path = os.path.join(model_path, fname)
            print(f"  Reading header: {fname}")
            header_cache[fname] = read_safetensors_header(path)
        return header_cache[fname]

    # Build expert_reads dict
    expert_reads = {}  # layer_idx (str) -> {comp_name -> {file, abs_offset, expert_stride, expert_size}}

    print(f"Scanning {NUM_LAYERS} layers × {len(COMPONENT_MAP)} components ...\n")

    for layer_idx in range(NUM_LAYERS):
        layer_key = str(layer_idx)
        layer_prefix = f"{PREFIX}.{layer_idx}."
        layer_info = {}

        for comp_name, suffix in COMPONENT_MAP.items():
            tensor_name = layer_prefix + suffix
            if tensor_name not in weight_map:
                print(f"  WARNING: layer {layer_idx} component {comp_name} not found in index")
                continue

            fname = weight_map[tensor_name]
            header, data_start = get_header(fname)

            if tensor_name not in header:
                print(f"  WARNING: {tensor_name} not in {fname} header")
                continue

            meta = header[tensor_name]
            start, end = meta["data_offsets"]
            total_bytes = end - start
            expert_stride = total_bytes // NUM_EXPERTS
            expert_size   = expert_stride  # bytes per expert for this component

            layer_info[comp_name] = {
                "file":          fname,
                "abs_offset":    data_start + start,   # byte offset of expert 0 in the file
                "expert_stride": expert_stride,         # bytes between consecutive experts
                "expert_size":   expert_size,           # == expert_stride for stacked layout
            }

        expert_reads[layer_key] = layer_info
        print(f"  Layer {layer_idx:2d}: {len(layer_info)}/{len(COMPONENT_MAP)} components indexed")

    # Verify all layers have all components
    missing = 0
    for layer_idx in range(NUM_LAYERS):
        layer_key = str(layer_idx)
        for comp_name in COMPONENT_MAP:
            if comp_name not in expert_reads.get(layer_key, {}):
                print(f"  MISSING: layer {layer_idx}, {comp_name}")
                missing += 1
    if missing:
        print(f"\nERROR: {missing} components missing. Aborting.", file=sys.stderr)
        sys.exit(1)

    output = {
        "model_path":   model_path,
        "num_layers":   NUM_LAYERS,
        "num_experts":  NUM_EXPERTS,
        "expert_reads": expert_reads,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {output_path}")
    print(f"  Layers:     {NUM_LAYERS}")
    print(f"  Experts:    {NUM_EXPERTS}")
    print(f"  Components: {len(COMPONENT_MAP)} per layer")


if __name__ == "__main__":
    main()
