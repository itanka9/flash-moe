#!/usr/bin/env python3
"""
Unified model preparation: wraps steps 1-4 from usage.txt into one command.

Takes a source model directory (e.g. mlx-community/Qwen3.5-122B-A10B-4bit) and:
  1. Generates expert_index.json (expert byte offsets)
  2. Extracts non-expert weights → model_weights.bin + model_weights.json
  3. Exports tokenizer → tokenizer.bin + vocab.bin
  4. Repacks expert weights → packed_experts/layer_XX.bin + layout.json

All artifacts go into a single output directory ready for the C inference engine.

Usage:
    # Minimal: output dir auto-derived from model name
    python prepare_model.py --model /path/to/Qwen3.x-Model-4bit

    # Custom output directory
    python prepare_model.py --model /path/to/model --output-dir /path/to/out

    # Skip expert repacking (fast, for debugging)
    python prepare_model.py --model /path/to/model --no-repack --light

    # Dry run (just print what would happen)
    python prepare_model.py --model /path/to/model --dry-run

Invoke from the flash-moe project root (or any dir in the project tree).
"""

import argparse
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path


# ============================================================================
# Configuration: components and constants
# ============================================================================

# The repack expert scripts use these component names
COMPONENT_ORDER = [
    "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
    "up_proj.weight",   "up_proj.scales",   "up_proj.biases",
    "down_proj.weight",  "down_proj.scales",  "down_proj.biases",
]

# Component → safetensors key suffix (for expert indexing)
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

PREFIX = "language_model.model.layers"

CHUNK_LABEL = "language_model"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified model preparation (steps 1-4 for flash-moe inference)"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to source model directory (e.g. Qwen3.5-122B-A10B-4bit)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for all prepared artifacts (default: ./prepared/<model_name>/)",
    )
    parser.add_argument(
        "--repack-layers", default=None,
        help="Expert repack layer range: 'all', '0-4', '0,5,10' (default: all)",
    )
    parser.add_argument(
        "--no-repack", action="store_true",
        help="Skip expert repacking — for quick setup or debugging",
    )
    parser.add_argument(
        "--light", action="store_true",
        help="Skip large expert repack + tokenizer; only non-expert weights + index",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing anything",
    )
    return parser.parse_args()


def dtype_bytes(dtype):
    return {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "U8": 1, "I8": 1}.get(dtype, 4)


def parse_safetensors_header(filepath):
    """Return (header_dict, data_start_offset) for a safetensors file."""
    with open(filepath, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def parse_layers(spec, num_layers):
    """Parse layer specification like '0-4' or '0,5,10' or 'all'."""
    if spec is None or spec == 'all':
        return list(range(num_layers))
    layers = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


# ============================================================================
# Step 1: Generate expert_index.json
# ============================================================================

def step1_generate_expert_index(model_path, output_dir, dry_run):
    """Scan safetensors and build expert_index.json."""
    index_path = output_dir / "expert_index.json"

    if dry_run:
        print(f"  [1] Would scan {model_path} for expert offsets → {index_path}")
        return True

    print("[1] Generating expert index...")
    t0 = time.monotonic()

    # Load model config
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        print(f"  ERROR: {cfg_path} not found", file=sys.stderr)
        return False
    with open(cfg_path) as f:
        model_cfg = json.load(f)
    text_cfg = model_cfg.get("text_config", {})
    num_layers = text_cfg.get("num_hidden_layers")
    num_experts = text_cfg.get("num_experts")
    if not num_layers or not num_experts:
        print(f"  ERROR: could not detect num_layers/num_experts from {cfg_path}",
              file=sys.stderr)
        return False

    print(f"  Model: {model_path.name}, {num_layers} layers, {num_experts} experts")

    # Load weight map
    index_pt = model_path / "model.safetensors.index.json"
    if not index_pt.exists():
        print(f"  ERROR: {index_pt} not found", file=sys.stderr)
        return False
    with open(index_pt) as f:
        st_index = json.load(f)
    weight_map = st_index["weight_map"]

    # Cache safetensors headers
    header_cache = {}

    def get_header(fname):
        if fname not in header_cache:
            fp = model_path / fname
            h, ds = parse_safetensors_header(str(fp))
            header_cache[fname] = (h, ds)
        return header_cache[fname]

    expert_reads = {}
    for layer_idx in range(num_layers):
        layer_key = str(layer_idx)
        layer_prefix = f"{PREFIX}.{layer_idx}."
        layer_info = {}

        for comp_name, suffix in COMPONENT_MAP.items():
            tensor_name = layer_prefix + suffix
            if tensor_name not in weight_map:
                continue
            fname = weight_map[tensor_name]
            if fname not in header_cache:
                h, ds = get_header(fname)
            header, data_start = header_cache[fname]

            if tensor_name not in header:
                continue
            meta = header[tensor_name]
            start, end = meta["data_offsets"]
            total_bytes = end - start
            expert_stride = total_bytes // num_experts
            expert_size = expert_stride

            layer_info[comp_name] = {
                "file": fname,
                "abs_offset": data_start + start,
                "expert_stride": expert_stride,
                "expert_size": expert_size,
            }

        expert_reads[layer_key] = layer_info
        if (layer_idx + 1) % 10 == 0 or layer_idx == num_layers - 1:
            print(f"    [{layer_idx+1}/{num_layers}]", flush=True)

    # Verify all components present
    missing = 0
    for layer_idx in range(num_layers):
        lk = str(layer_idx)
        for cn in COMPONENT_MAP:
            if cn not in expert_reads.get(lk, {}):
                missing += 1
    if missing:
        print(f"  ERROR: {missing} component(s) missing across all layers", file=sys.stderr)
        return False

    output = {
        "model_path": str(model_path.resolve()),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "expert_reads": expert_reads,
    }
    with open(index_path, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.monotonic() - t0
    print(f"  -> {index_path.name}: {num_layers} layers, {num_experts} experts ({elapsed:.1f}s)")
    return True


# ============================================================================
# Step 2: Extract non-expert weights
# ============================================================================

def step2_extract_weights(model_path, output_dir, dry_run):
    """Extract all non-expert weights to model_weights.bin + model_weights.json."""
    bin_path = output_dir / "model_weights.bin"
    json_path = output_dir / "model_weights.json"

    if dry_run:
        print(f"  [2] Would extract non-expert weights → {bin_path}")
        return True

    print("[2] Extracting non-expert weights...")
    t0 = time.monotonic()

    # Load model config
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        print(f"  ERROR: {cfg_path} not found", file=sys.stderr)
        return False
    with open(cfg_path) as f:
        model_cfg = json.load(f)
    text_cfg = model_cfg.get("text_config", {})
    hidden_size = text_cfg.get("hidden_size", 4096)
    num_layers = text_cfg.get("num_hidden_layers", 60)
    num_attn_heads = text_cfg.get("num_attention_heads", 32)
    num_kv_heads = text_cfg.get("num_key_value_heads", 2)
    head_dim = text_cfg.get("head_dim", 256)
    vocab_size = text_cfg.get("vocab_size", 248320)
    num_experts = text_cfg.get("num_experts", 512)
    num_experts_per_tok = text_cfg.get("num_experts_per_tok", 10)
    moe_intermediate = text_cfg.get("moe_intermediate_size", 1024)
    shared_intermediate = text_cfg.get("shared_expert_intermediate_size", moe_intermediate)
    full_attn_interval = text_cfg.get("full_attention_interval", 4)
    rms_norm_eps = text_cfg.get("rms_norm_eps", 1e-6)
    rope_theta = text_cfg.get("rope_theta", 10000000.0)
    partial_rotary = text_cfg.get("partial_rotary_factor", 0.25)
    linear_num_v_heads = text_cfg.get("linear_num_value_heads", 64)
    linear_num_k_heads = text_cfg.get("linear_num_key_heads", 16)
    linear_key_dim = text_cfg.get("linear_key_head_dim", 128)
    linear_value_dim = text_cfg.get("linear_value_head_dim", 128)
    linear_conv_kernel = text_cfg.get("linear_conv_kernel_dim", 4)

    # Detect gate quantization from quantization_config
    quant_cfg = model_cfg.get("quantization_config", model_cfg.get("quantization", {}))
    gate_bits = int(quant_cfg.get("bits", 4))

    # Load weight index
    index_pt = model_path / "model.safetensors.index.json"
    if not index_pt.exists():
        print(f"  ERROR: {index_pt} not found", file=sys.stderr)
        return False
    with open(index_pt) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    # Filter: skip experts, skip vision
    expert_pat = __import__("re").compile(r"\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)")
    vision_pat = __import__("re").compile(r"^(vision_tower|model\.visual)")

    def keep(name):
        if vision_pat.match(name):
            return False
        if expert_pat.search(name):
            return False
        return True

    # Collect tensors to extract
    tensors = {n: f for n, f in weight_map.items() if keep(n)}

    # Group by shard file
    by_file = {}
    for name, fname in tensors.items():
        by_file.setdefault(fname, []).append(name)

    print(f"  {len(tensors)} non-expert tensors, {len(by_file)} shard files")

    # Build layer type map
    layer_types = []
    for i in range(num_layers):
        layer_types.append("full_attention" if (i + 1) % full_attn_interval == 0 else "linear_attention")

    manifest_data = {
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_attn_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "rms_norm_eps": rms_norm_eps,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "moe_intermediate_size": moe_intermediate,
        "shared_expert_intermediate_size": shared_intermediate,
        "full_attention_interval": full_attn_interval,
        "linear_num_value_heads": linear_num_v_heads,
        "linear_num_key_heads": linear_num_k_heads,
        "linear_key_head_dim": linear_key_dim,
        "linear_value_head_dim": linear_value_dim,
        "linear_conv_kernel_dim": linear_conv_kernel,
        "partial_rotary_factor": partial_rotary,
        "rope_theta": rope_theta,
        "gate_bits": gate_bits,
        "layer_types": layer_types,
    }

    manifest = {"tensors": {}, "config": manifest_data, "num_tensors": len(tensors)}

    # Read + write binary
    ALIGN = 64
    offset = 0

    with open(bin_path, "wb") as out:
        for fname in sorted(by_file.keys()):
            filepath = model_path / fname
            hdr, data_start = parse_safetensors_header(str(filepath))

            tensor_names = sorted(by_file[fname], key=lambda t: weight_map[t])

            for tname in tensor_names:
                if tname not in hdr:
                    continue
                meta = hdr[tname]
                start_off, end_off = meta["data_offsets"]
                byte_len = end_off - start_off
                shape = meta["shape"]
                dtype = meta["dtype"]

                # Align
                if offset % ALIGN != 0:
                    pad = ALIGN - (offset % ALIGN)
                    out.write(b"\x00" * pad)
                    offset += pad

                # Read from safetensors
                with open(filepath, "rb") as sf:
                    sf.seek(data_start + start_off)
                    data = sf.read(byte_len)

                # Special: convert A_log from BF16 → F32
                san_name = tname[len(f"{CHUNK_LABEL}."):] if tname.startswith(f"{CHUNK_LABEL}.") else tname
                if san_name.endswith(".linear_attn.A_log") and dtype == "BF16":
                    n = len(data) // 2
                    f32_data = struct.pack(f"{n}f", *[
                        struct.unpack("f",
                                     struct.pack("I",
                                                   int.from_bytes(data[i*2:i*2+2],
                                                                   "little") << 16))[0]
                        for i in range(n)
                    ])
                    data = f32_data
                    byte_len = len(data)
                    dtype = "F32"

                out.write(data)

                manifest["tensors"][san_name] = {
                    "offset": offset, "size": byte_len, "shape": shape, "dtype": dtype,
                }
                offset += byte_len

    elapsed = time.monotonic() - t0
    print(f"  -> {bin_path.name}: {offset/1e9:.2f} GB in {elapsed:.1f}s")

    # Write JSON manifest
    manifest["num_tensors"] = len(tensors)
    with open(json_path, "w") as f:
        json.dump({
            "model": str(model_path),
            "num_tensors": len(tensors),
            "tensors": {k: v for k, v in manifest["tensors"].items()},
            "config": manifest_data,
        }, f, indent=2)

    # Summary by category
    cats = {}
    for san_name, info in manifest["tensors"].items():
        cat = "other"
        if "embed_tokens" in san_name:
            cat = "embedding"
        elif "norm.weight" in san_name and "layers." not in san_name:
            cat = "final_norm"
        elif "lm_head" in san_name:
            cat = "lm_head"
        elif "input_layernorm" in san_name or "post_attention_layernorm" in san_name:
            cat = "layer_norms"
        elif "linear_attn" in san_name:
            cat = "linear_attention"
        elif "self_attn" in san_name:
            cat = "full_attention"
        elif "mlp.gate." in san_name:
            cat = "routing_gate"
        elif "shared_expert." in san_name:
            cat = "shared_expert"
        elif "shared_expert_gate" in san_name:
            cat = "shared_expert_gate"
        elif "switch_mlp" in san_name:
            cat = "routed_experts"
        cats.setdefault(cat, {"count": 0, "bytes": 0})
        cats[cat]["count"] += 1
        cats[cat]["bytes"] += info["size"]
    print("  Weights by category:")
    for cat in sorted(cats):
        c = cats[cat]
        print(f"    {cat:25s}: {c['count']:3d} tensors, {c['bytes']/1e6:7.1f} MB")

    return True


# ============================================================================
# Step 3a: Export tokenizer to binary (tokenizer.bin)
# ============================================================================

def bytes_to_unicode():
    """GPT-2 byte-level BPE character map (from HuggingFace transformers)."""
    bs = (
        list(range(ord('!'), ord('~') + 1)) +
        list(range(ord('¡'), ord('¬') + 1)) +
        list(range(ord('®'), ord('ÿ') + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {c: b for c, b in zip(cs, bs)}


def decode_bpe_token(token_str, byte_decoder, is_added=False):
    """Decode a BPE token string to raw bytes."""
    if is_added:
        return token_str.encode('utf-8')
    try:
        return bytes([byte_decoder[ord(c)] for c in token_str])
    except KeyError:
        return token_str.encode('utf-8')


def step3a_export_tokenizer(model_path, output_dir, dry_run):
    """Export tokenizer.json → tokenizer.bin (BPE trie for C engine)."""
    out_path = output_dir / "tokenizer.bin"

    if dry_run:
        print(f"  [3a] Would export tokenizer → {out_path}")
        return True

    print("[3a] Exporting tokenizer...")
    t0 = time.monotonic()

    # Locate tokenizer.json
    tok_path = None
    for p in [model_path / "tokenizer.json", output_dir / "tokenizer.json"]:
        if p.exists():
            tok_path = p
            break
    if not tok_path:
        print(f"  ERROR: tokenizer.json not found", file=sys.stderr)
        return False

    with open(tok_path, 'r', encoding='utf-8') as f:
        t = json.load(f)

    model = t['model']
    vocab = model['vocab']  # str -> int
    merges = model['merges']  # list of [str, str]
    added = t['added_tokens']  # list of {id, content, special, ...}

    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    with open(out_path, 'wb') as f:
        f.write(b'BPET')
        f.write(struct.pack('<I', 1))  # version
        f.write(struct.pack('<I', len(sorted_vocab)))
        f.write(struct.pack('<I', len(merges)))
        f.write(struct.pack('<I', len(added)))

        # Vocab
        for token_str, token_id in sorted_vocab:
            b = token_str.encode('utf-8')
            f.write(struct.pack('<I', token_id))
            f.write(struct.pack('<H', len(b)))
            f.write(b)

        # Merges
        for pair in merges:
            a, b = pair[0], pair[1]
            ab = a.encode('utf-8')
            bb = b.encode('utf-8')
            f.write(struct.pack('<H', len(ab)))
            f.write(ab)
            f.write(struct.pack('<H', len(bb)))
            f.write(bb)

        # Added tokens
        for tok in added:
            b = tok['content'].encode('utf-8')
            f.write(struct.pack('<I', tok['id']))
            f.write(struct.pack('<H', len(b)))
            f.write(b)

    elapsed = time.monotonic() - t0
    print(f"  -> {out_path.name}: {len(sorted_vocab)} vocab, {len(merges)} merges, "
          f"{len(added)} added ({elapsed:.2f}s)")
    return True


# ============================================================================
# Step 3b: Export vocab to binary (vocab.bin)
# ============================================================================

def step3b_export_vocab(model_path, output_dir, dry_run):
    """Export tokenizer.json → vocab.bin (token id → string decoder)."""
    out_path = output_dir / "vocab.bin"

    if dry_run:
        print(f"  [3b] Would export vocab → {out_path}")
        return True

    print("[3b] Exporting vocabulary...")
    t0 = time.monotonic()

    # Locate tokenizer.json
    tok_path = None
    for p in [model_path / "tokenizer.json", output_dir / "tokenizer.json"]:
        if p.exists():
            tok_path = p
            break
    if not tok_path:
        print(f"  ERROR: tokenizer.json not found", file=sys.stderr)
        return False

    with open(tok_path, 'r', encoding='utf-8') as f:
        t = json.load(f)

    vocab = t['model']['vocab']  # BPE-encoded str -> int
    added_tokens = t.get('added_tokens', [])
    added_ids = {tok['id'] for tok in added_tokens}
    added = {tok['id']: tok['content'] for tok in added_tokens}

    byte_decoder = bytes_to_unicode()

    # Build id -> raw bytes
    id_to_bytes = {}
    for s, i in vocab.items():
        id_to_bytes[i] = decode_bpe_token(s, byte_decoder, is_added=False)
    for i, s in added.items():
        id_to_bytes[i] = decode_bpe_token(s, byte_decoder, is_added=True)

    num_entries = max(id_to_bytes.keys()) + 1
    max_id = num_entries - 1

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<I', num_entries))
        f.write(struct.pack('<I', max_id))
        for i in range(num_entries):
            b = id_to_bytes.get(i, b'')
            f.write(struct.pack('<H', len(b)))
            f.write(b)

    elapsed = time.monotonic() - t0
    print(f"  -> {out_path.name}: {num_entries} entries ({elapsed:.2f}s)")
    return True


# ============================================================================
# Step 4: Repack expert weights
# ============================================================================

def step4_repack_experts(model_path, output_dir, layers_spec, dry_run):
    """Repack expert weights into contiguous per-layer binary files."""
    if dry_run:
        print(f"  [3] Would repack experts → {output_dir / 'packed_experpts'}/")
        return True

    print("[3] Repacking experts...")
    t0 = time.monotonic()

    index_path = output_dir / "expert_index.json"
    if not index_path.exists():
        print(f"  ERROR: {index_path} not found — run step 1 first", file=sys.stderr)
        return False

    with open(index_path) as f:
        idx = json.load(f)
    expert_reads = idx["expert_reads"]
    src_model = Path(idx["model_path"])
    num_experts = idx["num_experts"]
    num_layers = idx["num_layers"]

    # Auto-detect layout
    components = []
    off = 0
    for cn in COMPONENT_ORDER:
        l0 = expert_reads.get("0", {})
        if cn not in l0:
            print(f"  ERROR: {cn} not in layer 0", file=sys.stderr)
            return False
        sz = l0[cn]["expert_size"]
        components.append({"name": cn, "offset": off, "size": sz})
        off += sz
    expert_size = off

    print(f"  {num_layers} layers, {num_experts} experts, {expert_size:,} bytes/expert")

    # Determine layers to process
    if layers_spec:
        layers = parse_layers(layers_spec, num_layers)
    else:
        layers = list(range(num_layers))

    total_bytes = len(layers) * num_experts * expert_size
    print(f"  Total: {total_bytes/1e12:.1f} TB")

    # Check disk space
    out_dir = output_dir / "packed_experts"
    out_dir.mkdir(parents=True, exist_ok=True)
    stat = __import__("os").statvfs(str(output_dir))
    free_gb = stat.f_bavail * stat.f_frsize / 1e9
    needed_gb = total_bytes / 1e9
    print(f"  Free: {free_gb:.0f} GB, needed: {needed_gb:.0f} GB")
    if free_gb < needed_gb:
        print(f"  WARNING: not enough space", file=sys.stderr)

    # Open source files
    needed = set()
    for l in layers:
        lk = str(l)
        for info in expert_reads.get(lk, {}).values():
            needed.add(info["file"])
    fds = {}
    for fn in sorted(needed):
        p = str(src_model / fn)
        fds[fn] = __import__("os").open(p, __import__("os").O_RDONLY)
    print(f"  Opened {len(fds)} source files")

    # Write layout metadata
    with open(out_dir / "layout.json", "w") as f:
        json.dump({
            "expert_size": expert_size,
            "num_layers": num_layers,
            "num_experts": num_experts,
            "components": components,
        }, f, indent=2)

    t_start = time.monotonic()
    total_written = 0

    for i, layer_idx in enumerate(layers):
        lk = str(layer_idx)
        li = expert_reads.get(lk, {})
        if not li:
            print(f"  Layer {layer_idx}: not in index, skipping")
            continue

        layer_path = out_dir / f"layer_{layer_idx:02d}.bin"
        layer_size = num_experts * expert_size

        # Pre-allocate
        fd = __import__("os").open(str(layer_path), __import__("os").O_RDWR | __import__("os").O_CREAT | __import__("os").O_TRUNC, 0o644)
        __import__("os").ftruncate(fd, layer_size)

        # Build read plan
        plan = []
        for e in range(num_experts):
            for c in components:
                info = li[c["name"]]
                sf = fds[info["file"]]
                src_off = info["abs_offset"] + e * info["expert_stride"]
                dst_off = e * expert_size + c["offset"]
                plan.append((sf, src_off, dst_off, c["size"]))

        plan.sort(key=lambda x: (x[0], x[1]))

        written = 0
        for sf, src_off, dst_off, sz in plan:
            data = __import__("os").pread(sf, sz, src_off)
            __import__("os").pwrite(fd, data, dst_off)
            written += sz
        total_written += written
        __import__("os").close(fd)

        if (i + 1) % 5 == 0:
            elapsed_t = time.monotonic() - t_start
            print(f"    [{i+1}/{len(layers)}] {total_written/1e9:.1f} GB ({elapsed_t:.0f}s)", flush=True)

    elapsed = time.monotonic() - t0
    print(f"  -> {out_dir}/: {total_written/1e9:.1f} GB in {elapsed:.1f}s")

    return True


# ============================================================================
# Main entry
# ============================================================================

def main():
    args = parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"ERROR: {model_path} not found", file=sys.stderr)
        sys.exit(1)

    # Determine output dir
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = Path("prepared") / model_path.name

    dry_run = args.dry_run

    print("=" * 70)
    print(f"  Flash-MoE: Unified Model Preparation")
    print(f"  Source: {model_path}")
    print(f"  Output: {output_dir}")
    if dry_run:
        print("  Mode:   DRY RUN (no files written)")
    print("=" * 70)

    # Create output dir
    if not dry_run:
        if output_dir.exists():
            ans = input(f"  {output_dir} exists. Overwrite? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                print("  Aborted.")
                sys.exit(0)
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Copy model config + weight index (used by inference)
    for fn in ["config.json", "model.safetensors.index.json"]:
        src = model_path / fn
        if src.exists():
            if not dry_run:
                shutil.copy2(str(src), str(output_dir / fn))
            print(f"  Copied {fn} -> {output_dir / fn}")

    # Copy tokenizer.json for vocab export
    tok_src = model_path / "tokenizer.json"
    if tok_src.exists():
        if not dry_run:
            shutil.copy2(str(tok_src), str(output_dir / "tokenizer.json"))
        print(f"  Copied tokenizer.json -> {output_dir / 'tokenizer.json'}")
    else:
        # Maybe it's in a subdirectory
        for p in model_path.rglob("tokenizer.json"):
            if not dry_run:
                shutil.copy2(str(p), str(output_dir / "tokenizer.json"))
            print(f"  Found tokenizer.json at {p}")
            break

    print()

    t_start = time.monotonic()

    # Step 1: expert index
    if not step1_generate_expert_index(model_path, output_dir, dry_run):
        sys.exit(1)

    # Step 2: non-expert weights (skip if --light or dry-run)
    if not args.light:
        if not step2_extract_weights(model_path, output_dir, dry_run):
            sys.exit(1)

    # Step 3a: export tokenizer (always, even in dry-run for file count)
    if not args.light:
        if not step3a_export_tokenizer(model_path, output_dir, dry_run):
            sys.exit(1)

    # Step 3b: export vocab (always)
    if not args.light:
        if not step3b_export_vocab(model_path, output_dir, dry_run):
            sys.exit(1)

    # Step 4: repack experts (skip if --no-repack or --light or dry-run)
    if not args.no_repack and not args.light:
        if not step4_repack_experts(model_path, output_dir, args.repack_layers, dry_run):
            sys.exit(1)

    elapsed = time.monotonic() - t_start

    print()
    print("=" * 70)
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Output: {output_dir}")
    print()

    # List artifacts
    print("  Artifacts:")
    for f in sorted(output_dir.iterdir()):
        if f.is_dir():
            total = sum(p.stat().st_size for p in f.rglob("*") if p.is_file())
            print(f"    {f.name}/: {total/1e9:.1f} GB")
            continue
        sz = f.stat().st_size
        if sz > 1e9:
            print(f"    {f.name:35s} {sz/1e9:.2f} GB")
        elif sz > 1e6:
            print(f"    {f.name:35s} {sz/1e6:.2f} MB")
        elif sz > 1e3:
            print(f"    {f.name:35s} {sz/1e3:.1f} KB")
        else:
            print(f"    {f.name:35s} {sz} B")
    print("=" * 70)

    # Inference usage hint
    if not dry_run:
        print()
        print("  To run inference:")
        print(f"    cd metal_infer && make")
        print(f"    ./infer --model {output_dir} --prompt 'Hello' --tokens 20")


if __name__ == "__main__":
    main()