#!/usr/bin/env python3
"""Export tokenizer.json vocabulary to vocab.bin for the C inference engine.

vocab.bin format (used by load_vocab() in infer.m):
  uint32  num_entries
  uint32  max_id
  For each token id 0..num_entries-1:
    uint16  byte_len
    char[byte_len]  UTF-8 bytes of the decoded token (empty for unused ids)

Handles GPT-2 ByteLevel BPE encoding: each BPE token string is first
decoded through the byte-level alphabet back to raw bytes, then stored.
This produces correct UTF-8 for all scripts (Chinese, Arabic, etc.)
and correct ASCII for Ġ→space, Ċ→newline, etc.

Usage:
    python export_vocab.py
    python export_vocab.py [tokenizer.json] [vocab.bin]
"""
import json
import os
import struct
import sys


def bytes_to_unicode():
    """GPT-2 byte-level BPE character map (from HuggingFace transformers).

    Returns a dict mapping Unicode codepoints → original byte values,
    i.e. the inverse of the encoding map.
    """
    bs = (list(range(ord('!'), ord('~') + 1)) +
          list(range(ord('¡'), ord('¬') + 1)) +
          list(range(ord('®'), ord('ÿ') + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    # cs[i] → bs[i]: Unicode codepoint cs[i] maps to byte value bs[i]
    return {c: b for c, b in zip(cs, bs)}


def decode_bpe_token(token_str, byte_decoder, is_added=False):
    """Decode a BPE token string to raw bytes.

    Added tokens (special tokens) are stored as plain UTF-8 strings
    like '<|endoftext|>' and should not be byte-decoded.
    """
    if is_added:
        return token_str.encode('utf-8')
    try:
        return bytes([byte_decoder[ord(c)] for c in token_str])
    except KeyError:
        # Fallback: return as UTF-8 (e.g. for tokens with characters
        # outside the byte-level alphabet)
        return token_str.encode('utf-8')


def main():
    tok_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        '~/.cache/modelscope/hub/models/mlx-community/Qwen3.5-122B-A10B-4bit/tokenizer.json'
    )
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'vocab.bin'

    with open(tok_path, 'r', encoding='utf-8') as f:
        t = json.load(f)

    vocab = t['model']['vocab']   # BPE-encoded str -> int
    added_tokens = t['added_tokens']  # list of {id, content, special, ...}
    added_ids = {tok['id'] for tok in added_tokens}
    added = {tok['id']: tok['content'] for tok in added_tokens}

    byte_decoder = bytes_to_unicode()

    # Build id -> raw bytes mapping; added tokens override base vocab
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

    sz = os.path.getsize(out_path)
    print(f'Wrote {out_path}')
    print(f'  Entries: {num_entries}')
    print(f'  Size: {sz / 1024:.1f} KB')

    # Spot-check a few tokens
    checks = [0, 17, 109266, 271, 248044]
    print('  Spot checks:')
    for i in checks:
        b = id_to_bytes.get(i, b'')
        try:
            s = b.decode('utf-8')
        except Exception:
            s = repr(b)
        print(f'    token {i}: {repr(s)}')


if __name__ == '__main__':
    main()
