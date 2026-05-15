#!/usr/bin/env python3
"""Convert ds4 MTP GGUF to standalone deepseek4 draft GGUF for Fringe210 -md flag.

Reads the MTP GGUF (mtp.0.* tensors), merges missing weights from the main model
(token_embd, output, output_norm), renames tensors to blk.0.*, sets arch=deepseek4.

Usage:
  python3 convert_mtp_gguf.py <mtp.gguf> --donor <main.gguf> [output.gguf]

Patches from Spark test session 2026-05-15:
  P1: State machine ordering — add_tensor BEFORE write_header/kv/ti
  P2: Removed redundant write_ti_data_to_file (write_tensors_to_file calls it)
  P3: Use ReaderField.contents() for metadata extraction
  P4: --donor flag merges token_embd/output/output_norm from main model
"""
import sys, os, numpy as np
from collections import OrderedDict
from typing import Any

for candidate in [
    os.path.expanduser("~/llama-dsv4-fringe/gguf-py"),
    "/tmp/fringe-analysis/gguf-py",
]:
    if candidate not in sys.path and os.path.isdir(candidate):
        sys.path.insert(0, candidate)
        break

try:
    from gguf import GGUFReader, GGUFValueType
    from gguf.gguf_writer import GGUFWriter
except ImportError:
    print("Error: gguf-py not found. Set PYTHONPATH to Fringe210/gguf-py")
    sys.exit(1)

DONOR_TENSORS = ["token_embd.weight", "output_norm.weight", "output.weight"]


def convert(input_path: str, donor_path: str, output_path: str):
    print(f"MTP GGUF:  {input_path}")
    reader = GGUFReader(input_path, 'r')

    # ---- metadata ----
    metadata = OrderedDict()
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        try:
            metadata[key] = (field.contents(), field.types)
        except Exception as e:
            print(f"  WARNING: key {key}: {e}")

    # Force block_count to 1 (MTP is single-layer)
    block_count_key = None
    for key in metadata:
        if key.endswith(".block_count"):
            block_count_key = key
            break
    if block_count_key:
        metadata[block_count_key] = (1, metadata[block_count_key][1])
        print(f"Forced {block_count_key} = 1")

    arch = "deepseek4"
    print(f"Target arch: {arch}")
    print(f"MTP tensors: {len(reader.tensors)}")

    # ---- open donor ----
    donor = None
    if donor_path and os.path.exists(donor_path):
        print(f"Donor GGUF: {donor_path}")
        donor = GGUFReader(donor_path, 'r')
        donor_tensors = {t.name: t for t in donor.tensors}
        for name in DONOR_TENSORS:
            if name in donor_tensors:
                t = donor_tensors[name]
                print(f"  Found: {name}  {list(t.shape)}  {t.tensor_type.name}  {t.n_bytes} bytes")
            else:
                print(f"  WARNING: {name} not found in donor GGUF")

    # ---- build writer ----
    writer = GGUFWriter(output_path, arch)

    # Copy metadata (skip general.architecture, writer sets it)
    for key, (val, types) in metadata.items():
        if key == "general.architecture":
            continue
        main_type = types[0]
        try:
            if main_type == GGUFValueType.STRING:
                writer.add_string(key, val)
            elif main_type == GGUFValueType.UINT32:
                writer.add_uint32(key, int(val))
            elif main_type == GGUFValueType.UINT64:
                writer.add_uint64(key, int(val))
            elif main_type == GGUFValueType.INT32:
                writer.add_int32(key, int(val))
            elif main_type == GGUFValueType.INT64:
                writer.add_int64(key, int(val))
            elif main_type == GGUFValueType.FLOAT32:
                writer.add_float32(key, float(val))
            elif main_type == GGUFValueType.FLOAT64:
                writer.add_float64(key, float(val))
            elif main_type == GGUFValueType.BOOL:
                writer.add_bool(key, bool(val))
            elif main_type == GGUFValueType.ARRAY:
                writer.add_array(key, val)
            elif main_type == GGUFValueType.UINT8:
                writer.add_uint8(key, int(val))
            elif main_type == GGUFValueType.UINT16:
                writer.add_uint16(key, int(val))
            elif main_type == GGUFValueType.INT8:
                writer.add_int8(key, int(val))
            elif main_type == GGUFValueType.INT16:
                writer.add_int16(key, int(val))
            else:
                print(f"  SKIP: {key} type={main_type}")
        except Exception as e:
            print(f"  ERROR: {key}: {e}")

    # ---- add tensors (P1: BEFORE write_*_to_file) ----
    print(f"\nAdding tensors...")

    # First, donor tensors (token_embd, output_norm, output)
    if donor:
        donor_tensors = {t.name: t for t in donor.tensors}
        for name in DONOR_TENSORS:
            if name in donor_tensors:
                t = donor_tensors[name]
                writer.add_tensor(name, t.data, raw_dtype=t.tensor_type)
                print(f"  [+donor] {name}  {list(t.shape)}  {t.n_bytes} bytes")

    # Then, remapped MTP tensors
    for i, rt in enumerate(reader.tensors):
        new_name = rt.name
        if rt.name.startswith("mtp.0."):
            new_name = "blk.0." + rt.name[len("mtp.0."):]
        if new_name != rt.name:
            print(f"  [{i}] {rt.name} -> {new_name}")
        writer.add_tensor(new_name, rt.data, raw_dtype=rt.tensor_type)

    # ---- write file (P1+P2: correct order, no redundant TI write) ----
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)

    print(f"\nDone: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(1)

    inp = sys.argv[1]
    donor = None
    out = "mtp-converted-deepseek4.gguf"
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--donor" and i + 1 < len(args):
            donor = args[i + 1]
            i += 2
        elif not args[i].startswith("--"):
            out = args[i]
            i += 1
        else:
            i += 1

    if not os.path.exists(inp):
        print(f"Error: MTP GGUF not found: {inp}")
        sys.exit(1)
    if donor and not os.path.exists(donor):
        print(f"Error: donor GGUF not found: {donor}")
        sys.exit(1)
    if not donor:
        print("Warning: no --donor specified. token_embd/output weights will be missing.")
        print("The draft model will fail to load without these.")

    convert(inp, donor, out)
